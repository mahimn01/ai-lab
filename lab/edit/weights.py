"""Read, change and record a model's weights; turn a weight change into a small LoRA file.

Generalised from neural-harness (harness/model.py, delta_to_lora.py, sd1/edits.py), where
the same steps showed the "Astra" edit of Qwen3.5-0.8B is a rank-1 change that fits in
a 37 KB file.

Plain-words glossary
  weight       one of the model's number tables, found by a dotted path such as
               "language_model.model.layers.15.mlp.down_proj" (use `list_weights` to see them)
  delta        the change to a weight: edited minus original
  rank         how many (column x row) products are needed to write the delta; a
               low-rank delta is tiny to store
  LoRA         storing a delta as two thin tables A and B, added to the layer at run
               time, so the original model file stays untouched
  checksum     sha256 of the raw bytes: proof of exactly which numbers were used

Every function is in-memory only. `save_lora` and `EditLog.save` are the only writers.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.tuner.lora import LoRALinear


def list_weights(model: nn.Module, contains: str = "") -> list[tuple[str, tuple[int, ...], str]]:
    """(path, shape, number format) for every weight whose path contains the text."""
    return [(k, tuple(v.shape), str(v.dtype)) for k, v in tree_flatten(model.parameters()) if contains in k]


def get_module(model: nn.Module, path: str) -> tuple[nn.Module, str]:
    """(parent module, attribute name) for a dotted module path such as 'blocks.3.fc'."""
    parts = path.split(".")
    obj = model
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    return obj, parts[-1]


def get_linear(model: nn.Module, path: str) -> nn.Module:
    parent, name = get_module(model, path)
    return parent[int(name)] if name.isdigit() else getattr(parent, name)


def get_weight(model: nn.Module, path: str) -> np.ndarray:
    """The layer's weight as float32 numpy, shape [out, in]. Quantized layers cannot be read this way."""
    layer = get_linear(model, path)
    if isinstance(layer, nn.QuantizedLinear):
        raise TypeError(f"{path} is quantized (packed into few bits). Load the full-precision model to "
                        "read or change its numbers, or use attach_lora, which works on quantized layers.")
    return np.array(layer.weight.astype(mx.float32))


def checksum(weight: mx.array) -> str:
    """sha256 of the weight's raw bytes in its own number format."""
    # Read the bits as unsigned integers of the same width: numpy has no bfloat16 type.
    bits = {1: mx.uint8, 2: mx.uint16, 4: mx.uint32, 8: mx.uint64}[weight.itemsize]
    return hashlib.sha256(np.array(weight.view(bits)).tobytes()).hexdigest()


@dataclass
class EditRecord:
    path: str
    scale: float
    sha256_before: str
    sha256_after: str
    delta_norm: float  # size of the change (Frobenius norm)
    relative_change: float  # delta_norm / size of the original weight
    note: str = ""


@dataclass
class EditLog:
    """A list of every edit applied, so a result can name exactly which weights it used."""

    model: str
    edits: list[EditRecord] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")


def apply_delta(model: nn.Module, path: str, delta: np.ndarray, scale: float = 1.0,
                log: EditLog | None = None, note: str = "") -> EditRecord:
    """Add scale * delta to the weight at `path`, in place. The sum is done in float32, then rounded once."""
    layer = get_linear(model, path)
    w = layer.weight
    if isinstance(layer, nn.QuantizedLinear):
        raise TypeError(f"{path} is quantized; apply_delta needs the full-precision model (or use attach_lora)")
    if tuple(delta.shape) != tuple(w.shape):
        raise ValueError(f"delta shape {delta.shape} does not match {path} shape {tuple(w.shape)}")
    before = checksum(w)
    w32 = w.astype(mx.float32)
    layer.weight = (w32 + scale * mx.array(delta, dtype=mx.float32)).astype(w.dtype)
    mx.eval(layer.weight)
    norm = float(np.linalg.norm(delta)) * abs(scale)
    record = EditRecord(path, scale, before, checksum(layer.weight), norm,
                        norm / float(np.linalg.norm(np.array(w32))), note)
    if log is not None:
        log.edits.append(record)
    return record


def delta_to_lora(delta: np.ndarray, rank: int | None = None, energy: float = 0.9999) -> tuple[np.ndarray, np.ndarray, dict]:
    """Factor a weight change [out, in] into LoRA tables in mlx-lm's layout.

    Returns (lora_a [in, r], lora_b [r, out], info), where delta is approximately
    (lora_a @ lora_b).T. If `rank` is not given, the smallest rank that keeps `energy`
    of the change's squared size is used.
    """
    u, s, vt = np.linalg.svd(delta.astype(np.float64), full_matrices=False)
    kept = np.cumsum(s**2) / np.sum(s**2)
    if rank is None:
        rank = int(np.searchsorted(kept, energy) + 1)
    root = np.sqrt(s[:rank])
    lora_b = (u[:, :rank] * root).T.astype(np.float32)  # [r, out]
    lora_a = (root[:, None] * vt[:rank]).T.astype(np.float32)  # [in, r]
    info = {"rank": rank, "energy_kept": float(kept[rank - 1]), "top_singular_values": s[:rank + 3].tolist(),
            "relative_residual": float(np.linalg.norm(delta - (lora_a @ lora_b).T) / np.linalg.norm(delta))}
    return lora_a, lora_b, info


def attach_lora(model: nn.Module, path: str, lora_a: np.ndarray, lora_b: np.ndarray, scale: float = 1.0) -> Callable[[], None]:
    """Run the layer as original + scale * LoRA without touching the original weight. Returns an undo function.

    Works on quantized layers too, so an edit found on the full model can be tried on its 4-bit version.
    """
    parent, name = get_module(model, path)
    original = get_linear(model, path)
    lora = LoRALinear.from_base(original, r=lora_a.shape[1], dropout=0.0, scale=scale)
    lora.lora_a, lora.lora_b = mx.array(lora_a), mx.array(lora_b)

    def put(layer: nn.Module) -> None:
        if name.isdigit():
            parent[int(name)] = layer
        else:
            setattr(parent, name, layer)

    put(lora)
    return lambda: put(original)


def save_lora(directory: str | Path, tables: dict[str, tuple[np.ndarray, np.ndarray]], info: dict) -> Path:
    """Write {module path: (lora_a, lora_b)} as adapters.safetensors plus a small JSON description."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for path, (a, b) in tables.items():
        arrays[f"{path}.lora_a"], arrays[f"{path}.lora_b"] = mx.array(a), mx.array(b)
    mx.save_safetensors(str(directory / "adapters.safetensors"), arrays)
    (directory / "adapter_info.json").write_text(json.dumps({"modules": sorted(tables), **info}, indent=2) + "\n")
    return directory


def load_lora(directory: str | Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    arrays = mx.load(str(Path(directory) / "adapters.safetensors"))
    paths = sorted({k.rsplit(".", 1)[0] for k in arrays})
    return {p: (np.array(arrays[f"{p}.lora_a"]), np.array(arrays[f"{p}.lora_b"])) for p in paths}
