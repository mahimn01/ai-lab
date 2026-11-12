"""LoRA fine-tuning that computes the frozen bottom of the model once per example, not once per step.

The idea
  LoRA normally trains add-on tables in the last few layers only ("--num-layers 8" of
  36, say). The layers below them never change, so for a given training example they
  always produce the same result. This script runs those bottom layers once for every
  batch of examples, keeps the results in memory, and then trains with only the top
  layers in the loop. Each step skips the bottom part of the forward pass.

When it pays
  Whenever the same examples are seen more than once: several passes over a small
  dataset, or a sweep of learning rates / ranks on the same data and the same number of
  trained layers. The fewer layers are trained, the bigger the saving. For a single pass
  over fresh data it only moves the work to the start.

What stays the same
  The adapter it saves has the same format as mlx-lm's and loads with
  `mlx_lm.generate --adapter-path`. Before training, the script checks that its split
  forward pass gives the same loss as the normal one on the first batch and stops if it
  does not (some model families rescale their inputs, which this trick cannot handle).
  mlx-lm's prompt masking is not supported: every token of an example is trained on.

Sweeps
  The cached results are also written to runs/prefix-cache/. The file name carries the
  model's settings, a fingerprint of its frozen weights, the split point and the exact
  tokens. A second run where all of those match (another learning rate, rank or seed)
  loads them in a second instead of computing them again; change the model or the data
  and they are computed afresh.

Run:     uv run python -m lab.finetune.cached --model mlx-community/Qwen3-4B-Instruct-2507-4bit \\
             --num-layers 8 --iters 200
Changes: writes the adapter files under --adapter-path (default runs/lora/cached) and the
         cached layer results under runs/prefix-cache/ (about 0.5 GB for 1,000 short
         examples on a 4B model; safe to delete). Holds the same amount in memory.
         Downloads the model and dataset to the Hugging Face cache if needed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.lora import CONFIG_DEFAULTS
from mlx_lm.tuner.datasets import load_dataset
from mlx_lm.tuner.trainer import default_loss, grad_checkpoint
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.utils import save_config

from lab.edit import hooks
from lab.finetune import gated_delta
from lab.finetune.lora import ByLength, keeps_layer_class, wants_checkpoint
from lab.infer.batch import CACHE_LIMIT
from lab.kernels import qmm

ROOT = Path(__file__).resolve().parent.parent.parent
PAD_TO = 32  # mlx-lm pads every batch to one more than a multiple of this
MAX_CACHE_GB = 6.0
CACHE_DIR = ROOT / "runs" / "prefix-cache"


def make_batches(dataset, batch_size: int, max_seq_length: int, sort: bool = True) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cut a dataset into padded batches the way mlx-lm's trainer does. Returns (tokens [B, L], lengths [B]) pairs.

    sort=True puts examples of similar length in the same batch; False keeps the order of the file,
    which is what mlx-lm itself ends up doing (see lab.finetune.lora.ByLength).
    """
    examples = ByLength(dataset)
    if any(example.offset for example in examples):
        raise ValueError("mlx-lm's prompt masking (mask_prompt) is not supported here: every token is trained on")
    order = sorted(range(len(examples)), key=lambda i: len(examples[i])) if sort else list(range(len(examples)))
    batches = []
    for i in range(0, len(order) - batch_size + 1, batch_size):
        rows = [examples[j] for j in order[i:i + batch_size]]
        lengths = np.array([min(len(r), max_seq_length) for r in rows])
        width = min(1 + PAD_TO * ((lengths.max() + PAD_TO - 1) // PAD_TO), max_seq_length)
        tokens = np.zeros((batch_size, width), np.int32)
        for k, r in enumerate(rows):
            tokens[k, :lengths[k]] = r[:lengths[k]]
        batches.append((tokens, lengths))
    return batches


def frozen_fingerprint(model: nn.Module, config: dict, split: int) -> str:
    """A short code that changes whenever the model's settings or any weight below layer `split` change.

    It is part of the cache file's name, so results cached from one set of weights are never reused
    after those weights were edited, even when the model is loaded from the same path.
    """
    digest = hashlib.sha1(json.dumps(config, sort_keys=True, default=str).encode())
    for name, value in tree_flatten(model.parameters()):
        index = re.search(r"\blayers\.(\d+)\.", name)
        if index is None or int(index.group(1)) < split:
            as_numbers = value if value.dtype in (mx.uint32, mx.float32) else value.astype(mx.float32)
            digest.update(f"{name}{value.shape}{mx.sum(as_numbers).item()!r}{mx.sum(as_numbers * as_numbers).item()!r}".encode())
    return digest.hexdigest()[:10]


def top_loss(model: nn.Module, states: mx.array, targets: mx.array, lengths: mx.array) -> tuple[mx.array, mx.array]:
    """mlx-lm's training loss, starting from the cached layer results instead of the token ids."""
    logits = model(targets, input_embeddings=states)  # the token argument is ignored when states are given
    mask = mx.arange(1, targets.shape[1] + 1) <= lengths[:, None]
    ce = nn.losses.cross_entropy(logits, targets) * mask
    ntoks = mask.sum()
    return ce.astype(mx.float32).sum() / ntoks, ntoks


def finetune_cached(model_path: str, *, data: str = CONFIG_DEFAULTS["data"], adapter_path: str | Path = ROOT / "runs/lora/cached",
                    iters: int = 200, batch_size: int = 4, num_layers: int = 8, max_seq_length: int = 512,
                    learning_rate: float = 1e-5, rank: int = 8, val_batches: int = 10, seed: int = 0,
                    reuse_cache: bool = True, sort_by_length: bool = True, cache_dir: str | Path = CACHE_DIR) -> dict:
    """Train LoRA tables in the last `num_layers` layers with the layers below them computed once. Returns a summary.

    reuse_cache=False computes the cached layer results again even if an earlier run saved them in cache_dir.
    Validation uses the first `val_batches` batches (-1 = all of them); mlx-lm picks its
    validation batches at random, so the two only score the same examples with val_batches=-1.
    sort_by_length is as in lab.finetune.lora.finetune.
    """
    config = copy.deepcopy(CONFIG_DEFAULTS) | {
        "model": model_path, "train": True, "data": data, "adapter_path": str(adapter_path), "iters": iters,
        "batch_size": batch_size, "num_layers": num_layers, "max_seq_length": max_seq_length,
        "learning_rate": learning_rate, "fine_tune_type": "lora", "val_batches": val_batches, "seed": seed,
    }
    config["lora_parameters"]["rank"] = rank
    args = argparse.Namespace(**config)
    np.random.seed(seed)
    mx.random.seed(seed)
    mx.set_cache_limit(CACHE_LIMIT)

    model, tokenizer, model_config = load(model_path, return_config=True)
    qmm.install(model)  # the frozen layers read many rows at once: the lab's dense path is faster there
    gated_delta.install()  # Qwen3.5 family: the 64-tokens-at-a-time update instead of mlx-lm's training loop
    train_set, valid_set, _ = load_dataset(args, tokenizer)
    train_batches = make_batches(train_set, batch_size, max_seq_length, sort_by_length)
    valid_batches = make_batches(valid_set, batch_size, max_seq_length, sort_by_length)
    if val_batches >= 0:  # -1 = the whole validation set, as in mlx-lm
        valid_batches = valid_batches[:val_batches]

    model.freeze()
    linear_to_lora_layers(model, num_layers, config["lora_parameters"])
    full = list(hooks.layer_list(model))
    split = len(full) - num_layers
    if split < 1:
        raise ValueError(f"the model has {len(full)} layers; train fewer than that for there to be a frozen part")
    checkpoint = wants_checkpoint(model, True, num_layers, max(t[:, :-1].size for t, _ in train_batches))

    # 1. The frozen bottom, once per batch: results entering layer `split`, kept as raw 16-bit values.
    probe = hooks.run_to(model, mx.array(train_batches[0][0][:, :-1]), split - 1)  # lazy: only its shape and type are used
    cache_tokens = sum(t[:, :-1].size for t, _ in train_batches + valid_batches)
    need_gb = cache_tokens * probe.shape[-1] * probe.dtype.size / 2**30
    if need_gb > MAX_CACHE_GB:
        raise MemoryError(f"the cached layer results would take {need_gb:.1f} GB; use fewer examples or a shorter --max-seq-length")
    model.eval()
    t0 = time.perf_counter()
    loaded: list[str] = []  # which splits came from disk
    weights_code = frozen_fingerprint(model, model_config, split)

    def cache(batches: list[tuple[np.ndarray, np.ndarray]], split_name: str) -> list[np.ndarray]:
        # The file name carries everything the results depend on: the frozen weights and the exact tokens.
        tokens_hash = hashlib.sha1(b"".join(str(t.shape).encode() + t.tobytes() for t, _ in batches)).hexdigest()[:12]
        name = f"{model_path.replace('/', '--')}_{weights_code}_layer{split}_{split_name}_{tokens_hash}.npz"
        file = Path(cache_dir) / name
        if reuse_cache and file.exists():
            loaded.append(split_name)
            with np.load(file) as saved:
                return [saved[f"arr_{i}"] for i in range(len(batches))]
        out = []
        for tokens, _ in batches:
            states = hooks.run_to(model, mx.array(tokens[:, :-1]), split - 1)
            out.append(np.array(states.view(mx.uint16)) if states.dtype == mx.bfloat16 else np.array(states))
        file.parent.mkdir(parents=True, exist_ok=True)
        unfinished = file.with_name("unfinished-" + file.name)  # so a stopped run never leaves half a file under the real name
        np.savez(unfinished, *out)
        os.replace(unfinished, file)
        return out

    train_states, valid_states = cache(train_batches, "train"), cache(valid_batches, "valid")
    cache_s = time.perf_counter() - t0
    as_states = (lambda a: mx.array(a).view(mx.bfloat16)) if probe.dtype == mx.bfloat16 else mx.array

    def batch_args(i: int, batches, states) -> tuple[mx.array, mx.array, mx.array]:
        tokens, lengths = batches[i]
        return as_states(states[i]), mx.array(tokens[:, 1:]), mx.array(lengths)

    # 2. Check the split against the normal forward pass on one batch, then drop the bottom layers from the loop.
    tokens0, lengths0 = train_batches[0]
    normal = default_loss(model, mx.array(tokens0), mx.array(np.stack([np.zeros_like(lengths0), lengths0], axis=1)))[0].item()
    # The whole model is back after this block, so the adapter gets the right layer numbers.
    with keeps_layer_class(full[-1]), hooks.top_only(model, split):
        if checkpoint:  # big Qwen3.5-family models or long batches: recompute instead of storing, to save memory
            grad_checkpoint(full[-1])
        split_loss = top_loss(model, *batch_args(0, train_batches, train_states))[0].item()
        if abs(split_loss - normal) > 0.005 * max(1.0, abs(normal)):
            raise RuntimeError(f"the split forward pass gives loss {split_loss:.4f} but the normal one {normal:.4f}: "
                               "this model family cannot be trained from cached layer results")

        def mean_loss(batches, states) -> float:
            parts = [top_loss(model, *batch_args(i, batches, states)) for i in range(len(batches))]
            return (sum(l.item() * n.item() for l, n in parts) / sum(n.item() for _, n in parts)) if parts else float("nan")

        val_before = mean_loss(valid_batches, valid_states)

        # 3. Train: only the top layers run, forwards and backwards.
        model.train()
        optimizer = optim.Adam(learning_rate=learning_rate)
        loss_and_grad = nn.value_and_grad(model, top_loss)
        mx.reset_peak_memory()
        order: list[int] = []
        step_tokens, step_s, losses = [], [], []
        for step in range(iters):
            if not order:
                order = list(np.random.permutation(len(train_batches)))
            i = order.pop()
            t1 = time.perf_counter()
            (loss, ntoks), grads = loss_and_grad(model, *batch_args(i, train_batches, train_states))
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss, ntoks)
            step_s.append(time.perf_counter() - t1)
            step_tokens.append(ntoks.item())
            losses.append(loss.item())
            if (step + 1) % 10 == 0:
                print(f"  step {step + 1:5d}  loss {np.mean(losses[-10:]):.3f}  "
                      f"{sum(step_tokens[-10:]) / sum(step_s[-10:]):,.0f} tok/s", flush=True)
        model.eval()
        val_after = mean_loss(valid_batches, valid_states)

    adapter_path = Path(adapter_path)
    adapter_path.mkdir(parents=True, exist_ok=True)
    save_config(config, adapter_path / "adapter_config.json")
    mx.save_safetensors(str(adapter_path / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    skip = 1 if iters > 1 else 0  # the first step includes one-off GPU compilation
    return {
        "model": model_path, "data": data, "iters": iters, "batch_size": batch_size, "num_layers": num_layers,
        "layers_total": len(full), "max_seq_length": max_seq_length, "rank": rank,
        "tokens_per_second": sum(step_tokens[skip:]) / sum(step_s[skip:]),
        "tokens_per_step": float(np.mean(step_tokens)),
        "cache_seconds": cache_s, "cache_tokens_per_second": cache_tokens / cache_s, "cache_gb": need_gb,
        "cache_from_disk": loaded == ["train", "valid"], "grad_checkpoint": checkpoint, "sort_by_length": sort_by_length,
        "peak_memory_gb": mx.get_peak_memory() / 2**30,
        "loss_check_normal": normal, "loss_check_split": split_loss,
        "train_loss_first": float(np.mean(losses[:10])), "train_loss_last": float(np.mean(losses[-10:])),
        "val_loss_before": val_before, "val_loss_after": val_after, "adapter_path": str(adapter_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=CONFIG_DEFAULTS["data"])
    ap.add_argument("--adapter-path", default=str(ROOT / "runs/lora/cached"))
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-layers", type=int, default=8, help="how many of the last layers get LoRA tables")
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--rank", type=int, default=8)
    args = ap.parse_args()
    summary = finetune_cached(args.model, data=args.data, adapter_path=args.adapter_path, iters=args.iters,
                              batch_size=args.batch_size, num_layers=args.num_layers, max_seq_length=args.max_seq_length,
                              learning_rate=args.learning_rate, rank=args.rank)
    print("RESULT " + json.dumps(summary))


if __name__ == "__main__":
    main()
