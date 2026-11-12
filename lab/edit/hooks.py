"""Look inside a model while it runs: record, change or steer what a layer outputs.

A model is a stack of layers. Each layer reads a table of numbers [batch, tokens, width]
(the "residual stream": the model's working memory at each token) and writes an updated
one. These tools sit between two layers:

  capture(model, tokens, layers)      record what the chosen layers wrote
  hook(model, layer, fn)              replace a layer's output with fn(output) while the
                                      `with` block runs (ablation, patching)
  steer(model, layer, vector, scale)  add a fixed direction to a layer's output (steering)

and three more avoid running layers whose result is not needed or already known:

  run_to(model, tokens, layer)        a layer's output, without running the layers after it
  run_from(model, states, layer)      the model's scores from a layer's output, running only
                                      the layers after it (for trying many changes high up)
  last_token_scores(model, tokens)    the scores after the last token only, skipping the
                                      output layer for every other position
  save_activations(...)               a layer's output for a whole dataset, to one file

They work on any mlx-lm model and on the lab's own GPT, full precision or quantized.
Only save_activations writes to disk, and the model is restored when each call or `with`
block ends.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Callable, Iterator

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def layer_list(model: nn.Module) -> list:
    """The model's real list of layers (the list the forward pass walks), so entries can be swapped.

    mlx-lm models keep it in an attribute called `layers` somewhere inside the model
    (their `model.layers` property returns a copy, which is no use for swapping); the
    lab's GPT calls it `blocks`. The longest such list is the layer stack.
    """
    best: list = []
    for _, module in [("", model), *model.named_modules()]:
        for key in ("layers", "blocks"):
            found = module.get(key)
            if isinstance(found, list) and len(found) > len(best):
                best = found
    if not best:
        raise ValueError("could not find the model's layer list (no `layers` or `blocks` attribute)")
    return best


class _Hooked(nn.Module):
    """Stands in for one layer: runs it, then passes its output through `fn`."""

    def __init__(self, inner: nn.Module, fn: Callable[[mx.array], mx.array]):
        super().__init__()
        self.inner = inner
        self._fn = fn

    def __call__(self, *args, **kwargs):
        out = self.inner(*args, **kwargs)
        if isinstance(out, tuple):
            return (self._fn(out[0]), *out[1:])
        return self._fn(out)

    def __getattr__(self, name: str):
        # Models read flags off their layers (Qwen3.5 checks `layer.is_linear`): pass those through.
        try:
            return super().__getattr__(name)
        except AttributeError:
            if "inner" not in self:  # still being built
                raise
            return getattr(self["inner"], name)


@contextmanager
def _swapped(layers: list, index: int, fn: Callable[[mx.array], mx.array]) -> Iterator[None]:
    original = layers[index]
    layers[index] = _Hooked(original, fn)
    try:
        yield
    finally:
        layers[index] = original


@contextmanager
def hook(model: nn.Module, layer: int, fn: Callable[[mx.array], mx.array]) -> Iterator[None]:
    """While the block runs, layer `layer` outputs fn(its normal output)."""
    with _swapped(layer_list(model), layer, fn):
        yield


@contextmanager
def steer(model: nn.Module, layer: int, vector: np.ndarray | mx.array, scale: float = 1.0) -> Iterator[None]:
    """While the block runs, add scale * vector to every token's output at `layer`."""
    v = mx.array(vector)

    def add(out: mx.array) -> mx.array:
        return out + (scale * v).astype(out.dtype)

    with hook(model, layer, add):
        yield


def capture(model: nn.Module, tokens: mx.array, layers: list[int] | None = None) -> tuple[mx.array, dict[int, mx.array]]:
    """Run the model once on tokens [batch, T]; return (logits, {layer index: output [batch, T, width]}).

    With layers=None every layer is recorded. Outputs keep the model's number format.
    """
    stack = layer_list(model)
    seen: dict[int, mx.array] = {}

    def recorder(i: int) -> Callable[[mx.array], mx.array]:
        def record(out: mx.array) -> mx.array:
            seen[i] = out
            return out
        return record

    with ExitStack() as recording:
        for i in range(len(stack)) if layers is None else layers:
            recording.enter_context(_swapped(stack, i, recorder(i)))
        logits = model(tokens)
        mx.eval(logits, list(seen.values()))
    return logits, seen


class _Stop(Exception):
    """Raised inside a hook to end the model's forward pass early."""


def run_to(model: nn.Module, tokens: mx.array, layer: int) -> mx.array:
    """Output of layer `layer` for tokens [batch, T], without computing anything after that layer.

    Reading layer 12 of a 36-layer model this way costs a third of a full pass. The result is
    lazy like any MLX array; call mx.eval on it (or use it) to compute it.
    """
    seen: list[mx.array] = []

    def grab(out: mx.array) -> mx.array:
        seen.append(out)
        raise _Stop

    with hook(model, layer, grab):
        try:
            model(tokens)
        except _Stop:
            pass
    return seen[0]


@contextmanager
def top_only(model: nn.Module, first: int) -> Iterator[list]:
    """While the block runs, the model is only its layers from index `first` on (the block is handed that list).

    Call it as model(tokens, input_embeddings=states) with states = what layer `first - 1` wrote.
    Layer numbers shift inside the block, so set any hook() or steer() before entering it.
    """
    layers = layer_list(model)
    full = list(layers)
    layers[:] = full[first:]
    try:
        yield layers
    finally:
        layers[:] = full


def _last_position(out: mx.array) -> mx.array:
    return out[:, -1:, :]


def last_token_scores(model: nn.Module, tokens: mx.array) -> mx.array:
    """The model's scores [batch, vocab] for what follows the last token of tokens [batch, T].

    Same numbers as model(tokens)[:, -1, :], but the output layer runs on one position per prompt.
    That layer is a tenth of all the weights of a 4B Qwen model (151,936 words x 2,560 wide).
    """
    with hook(model, -1, _last_position):
        return model(tokens)[:, 0, :]


def run_from(model: nn.Module, states: mx.array, layer: int, last_only: bool = False) -> mx.array:
    """The model's scores [batch, T, vocab] from `states`, the output of layer `layer`, running only the later layers.

    With states = run_to(model, tokens, layer) this equals model(tokens). Its use: compute the
    states once, then try many changes to them or to later layers (wrap the call in hook() or
    steer()) without paying for the layers below again. It relies on the model starting from
    `input_embeddings` unchanged, which holds for the lab's GPT and the Qwen families; check
    run_from(run_to(...)) against model(tokens) once on any other family.

    last_only=True returns only the scores after the last token, [batch, vocab], as
    last_token_scores does.
    """
    tokens = mx.zeros(states.shape[:2], dtype=mx.int32)  # ignored by the model when states are given
    with top_only(model, layer + 1) as remaining:
        if not last_only:
            return model(tokens, input_embeddings=states)
        if not remaining:  # `layer` was the last one: only the output layer is left to run
            return model(tokens[:, -1:], input_embeddings=_last_position(states))[:, 0, :]
        with _swapped(remaining, -1, _last_position):
            return model(tokens, input_embeddings=states)[:, 0, :]


def save_activations(model: nn.Module, batches: list[mx.array], layer: int, path: str | Path) -> dict:
    """Write layer `layer`'s output for every token of every batch to one .npy file: [tokens in total, width], float16.

    Each batch is tokens [batch, T]. Only the layers up to `layer` are run (see run_to), so collecting
    an early layer is much cheaper than a full pass. The file can be opened without loading it with
    np.load(path, mmap_mode="r"). float16 keeps 3 significant digits, enough for probes and
    sparse autoencoders; values beyond +-65504 would overflow. The summary says whether that
    happened ("overflowed") and whether the model itself produced any NaN or infinity ("all_finite").
    """
    width = run_to(model, batches[0][:1, :2], layer).shape[-1]
    total = sum(b.size for b in batches)
    out = np.lib.format.open_memmap(str(path), mode="w+", dtype=np.float16, shape=(total, width))
    at, largest, finite = 0, 0.0, True
    for tokens in batches:
        acts = run_to(model, tokens, layer).astype(mx.float32).reshape(-1, width)
        finite = finite and mx.isfinite(acts).all().item()
        largest = max(largest, mx.abs(mx.nan_to_num(acts)).max().item())
        out[at:at + acts.shape[0]] = np.array(acts.astype(mx.float16))
        at += acts.shape[0]
    out.flush()
    return {"tokens": total, "width": width, "largest_value": largest, "overflowed": largest > 65504,
            "all_finite": finite, "path": str(path)}


def mean_difference(acts_a: mx.array, acts_b: mx.array) -> np.ndarray:
    """The simplest steering direction: average activation on prompts A minus prompts B.

    Each input is [batch, T, width]; the average runs over batch and tokens. Unit length.
    """
    d = np.array((acts_a.astype(mx.float32).mean(axis=(0, 1)) - acts_b.astype(mx.float32).mean(axis=(0, 1))))
    return d / np.linalg.norm(d)
