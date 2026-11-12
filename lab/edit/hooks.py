"""Look inside a model while it runs: record, change or steer what a layer outputs.

A model is a stack of layers. Each layer reads a table of numbers [batch, tokens, width]
(the "residual stream": the model's working memory at each token) and writes an updated
one. These tools sit between two layers:

  capture(model, tokens, layers)      record what the chosen layers wrote
  hook(model, layer, fn)              replace a layer's output with fn(output) while the
                                      `with` block runs (ablation, patching)
  steer(model, layer, vector, scale)  add a fixed direction to a layer's output (steering)

They work on any mlx-lm model and on the lab's own GPT, full precision or quantized.
Nothing is written to disk and the model is restored when the `with` block ends.
"""

from __future__ import annotations

from contextlib import contextmanager
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
def hook(model: nn.Module, layer: int, fn: Callable[[mx.array], mx.array]) -> Iterator[None]:
    """While the block runs, layer `layer` outputs fn(its normal output)."""
    layers = layer_list(model)
    original = layers[layer]
    layers[layer] = _Hooked(original, fn)
    try:
        yield
    finally:
        layers[layer] = original


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
    wanted = list(range(len(stack))) if layers is None else layers
    seen: dict[int, mx.array] = {}
    originals = {i: stack[i] for i in wanted}

    def recorder(i: int) -> Callable[[mx.array], mx.array]:
        def record(out: mx.array) -> mx.array:
            seen[i] = out
            return out
        return record

    for i in wanted:
        stack[i] = _Hooked(originals[i], recorder(i))
    try:
        logits = model(tokens)
        mx.eval(logits, list(seen.values()))
    finally:
        for i, layer in originals.items():
            stack[i] = layer
    return logits, seen


def mean_difference(acts_a: mx.array, acts_b: mx.array) -> np.ndarray:
    """The simplest steering direction: average activation on prompts A minus prompts B.

    Each input is [batch, T, width]; the average runs over batch and tokens. Unit length.
    """
    d = np.array((acts_a.astype(mx.float32).mean(axis=(0, 1)) - acts_b.astype(mx.float32).mean(axis=(0, 1))))
    return d / np.linalg.norm(d)
