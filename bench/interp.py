"""Speed experiment 15: reading and changing a model's internals without running layers that are not needed.

Background
  Interpretability work mostly repeats two jobs:
    collect   record what one layer writes for a whole dataset (to train probes or
              sparse autoencoders on it)
    sweep     change something at or above a layer (a steering direction, an ablation)
              and read the model's scores again, for many variants on the same prompts
  A plain forward pass runs every layer both times. lab/edit/hooks.py has run_to (stop
  after the layer) and run_from (start after the layer, from stored results), and the
  lab's dense path (lab/kernels/qmm.py) speeds up the layers that do run.

What it measures
  For one model and a batch of prompts, in tokens per second, each with MLX's kernels and
  with the lab's:
    full pass              model(tokens): every layer, and a score for every position
    full pass, last token  every layer, scores only for what follows the last token
                           (hooks.last_token_scores)
  and at layers a quarter, half and three quarters of the way up:
    collect                run_to: that layer's output for every token
    sweep                  run_from(..., last_only=True): last-token scores from that
                           layer's stored output
  plus the largest difference between the last-token scores of run_from(run_to(...)) and
  of the full pass (it should be zero or rounding).

  MLX only computes what is asked for, so recording a layer inside a full pass and asking
  for nothing else costs the same as run_to. The comparison with the full pass is with a
  workflow that scores the text as well, as hooks.capture does.

Run:     uv run python -m bench.interp --model LABEL=PATH_OR_REPO
Changes: writes bench/results/interp.json. Nothing else.
"""

from __future__ import annotations

import argparse
import time
from typing import Callable

import mlx.core as mx
from mlx_lm import load

from bench.common import environment, free_gpu_memory, save
from lab.edit import hooks
from lab.infer.batch import CACHE_LIMIT
from lab.kernels import qmm

PROMPTS, LENGTH = 8, 256
FRACTIONS = (0.25, 0.5, 0.75)


def rate(fn: Callable[[], mx.array], tokens: int, reps: int = 3) -> float:
    """Best tokens per second over `reps` runs of fn, after one warm-up run."""
    mx.eval(fn())
    best = 0.0
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        best = max(best, tokens / (time.perf_counter() - t0))
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", action="append", required=True, metavar="LABEL=PATH")
    args = ap.parse_args()
    mx.set_cache_limit(CACHE_LIMIT)
    for spec in args.model:
        label, _, path = spec.partition("=")
        model, _, config = load(path, return_config=True)
        vocab = config.get("vocab_size") or config["text_config"]["vocab_size"]
        depth = len(hooks.layer_list(model))
        mx.random.seed(0)
        tokens = mx.random.randint(0, vocab, (PROMPTS, LENGTH))
        n = tokens.size

        entry = {"environment": environment(), "path": path, "prompts": PROMPTS, "tokens_each": LENGTH, "depth": depth,
                 "full_pass": {}, "full_pass_last_token": {}, "layers": {}}
        for kernels in ("mlx", "lab"):
            if kernels == "lab":
                qmm.install(model)
            entry["full_pass"][kernels] = rate(lambda: model(tokens), n)
            entry["full_pass_last_token"][kernels] = rate(lambda: hooks.last_token_scores(model, tokens), n)
        reference = model(tokens)[:, -1, :].astype(mx.float32)
        qmm.uninstall(model)
        full, last = entry["full_pass"], entry["full_pass_last_token"]
        print(f"{label} ({depth} layers, {PROMPTS} x {LENGTH} tokens): full pass {full['mlx']:.0f} -> {full['lab']:.0f} tok/s; "
              f"scoring only the last token {last['mlx']:.0f} -> {last['lab']:.0f} tok/s", flush=True)

        for fraction in FRACTIONS:
            layer = int(depth * fraction) - 1  # the last layer of the bottom quarter, half, three quarters
            states = hooks.run_to(model, tokens, layer)
            mx.eval(states)
            result = {"layers_below_or_at": layer + 1}
            for kernels in ("mlx", "lab"):
                if kernels == "lab":
                    qmm.install(model)
                result[f"collect_{kernels}"] = rate(lambda: hooks.run_to(model, tokens, layer), n)
                result[f"sweep_{kernels}"] = rate(lambda: hooks.run_from(model, states, layer, last_only=True), n)
            resumed = hooks.run_from(model, hooks.run_to(model, tokens, layer), layer, last_only=True).astype(mx.float32)
            result["largest_score_difference"] = mx.abs(resumed - reference).max().item()
            qmm.uninstall(model)
            entry["layers"][str(layer)] = result
            print(f"  layer {layer + 1:2d} of {depth}: collect {result['collect_mlx']:6.0f} -> {result['collect_lab']:6.0f} tok/s "
                  f"({result['collect_lab'] / full['mlx']:.1f}x the full pass with MLX's kernels)   sweep {result['sweep_mlx']:6.0f} -> "
                  f"{result['sweep_lab']:6.0f} tok/s ({result['sweep_lab'] / last['mlx']:.1f}x the last-token full pass with MLX's kernels)   "
                  f"largest score difference {result['largest_score_difference']:.3f}", flush=True)
            save("interp", label, entry)
        del model
        free_gpu_memory()


if __name__ == "__main__":
    main()
