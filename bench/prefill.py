"""Speed experiment 11: reading prompts with the lab's dense path for many rows against MLX's quantized kernel.

Background
  Reading a prompt runs every weight table on one row per prompt token, hundreds or
  thousands of rows at once. MLX's quantized kernel reaches about half of this GPU's
  arithmetic limit there, its dense kernel about two thirds. lab/kernels/qmm.py therefore
  decodes each layer's weights to a temporary dense matrix when 256 or more rows arrive
  and uses the dense kernel.

What it measures
  Tokens per second for reading a few prompt shapes and scoring every position (what
  scoring a text needs; reading a prompt before generating skips the output layer, and
  bench/float16.py times that case), with MLX's kernels and with the lab path installed,
  the peak memory of each, and how far apart the two sets of last-token scores are.
  MLX's own block kernel also rounds the decoded weights to 16 bits, so the two usually
  agree exactly; when the sums happen to run in a different order they differ by
  rounding, which the largest difference shows.

Run:     uv run python -m bench.prefill --model LABEL=PATH_OR_REPO
         (one model per run: with several in one process, the first memory peak of each
         later model still counted the model before it)
Changes: writes bench/results/prefill.json. Nothing else.
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
from mlx_lm import load

from bench.common import environment, free_gpu_memory, save
from lab.infer.batch import CACHE_LIMIT
from lab.kernels import qmm

SHAPES = ((1, 512), (1, 2048), (8, 512), (32, 256))  # (prompts, tokens each)


def read(model, tokens: mx.array, reps: int = 3) -> tuple[float, mx.array, float]:
    """Read the prompts `reps` times. Returns (best tokens per second, last-token scores, peak memory in GB)."""
    mx.eval(model(tokens)[:, -1, :])  # warm-up
    mx.reset_peak_memory()
    best, scores = 0.0, None
    for _ in range(reps):
        t0 = time.perf_counter()
        scores = model(tokens)[:, -1, :]  # after reading a prompt only the last position's scores are needed
        mx.eval(scores)
        best = max(best, tokens.size / (time.perf_counter() - t0))
    return best, scores, mx.get_peak_memory() / 2**30


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", action="append", required=True, metavar="LABEL=PATH")
    args = ap.parse_args()
    mx.set_cache_limit(CACHE_LIMIT)
    for spec in args.model:
        label, _, path = spec.partition("=")
        model, _, config = load(path, return_config=True)
        vocab = config.get("vocab_size") or config["text_config"]["vocab_size"]
        entry = {"environment": environment(), "path": path, "shapes": {}}
        print(label, flush=True)
        mx.random.seed(0)
        for prompts, length in SHAPES:
            tokens = mx.random.randint(0, vocab, (prompts, length))
            mlx_tps, mlx_scores, mlx_gb = read(model, tokens)
            qmm.install(model)
            lab_tps, lab_scores, lab_gb = read(model, tokens)
            qmm.uninstall(model)
            a, b = mlx_scores.astype(mx.float32), lab_scores.astype(mx.float32)
            gap = mx.abs(a - b).max().item()
            same_pick = (mx.argmax(a, axis=-1) == mx.argmax(b, axis=-1)).all().item()
            entry["shapes"][f"{prompts} x {length}"] = {
                "mlx_tokens_per_s": mlx_tps, "lab_tokens_per_s": lab_tps, "mlx_peak_gb": mlx_gb, "lab_peak_gb": lab_gb,
                "largest_score_difference": gap, "score_spread": a.std().item(), "same_top_token": same_pick}
            print(f"  {prompts:2d} prompt(s) x {length:4d} tokens: MLX {mlx_tps:6.0f} tok/s ({mlx_gb:4.1f} GB)   "
                  f"lab {lab_tps:6.0f} tok/s ({lab_gb:4.1f} GB)   {lab_tps / mlx_tps:.2f}x   largest score difference "
                  f"{gap:.3f} (spread {a.std().item():.2f}), same top token: {same_pick}", flush=True)
            save("prefill", label, entry)
        del model
        free_gpu_memory()


if __name__ == "__main__":
    main()
