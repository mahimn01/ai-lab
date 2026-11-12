"""Speed experiment 10: the lab's multi-row GPU kernel against MLX's own, layer by layer and end to end.

What it measures, for each model
  layers   every quantized weight table of the model is run once on `rows` rows of
           random input: first with MLX's kernel, then with lab/kernels/qmm.py. A
           decoding step also does other work the kernel does not touch, so the gain
           seen here is larger than the gain in a real step.
  decode   a real greedy decoding loop for `batch` prompts at once: ms per step and
           tokens per second, with MLX's kernels and with the lab kernel installed.
           This is the number that counts.
           It also records how many of the written tokens are identical in the two
           runs. The prompts are random tokens, where the model is unsure, and one
           changed token changes everything after it, so this is a harsh measure.
  agree    the fair accuracy check: both kernels read the same real text, 8 passages
           at once, one token per step. At every step they see identical inputs, so
           any difference is the kernels' rounding. Reports how often they pick the
           same next token and the model's loss on the true next token for each.
           For scale it reports the same comparison between two of MLX's own kernels
           (the 8 passages together against one passage at a time).

Run:     uv run python -m bench.kernel_rows --model LABEL=PATH_OR_REPO [--max-batch 32]
Changes: writes bench/results/kernel_rows.json. Nothing else.
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from bench.common import environment, free_gpu_memory, save
from bench.quant_frontier import held_out_text
from lab.infer.batch import CACHE_LIMIT
from lab.kernels import qmm

LAYER_ROWS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 32)
BATCHES = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 18, 24, 32)
PROMPT_TOKENS, STEPS = 64, 48


def best_ms(fn, reps: int = 5) -> float:
    fn()  # warm-up: compiles the kernel and touches the weights
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times) * 1000


def layer_times(model: nn.Module) -> dict:
    """Total ms to run all supported quantized layers once, per row count, for both kernels."""
    layers = [m for _, m in model.named_modules() if type(m) is nn.QuantizedLinear and qmm.takes(m)]
    out = {"layers": len(layers), "weights_billion": sum(m.weight.shape[0] * qmm.layer_inputs(m) for m in layers) / 1e9,
           "rows": {}}
    dtype = layers[0].scales.dtype
    for rows in LAYER_ROWS:
        xs = {k: mx.random.normal((rows, k)).astype(dtype) for k in {qmm.layer_inputs(m) for m in layers}}
        mx.eval(list(xs.values()))

        def mlx_kernel() -> None:
            mx.eval([mx.quantized_matmul(xs[qmm.layer_inputs(m)], m.weight, m.scales, m.biases, transpose=True,
                                         group_size=m.group_size, bits=m.bits) for m in layers])

        def lab_kernel() -> None:
            mx.eval([qmm.matmul_rows(xs[qmm.layer_inputs(m)], m.weight, m.scales, m.biases, bits=m.bits,
                                     group_size=m.group_size) for m in layers])

        row = {"mlx_ms": best_ms(mlx_kernel)}
        if 2 <= rows <= qmm.MAX_ROWS:
            row["lab_ms"] = best_ms(lab_kernel)
        out["rows"][str(rows)] = row
        shown = f"lab {row['lab_ms']:7.2f} ms  ({row['mlx_ms'] / row['lab_ms']:.2f}x)" if "lab_ms" in row else "lab: uses MLX's kernel"
        print(f"  layers, {rows:2d} rows: MLX {row['mlx_ms']:7.2f} ms   {shown}", flush=True)
    return out


def decode(model: nn.Module, prompts: list[list[int]]) -> tuple[float, mx.array]:
    """Greedy-decode STEPS tokens for all prompts at once. Returns (ms per step, the tokens written)."""
    cache = make_prompt_cache(model)
    y = mx.argmax(model(mx.array(prompts), cache=cache)[:, -1, :], axis=-1)
    mx.eval(y)
    for _ in range(4):  # warm-up for the one-token-per-prompt shape
        y = mx.argmax(model(y[:, None], cache=cache)[:, -1, :], axis=-1)
        mx.eval(y)
    tokens = []
    t0 = time.perf_counter()
    for _ in range(STEPS):
        y = mx.argmax(model(y[:, None], cache=cache)[:, -1, :], axis=-1)
        mx.eval(y)
        tokens.append(y)
    return (time.perf_counter() - t0) / STEPS * 1000, mx.stack(tokens)


def decode_times(model: nn.Module, vocab: int, max_batch: int) -> dict:
    mx.random.seed(0)
    prompts = mx.random.randint(0, vocab, (max(BATCHES), PROMPT_TOKENS)).tolist()
    out = {}
    for b in (b for b in BATCHES if b <= max_batch):
        decode(model, prompts[:b])  # first pass warms both the model and the allocator at this batch size
        mlx_ms, mlx_tokens = decode(model, prompts[:b])
        switched = qmm.install(model)
        decode(model, prompts[:b])
        lab_ms, lab_tokens = decode(model, prompts[:b])
        qmm.uninstall(model)
        same = (mlx_tokens == lab_tokens).astype(mx.float32).mean().item()
        out[str(b)] = {"mlx_ms_per_step": mlx_ms, "lab_ms_per_step": lab_ms, "mlx_tokens_per_s": b / mlx_ms * 1000,
                       "lab_tokens_per_s": b / lab_ms * 1000, "identical_tokens": same, "layers_switched": switched,
                       "peak_memory_gb": mx.get_peak_memory() / 2**30}
        print(f"  decode, batch {b:2d}: MLX {mlx_ms:7.2f} ms/step ({b / mlx_ms * 1000:6.1f} tok/s)   "
              f"lab {lab_ms:7.2f} ms/step ({b / lab_ms * 1000:6.1f} tok/s)   {mlx_ms / lab_ms:.2f}x   "
              f"identical tokens {same:.1%}", flush=True)
    return out


def read_text(model: nn.Module, passages: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """Feed the passages one token per step. Returns (chosen token, loss on the true next token, scores at 12 steps)."""
    cache = make_prompt_cache(model)
    picks, losses, kept = [], [], []
    for t in range(passages.shape[1] - 1):
        logits = model(passages[:, t:t + 1], cache=cache)[:, -1, :].astype(mx.float32)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        picks.append(mx.argmax(logits, axis=-1))
        losses.append(-mx.take_along_axis(logprobs, passages[:, t + 1:t + 2], axis=-1)[:, 0])
        if t % 8 == 7:
            kept.append(logits)
        mx.eval(picks[-1], losses[-1], kept)
    return mx.stack(picks), mx.stack(losses), mx.stack(kept)


def agreement(model: nn.Module, tokenizer, batch: int = 8, steps: int = 96) -> dict:
    ids = tokenizer.encode(held_out_text()[:200_000], add_special_tokens=False)
    passages = mx.array(ids[:batch * 2000]).reshape(batch, 2000)[:, :steps + 1]  # 8 passages from different places
    mlx_picks, mlx_loss, mlx_scores = read_text(model, passages)
    qmm.install(model)
    lab_picks, lab_loss, lab_scores = read_text(model, passages)
    qmm.uninstall(model)
    # MLX against itself: one passage at a time goes through its one-row kernel instead of the block kernel
    alone = [read_text(model, passages[i:i + 1]) for i in range(batch)]
    one_picks, one_loss, one_scores = (mx.concatenate([a[j] for a in alone], axis=1) for j in range(3))
    diff, own_diff = mx.abs(mlx_scores - lab_scores), mx.abs(mlx_scores - one_scores)
    out = {"batch": batch, "steps": steps,
           "same_next_token": (mlx_picks == lab_picks).astype(mx.float32).mean().item(),
           "mlx_loss": mlx_loss.mean().item(), "lab_loss": lab_loss.mean().item(),
           "score_difference_mean": diff.mean().item(), "score_difference_max": diff.max().item(),
           "score_spread": mlx_scores.std().item(),
           "mlx_vs_mlx_same_next_token": (mlx_picks == one_picks).astype(mx.float32).mean().item(),
           "mlx_one_at_a_time_loss": one_loss.mean().item(),
           "mlx_vs_mlx_score_difference_mean": own_diff.mean().item(),
           "mlx_vs_mlx_score_difference_max": own_diff.max().item()}
    print(f"  agree, lab vs MLX: same next token {out['same_next_token']:.2%}; loss MLX {out['mlx_loss']:.4f}, "
          f"lab {out['lab_loss']:.4f}; scores differ by {out['score_difference_mean']:.4f} on average "
          f"(max {out['score_difference_max']:.3f}; spread of the scores {out['score_spread']:.2f})", flush=True)
    print(f"  agree, MLX vs MLX (batch {batch} against 1): same next token {out['mlx_vs_mlx_same_next_token']:.2%}; "
          f"loss {out['mlx_one_at_a_time_loss']:.4f}; scores differ by {out['mlx_vs_mlx_score_difference_mean']:.4f} "
          f"on average (max {out['mlx_vs_mlx_score_difference_max']:.3f})", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", action="append", required=True, metavar="LABEL=PATH")
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--skip-layers", action="store_true")
    args = ap.parse_args()
    mx.set_cache_limit(CACHE_LIMIT)
    for spec in args.model:
        label, _, path = spec.partition("=")
        model, tokenizer, config = load(path, return_config=True)
        vocab = config.get("vocab_size") or config["text_config"]["vocab_size"]
        print(label, flush=True)
        entry = {"environment": environment(), "path": path}
        if not args.skip_layers:
            entry["layers"] = layer_times(model)
        entry["decode"] = decode_times(model, vocab, args.max_batch)
        entry["agreement"] = agreement(model, tokenizer, batch=min(8, args.max_batch))
        save("kernel_rows", label, entry)
        del model
        free_gpu_memory()


if __name__ == "__main__":
    main()
