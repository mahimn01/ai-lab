"""Speed experiment 13 (part): three ways to run the Qwen3.5 recurrent layer in training, one layer at a time.

What it measures
  For one layer with the 4B model's shape (16 key heads, 32 value heads, width 128) and a
  batch of 4, at 100, 256, 512 and 1,024 tokens: the forward pass alone and forward plus
  backward, and the peak memory of the latter, for
    loop      mlx-lm's own training path: a Python loop over the tokens
    built-in  MLX's mx.fast.gated_delta_update (fast forward; on chips before the M5 its
              backward pass goes token by token and stores every state)
    chunked   lab/finetune/gated_delta.py: the same recurrence, 64 tokens at a time
  It also records how far the built-in and the chunked outputs are from the loop's.

Run:     uv run python -m bench.gated_delta
Changes: writes bench/results/gated_delta.json. Nothing else.
"""

from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.gated_delta import gated_delta_ops

from bench.common import environment, save
from bench.kernel_rows import best_ms
from lab.finetune.gated_delta import chunked_update

B, HK, HV, D = 4, 16, 32, 128
LENGTHS = (100, 256, 512, 1024)
# Lightest first: a variant measured straight after one that used 8-10 GB came out several times too slow.
WAYS = {"chunked": chunked_update, "built-in": lambda *a: tuple(mx.fast.gated_delta_update(*a)), "loop": gated_delta_ops}


def main() -> None:
    results = {"environment": environment(), "shape": {"batch": B, "key_heads": HK, "value_heads": HV, "width": D}, "tokens": {}}
    mx.random.seed(0)
    for T in LENGTHS:
        q = (mx.random.normal((B, T, HK, D)) * 0.09).astype(mx.bfloat16)
        k = (mx.random.normal((B, T, HK, D)) * 0.09).astype(mx.bfloat16)
        v = mx.random.normal((B, T, HV, D)).astype(mx.bfloat16)
        g, beta = mx.sigmoid(mx.random.normal((B, T, HV)) + 2), mx.sigmoid(mx.random.normal((B, T, HV)))
        state = mx.zeros((B, HV, D, D))
        mx.eval(q, k, v, g, beta)
        reference = gated_delta_ops(q, k, v, g, beta, state)[0].astype(mx.float32)
        row = {}
        for name, fn in WAYS.items():
            if name == "loop" and T > 512:
                continue  # the loop would need about 20 GB here
            reps = 1 if name == "loop" else 4
            mx.clear_cache()

            def forward() -> None:
                mx.eval(fn(q, k, v, g, beta, state))

            grad = mx.grad(lambda q, k, v: fn(q, k, v, g, beta, state)[0].astype(mx.float32).sum(), argnums=[0, 1, 2])

            def both() -> None:
                mx.eval(grad(q, k, v))

            forward_ms, both_ms = best_ms(forward, reps), best_ms(both, reps)
            mx.reset_peak_memory()
            both()
            y = fn(q, k, v, g, beta, state)[0].astype(mx.float32)
            row[name] = {"forward_ms": forward_ms, "forward_backward_ms": both_ms, "peak_gb": mx.get_peak_memory() / 2**30,
                         "difference_from_loop": (mx.abs(y - reference).max() / mx.abs(reference).max()).item()}
            print(f"{T:4d} tokens  {name:9s} forward {forward_ms:7.1f} ms   forward+backward {both_ms:7.1f} ms   "
                  f"peak {row[name]['peak_gb']:4.1f} GB   differs from the loop by {row[name]['difference_from_loop']:.0e}", flush=True)
        results["tokens"][str(T)] = row
    save("gated_delta", "one layer, 4B shape", results)


if __name__ == "__main__":
    main()
