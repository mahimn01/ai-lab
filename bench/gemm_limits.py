"""How close MLX's matrix multiplies run to this GPU's arithmetic limit, dense and quantized.

Why
  Every large job (training, fine-tuning, reading prompts, big batches) is a pile of
  matrix multiplies. Before trying to speed them up it is worth knowing how much room
  there is. The limit used here comes from public measurements of the M1 Max GPU:
  32 cores x 128 multiply-adds per clock cycle x 1.296 billion cycles per second
  = 5.3 trillion multiply-adds per second.

What it measures
  dense      MLX's dense multiply for the shapes of the lab's small GPT and of LLM layers,
             in 32-bit, 16-bit (float16) and bfloat16 numbers.
  quantized  one real 4-bit layer of each main shape from a model, at 32 to 2,048 rows:
             MLX's quantized kernel against "decode the weights to a dense matrix, then
             dense multiply" (what lab/kernels/qmm.py does from 256 rows up).

Run:     uv run python -m bench.gemm_limits [--model mlx-community/Qwen3.5-9B-4bit]
Changes: writes bench/results/gemm_limits.json. Nothing else.
"""

from __future__ import annotations

import argparse

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

from bench.common import environment, save
from bench.kernel_rows import best_ms
from lab.kernels import qmm

LIMIT = 32 * 128 * 1.296e9  # multiply-adds per second
DENSE_SHAPES = ((16384, 384, 1536, "small GPT, into the wide layer"), (16384, 1536, 384, "small GPT, out of the wide layer"),
                (16384, 384, 8192, "small GPT, output over the vocabulary"), (2048, 2560, 9728, "4B-size layer, 2,048 rows"),
                (512, 4096, 12288, "9B-size layer, 512 rows"))
ROWS = (32, 64, 128, 256, 512, 2048)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="mlx-community/Qwen3.5-9B-4bit")
    args = ap.parse_args()
    results = {"environment": environment(), "limit_multiply_adds_per_s": LIMIT, "dense": {}, "quantized": {}}

    print("dense multiply, share of the arithmetic limit")
    for rows, k, n, what in DENSE_SHAPES:
        entry = {}
        for name, dtype in (("float32", mx.float32), ("float16", mx.float16), ("bfloat16", mx.bfloat16)):
            a, b = mx.random.normal((rows, k)).astype(dtype), mx.random.normal((k, n)).astype(dtype)
            mx.eval(a, b)
            ms = best_ms(lambda: mx.eval([a @ b for _ in range(4)])) / 4  # 4 per timing, so start-up cost does not dominate
            entry[name] = {"ms": ms, "share_of_limit": rows * k * n / (ms / 1000) / LIMIT}
        results["dense"][f"{rows} x {k} x {n} ({what})"] = entry
        print(f"  {rows:6d} x {k:5d} x {n:5d}  {what:38s} " + "  ".join(f"{d} {e['share_of_limit']:4.0%}" for d, e in entry.items()), flush=True)

    model, _ = load(args.model)
    shapes: dict[tuple[int, int], list] = {}
    for _, m in model.named_modules():
        if type(m) is nn.QuantizedLinear and qmm.takes(m) and m.weight.shape[0] <= 20_000:
            shapes.setdefault((m.weight.shape[0], qmm.layer_inputs(m)), []).append(m)
    print("quantized layers: MLX's quantized kernel against decode-then-dense, share of the limit (ms per layer)")
    for (n, k), layers in shapes.items():
        layers = layers[:6]
        for rows in ROWS:
            x = mx.random.normal((rows, k)).astype(layers[0].scales.dtype)
            mx.eval(x)

            def quantized() -> None:
                mx.eval([mx.quantized_matmul(x, m.weight, m.scales, m.biases, transpose=True, group_size=m.group_size,
                                             bits=m.bits) for m in layers])

            def decoded() -> None:
                mx.eval([x @ mx.dequantize(m.weight, m.scales, m.biases, group_size=m.group_size, bits=m.bits).T for m in layers])

            q_ms, d_ms = best_ms(quantized) / len(layers), best_ms(decoded) / len(layers)
            entry = {"quantized_ms": q_ms, "decoded_ms": d_ms, "quantized_share": rows * n * k / (q_ms / 1000) / LIMIT,
                     "decoded_share": rows * n * k / (d_ms / 1000) / LIMIT}
            results["quantized"][f"{n} x {k}, {rows} rows"] = entry
            print(f"  {n:6d} x {k:5d}  {rows:5d} rows: quantized {entry['quantized_share']:4.0%} ({q_ms:6.2f})   "
                  f"decoded {entry['decoded_share']:4.0%} ({d_ms:6.2f})", flush=True)
    save("gemm_limits", args.model, results)


if __name__ == "__main__":
    main()
