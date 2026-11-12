"""Accuracy of the quantized matmul kernels against the exact answer, on a real model's layers.

What it measures
  For the model's main layer shapes and for 4, 8 and 16 rows of random input, the error
  of each kernel against the exact result (weights fully decoded to 32-bit floats,
  multiplied in 32-bit). The error is given as a share of the typical output size (root
  mean square). Four columns:
    MLX, one row at a time   MLX's one-row kernel
    MLX, rows together       what MLX runs for that row count (its 32-row block kernel
                             once the row count passes its threshold)
    lab kernel               lab/kernels/qmm.py
    ideal                    the exact result rounded once to the 16-bit output type:
                             no kernel can do better than this

Run:     uv run python -m bench.kernel_exact [--model mlx-community/Qwen3.5-9B-4bit]
Changes: writes bench/results/kernel_exact.json. Nothing else.
"""

from __future__ import annotations

import argparse

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

from bench.common import environment, save
from lab.kernels import qmm

ROWS = (4, 8, 16)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="mlx-community/Qwen3.5-9B-4bit")
    args = ap.parse_args()
    model, _ = load(args.model)
    shapes: dict[tuple[int, int], nn.Module] = {}  # the first layer of every shape the lab kernel would take
    for _, m in model.named_modules():
        if type(m) is nn.QuantizedLinear and qmm.takes(m) and m.weight.shape[0] <= 20_000:
            shapes.setdefault((m.weight.shape[0], qmm.layer_inputs(m)), m)
    mx.random.seed(1)
    results = {"environment": environment(), "layers": {}}
    print("error against the exact result, as a share of the typical output size")
    print("outputs x inputs    rows   MLX one row   MLX together   lab kernel   ideal")
    for (n, k), m in shapes.items():
        exact_w = mx.dequantize(m.weight, m.scales.astype(mx.float32), m.biases.astype(mx.float32),
                                group_size=m.group_size, bits=m.bits)
        for rows in ROWS:
            x = mx.random.normal((rows, k)).astype(m.scales.dtype)
            exact = x.astype(mx.float32) @ exact_w.T
            size = mx.sqrt((exact * exact).mean())

            def error(y: mx.array) -> float:
                d = y.astype(mx.float32) - exact
                return (mx.sqrt((d * d).mean()) / size).item()

            def mlx_kernel(inp: mx.array) -> mx.array:
                return mx.quantized_matmul(inp, m.weight, m.scales, m.biases, transpose=True, group_size=m.group_size,
                                           bits=m.bits)

            row = {"mlx_one_row": error(mx.concatenate([mlx_kernel(x[i:i + 1]) for i in range(rows)])),
                   "mlx_together": error(mlx_kernel(x)),
                   "lab": error(qmm.matmul_rows(x, m.weight, m.scales, m.biases, bits=m.bits, group_size=m.group_size)),
                   "ideal": error(exact.astype(m.scales.dtype))}
            results["layers"][f"{n}x{k}, {rows} rows"] = row
            print(f"{n:7d} x {k:5d}    {rows:3d}    {row['mlx_one_row']:.2e}      {row['mlx_together']:.2e}     "
                  f"{row['lab']:.2e}   {row['ideal']:.2e}", flush=True)
    save("kernel_exact", args.model, results)


if __name__ == "__main__":
    main()
