"""What limits the quantized matmul on this GPU? The lab kernel built up one step at a time.

Why
  To know whether a faster kernel is possible, and for which row counts, the cost has to
  be split into its parts. This script runs cut-down versions of lab/kernels/qmm.py on a
  real model's weight tables (4-bit, group 64, bfloat16) and times each:
    0  start the threads, loop over the inputs, sum across threads, store. No reads.
    1  + read the weight bytes
    2  + turn every 4-bit value into a number
    3  + read scale and bias and decode the weight (w = scale * q + bias)
    4  + read the rows of x into registers
    5  + the multiply-adds: the full kernel
  Each is run twice: on the real weights, and "cached", where every output reads the
  same row of weights, so nothing has to come from main memory. Step 1 minus step 0 on
  the real weights is the time to stream the weight bytes from memory.

How to read the result
  If step 1 on real weights is already close to the full kernel, the kernel is limited by
  memory speed and no cleverer arithmetic can help. That is the case for one row.

Run:     uv run python -m bench.kernel_limits [--model mlx-community/Qwen3.5-9B-4bit]
Changes: writes bench/results/kernel_limits.json. Nothing else.
"""

from __future__ import annotations

import argparse

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

from bench.common import environment, save
from bench.kernel_rows import best_ms
from lab.kernels.qmm import LANES, RPS, VPT

ROWS = (1, 4, 8)
STEPS = ("threads and loop only", "+ read weights", "+ unpack 4-bit values", "+ decode with scale and bias",
         "+ read the rows of x", "+ multiply-adds (full kernel)")


def source(rows: int, level: int, cached: bool) -> str:
    """The kernel of lab/kernels/qmm.py for 4-bit weights, cut off after `level` steps."""
    a = [
        "const uint3 tid = threadgroup_position_in_grid;",
        "const uint simd_gid = simdgroup_index_in_threadgroup;",
        "const uint simd_lid = thread_index_in_simdgroup;",
        "const int K_w = K / 2; const int K_g = K / 64;",
        f"const int out_row = tid.y * {2 * RPS} + simd_gid * {RPS};",
        f"const int w_row = {'0' if cached else 'out_row'};",
        f"const device uint8_t* ws = (const device uint8_t*)w + w_row * K_w + simd_lid * {VPT // 2};",
        f"const device bfloat16_t* sl = scales + w_row * K_g + simd_lid / {64 // VPT};",
        f"const device bfloat16_t* bl = biases + w_row * K_g + simd_lid / {64 // VPT};",
        f"const device bfloat16_t* xp = x + simd_lid * {VPT};",
        "uint isum = simd_lid;",
    ]
    a += [f"float r_{m}_{r} = 0;" for m in range(rows) for r in range(RPS)]
    a.append(f"for (int k = 0; k < K; k += {LANES * VPT}) {{")
    if level >= 4:
        for m in range(rows):  # two 16-bit inputs per 32-bit read, as in the real kernel
            for j in range(VPT // 2):
                a.append(f"uint xw_{m}_{j} = ((const device uint32_t*)(xp + {m} * K))[{j}];")
                a.append(f"float x_{m}_{2 * j} = as_type<float>(xw_{m}_{j} << 16); "
                         f"float x_{m}_{2 * j + 1} = as_type<float>(xw_{m}_{j} & 0xffff0000u);")
        if level == 4:  # use the values, so the compiler keeps the reads
            a += [f"r_{m}_0 += " + " + ".join(f"x_{m}_{i}" for i in range(VPT)) + ";" for m in range(rows)]
    for r in range(RPS):
        a.append(f"{{ const device uint8_t* wl = ws + {r} * K_w;")
        if level >= 1:
            a += [f"uint8_t wb{i} = wl[{i}];" for i in range(VPT // 2)]
        if level == 1:
            a.append("isum += " + " + ".join(f"uint(wb{i})" for i in range(VPT // 2)) + ";")
        if level == 2:
            a.append(f"r_0_{r} += " + " + ".join(f"float(wb{i} & 0x0f) + float(wb{i} & 0xf0)" for i in range(VPT // 2)) + ";")
        if level >= 3:
            a.append(f"float s = float(sl[{r} * K_g]); float b = float(bl[{r} * K_g]); float s16 = s * 0.0625f;")
            for i in range(VPT // 2):
                a.append(f"float q_{2 * i} = s * float(wb{i} & 0x0f) + b; float q_{2 * i + 1} = s16 * float(wb{i} & 0xf0) + b;")
            if level < 5:
                a.append(f"r_0_{r} += " + " + ".join(f"q_{i}" for i in range(VPT)) + ";")
            else:
                a += [f"r_{m}_{r} = fma(x_{m}_{i}, q_{i}, r_{m}_{r});" for m in range(rows) for i in range(VPT)]
        a.append("}")
    a.append(f"ws += {LANES * VPT // 2}; sl += {LANES * VPT // 64}; bl += {LANES * VPT // 64}; xp += {LANES * VPT}; }}")
    a.append("r_0_0 += float(isum & 1u);")
    a += [f"r_{m}_{r} = simd_sum(r_{m}_{r});" for m in range(rows) for r in range(RPS)]
    a.append("if (simd_lid == 0) {")
    a += [f"y[{m} * N + out_row + {r}] = static_cast<bfloat16_t>(r_{m}_{r});" for m in range(rows) for r in range(RPS)]
    a.append("}")
    return "\n".join(a)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="mlx-community/Qwen3.5-9B-4bit")
    args = ap.parse_args()
    model, _ = load(args.model)
    layers = [m for _, m in model.named_modules() if type(m) is nn.QuantizedLinear and m.bits == 4 and m.group_size == 64
              and m.scales.dtype == mx.bfloat16 and m.weight.shape[0] % (2 * RPS) == 0 and (m.weight.shape[1] * 8) % (LANES * VPT) == 0]
    mx.eval(model.parameters())
    weights = sum(m.weight.shape[0] * m.weight.shape[1] * 8 for m in layers)
    gb = sum(m.weight.nbytes for m in layers) / 1e9  # the packed weights alone (step 1 reads only these)
    results = {"environment": environment(), "layers": len(layers), "weights_billion": weights / 1e9, "weight_gb": gb, "ms": {}}
    print(f"{len(layers)} layers, {weights / 1e9:.2f} billion weights, {gb:.2f} GB; total ms to run all layers once")
    for rows in ROWS:
        xs = {k: mx.random.normal((rows, k)).astype(mx.bfloat16) for k in {m.weight.shape[1] * 8 for m in layers}}
        mx.eval(list(xs.values()))
        for cached in (False, True):
            row = []
            for level in range(len(STEPS)):
                kernel = mx.fast.metal_kernel(name=f"limits_m{rows}_l{level}_c{int(cached)}", output_names=["y"],
                                              input_names=["w", "scales", "biases", "x"], source=source(rows, level, cached))

                def run() -> None:
                    mx.eval([kernel(inputs=[m.weight, m.scales, m.biases, xs[m.weight.shape[1] * 8]],
                                    template=[("K", m.weight.shape[1] * 8), ("N", m.weight.shape[0])],
                                    grid=(LANES, 2 * (m.weight.shape[0] // (2 * RPS)), 1), threadgroup=(LANES, 2, 1),
                                    output_shapes=[(rows, m.weight.shape[0])], output_dtypes=[mx.bfloat16])[0] for m in layers])

                row.append(best_ms(run))
            label = f"{rows} rows, {'cached' if cached else 'real weights'}"
            results["ms"][label] = dict(zip(STEPS, row))
            print(f"  {label:22s} " + "  ".join(f"{t:6.2f}" for t in row), flush=True)
    real = results["ms"]["1 rows, real weights"]
    stream_ms = real[STEPS[1]] - real[STEPS[0]]
    results["memory_stream_gb_per_s"] = gb / stream_ms * 1000
    print(f"streaming the weights from memory: {stream_ms:.2f} ms for {gb:.2f} GB = {results['memory_stream_gb_per_s']:.0f} GB/s; "
          f"one row, full kernel: {real[STEPS[5]]:.2f} ms")
    save("kernel_limits", args.model, results)


if __name__ == "__main__":
    main()
