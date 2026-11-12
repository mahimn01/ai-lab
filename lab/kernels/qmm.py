"""Faster quantized matmul on Apple GPUs: a custom kernel for 2-18 rows, and a dense path for 256+ rows.

The problem (measured on the M1 Max, MLX 0.32.4, SPEED-RESULTS.md section 9)
  Every weight table in a quantized model is used through y = x @ W.T, where x has one
  row per prompt being decoded. MLX has two kernels for this on M1-generation GPUs:
    2-6 rows    it runs the one-row kernel once per row, so every row decodes all the
                weights again: cost grows in step with the row count.
    7-31 rows   it switches to a kernel that always works on a block of 32 rows,
                padding with zeros: 8 rows cost the same as 32.

What this kernel does differently
  One pass over the weights serves all the rows. Each GPU thread keeps its slice of
  the inputs of every row in registers, decodes each weight once (w = scale * q + bias;
  3, 4, 5, 6 or 8 bits per weight), and reuses it for all rows. Up to 8 rows go through in one
  pass; 9-18 rows are split into 2 or 3 equal chunks. From 19 rows MLX's block kernel
  is as fast and is used as before. One row is left to MLX too: that case is already
  limited by how fast the Mac can read the weights from memory (about 375 GB/s).

Many rows (256 and more: reading prompts, fine-tuning)
  Measured here (SPEED-RESULTS.md section 11): MLX's quantized block kernel reaches about
  half of this GPU's arithmetic limit however many rows it gets, because it decodes each
  block of weights again for every 32 rows. MLX's dense kernel reaches 71-87%. So for 256
  rows or more the layer decodes its weights to a temporary dense matrix (0.24 ms for a
  34-million-weight layer) and uses the dense kernel. Prompts are read 1.2-1.33x faster,
  and the result is the same bit for bit in most cases, because MLX's block kernel also
  rounds the decoded weights to 16 bits (only the order of the sums can differ). This
  path uses only MLX's own operations, so it has gradients and also runs in training
  mode. Its backward pass decodes the weights a second time instead of keeping the
  decoded copy alive between forward and backward (that copy cost 3.5 GB when
  fine-tuning a 4B model).

Why the code is generated
  The kernel source is written out as plain scalars for one exact row count. With
  loops and arrays the Metal compiler moved the values to slow stack memory (10x
  slower, measured). A thread on this GPU has 128 registers, and the more a kernel
  uses, the fewer threads the GPU runs at once.

Accuracy
  Sums are done in 32-bit floats like MLX's kernels, in a different order, so results
  differ from MLX's by rounding only (tests/test_kernels.py checks both against an
  exact 32-bit reference).

Use
  from lab.kernels import qmm
  qmm.install(model)      # every supported quantized layer now uses it for 2-18 rows
  qmm.uninstall(model)    # back to MLX's kernels
The 2-18 row kernel is for inference only: it has no gradient, so in training mode those
row counts fall back to MLX's kernels. The many-rows path has gradients and stays on.
Changes nothing on disk.

Where it was measured
  Only on the M1 Max (GPU "applegpu_g13s"). MLX picks different kernels on M3 and
  later chips, so on another Mac run `uv run python -m bench.kernel_rows` before
  trusting the speed-up. `install` checks the kernel's numbers against MLX's on a
  small layer first and refuses to switch anything if they disagree.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

LANES = 32  # threads in one SIMD group on Apple GPUs
VPT = 8  # inputs each thread owns per block
BLOCK = LANES * VPT  # inputs covered by one SIMD group per loop step
RPS = 4  # outputs each SIMD group produces (8 halves the reads of x but needs too many registers)
MAX_ROWS = 18  # above this MLX's 32-row block kernel is as fast or faster (measured)
DENSE_MIN_ROWS = 256  # from here decoding the weights and using MLX's dense kernel beats its quantized kernel
DENSE_MAX_BYTES = 2 * 2**30  # a table whose decoded copy would be bigger than this stays on MLX's kernel
# In float16 MLX's own kernel is nearly as fast as the many-rows path (3-13% slower on a 9B model), and on a 27B
# model, which fills most of this Mac's memory, the many-rows path was the slower one (140 against 154 tok/s).
DENSE_FLOAT16_MEMORY_SHARE = 1 / 3  # float16 models with more weights than this share of memory skip the many-rows path
MIN_OUTPUTS = 1024  # smaller layers have too few threads to fill the GPU; MLX's kernel is faster there
MSL_TYPES = {mx.bfloat16: "bfloat16_t", mx.float16: "half"}


def plan(rows: int) -> tuple[int, int]:
    """How to run `rows` rows: (number of chunks, rows per chunk). At most 8 rows fit in one pass."""
    chunks = -(-rows // 8)
    return chunks, -(-rows // chunks)


def _source(*, bits: int, group: int, chunk: int, pair: bool, tail: bool, clamp: int, T: str) -> str:
    """Metal source for one exact shape of the kernel.

    chunk  rows handled per pass
    pair   read two 16-bit inputs with one 32-bit load (bfloat16 only)
    tail   K is not a multiple of BLOCK, so a last partial block is needed
    clamp  how many trailing rows of the last chunk lie past the real row count MT
    """
    w_bytes = VPT * bits // 8  # weight bytes one thread reads per output per block

    def x_base(m: int) -> str:
        return f"(xp + o_{m})" if m >= chunk - clamp else f"(xp + {m} * K)"

    body: list[str] = []
    for m in range(chunk):  # the thread's slice of every row of x, into registers
        if pair:
            for j in range(VPT // 2):
                body.append(f"uint xw_{m}_{j} = ((const device uint32_t*){x_base(m)})[{j}];")
                body.append(f"float x_{m}_{2 * j} = as_type<float>(xw_{m}_{j} << 16);")
                body.append(f"float x_{m}_{2 * j + 1} = as_type<float>(xw_{m}_{j} & 0xffff0000u);")
        else:
            body.extend(f"float x_{m}_{i} = float({x_base(m)}[{i}]);" for i in range(VPT))
    for r in range(RPS):  # one output value at a time: decode its weights once, use them for every row
        body.append("{")
        body.append(f"const device uint8_t* wl = ws + {r} * K_w;")
        body.append(f"float s = float(sl[{r} * K_g]); float b = float(bl[{r} * K_g]);")
        if bits == 4:
            body.append("float s16 = s * 0.0625f;")  # the high 4 bits are used unshifted, so scale by 1/16
            for i in range(VPT // 2):
                body.append(f"uint8_t wb{i} = wl[{i}];")
                body.append(f"float q_{2 * i} = s * float(wb{i} & 0x0f) + b;")
                body.append(f"float q_{2 * i + 1} = s16 * float(wb{i} & 0xf0) + b;")
        elif bits == 8:
            body.extend(f"float q_{i} = s * float(wl[{i}]) + b;" for i in range(VPT))
        else:  # 3, 5, 6 bits: the values sit back to back in the bytes, lowest bits first
            body.extend(f"uint wb{i} = wl[{i}];" for i in range(w_bytes))
            for i in range(VPT):
                first, shift = divmod(bits * i, 8)
                field = f"(wb{first} >> {shift})"
                if shift + bits > 8:  # the value continues in the next byte
                    field = f"({field} | (wb{first + 1} << {8 - shift}))"
                body.append(f"float q_{i} = s * float({field} & {(1 << bits) - 1}u) + b;")
        body.extend(f"r_{m}_{r} = fma(x_{m}_{i}, q_{i}, r_{m}_{r});" for m in range(chunk) for i in range(VPT))
        body.append("}")
    block = "\n            ".join(body)

    lines = [
        "const uint3 tid = threadgroup_position_in_grid;",
        "const uint simd_gid = simdgroup_index_in_threadgroup;",
        "const uint simd_lid = thread_index_in_simdgroup;",
        f"const int K_w = K * {bits} / 8;",
        f"const int K_g = K / {group};",
        f"const int out_row = tid.y * {2 * RPS} + simd_gid * {RPS};",
        f"const int row0 = tid.x * {chunk};",
        f"const device uint8_t* ws = (const device uint8_t*)w + out_row * K_w + simd_lid * {w_bytes};",
        f"const device {T}* sl = scales + out_row * K_g + (simd_lid * {VPT}) / {group};",
        f"const device {T}* bl = biases + out_row * K_g + (simd_lid * {VPT}) / {group};",
        f"const device {T}* xp = x + row0 * K + simd_lid * {VPT};",
    ]
    lines += [f"const int o_{m} = (min(row0 + {m}, MT - 1) - row0) * K;" for m in range(chunk - clamp, chunk)]
    lines += [f"float r_{m}_{r} = 0;" for m in range(chunk) for r in range(RPS)]
    step = f"ws += {LANES * w_bytes}; sl += {BLOCK // group}; bl += {BLOCK // group}; xp += {BLOCK};"
    lines.append(f"for (int k = 0; k + {BLOCK} <= K; k += {BLOCK}) {{\n            {block}\n            {step}\n        }}")
    if tail:  # the last partial block: only the threads whose slice is still inside K take part
        lines.append(f"if ((K / {BLOCK}) * {BLOCK} + int(simd_lid) * {VPT} < K) {{\n            {block}\n        }}")
    lines += [f"r_{m}_{r} = simd_sum(r_{m}_{r});" for m in range(chunk) for r in range(RPS)]
    lines.append("if (simd_lid == 0) {")
    for m in range(chunk):
        guard = f"if (row0 + {m} < MT) " if m >= chunk - clamp else ""
        lines += [f"    {guard}y[(row0 + {m}) * N + out_row + {r}] = static_cast<{T}>(r_{m}_{r});" for r in range(RPS)]
    lines.append("}")
    return "\n        ".join(lines)


_kernels: dict[tuple, object] = {}


def supported(*, bits: int, group_size: int, mode: str, K: int, N: int, dtype: mx.Dtype) -> bool:
    """True if this kernel can run a layer of this shape (otherwise MLX's kernels are used)."""
    return (mode == "affine" and bits in (3, 4, 5, 6, 8) and group_size in (32, 64, 128) and dtype in MSL_TYPES
            and K % group_size == 0 and K >= BLOCK and N % (2 * RPS) == 0)


def prepare(shape: tuple[int, ...], dtype: mx.Dtype, N: int, *, bits: int, group_size: int) -> tuple[object, dict]:
    """The compiled kernel and the fixed call arguments for inputs of this shape ([..., K], 2-18 rows in total)."""
    K, rows = shape[-1], math.prod(shape[:-1])
    chunks, chunk = plan(rows)
    pair, tail, clamp = dtype == mx.bfloat16, K % BLOCK != 0, chunks * chunk - rows
    key = (bits, group_size, chunk, pair, tail, clamp, dtype)
    kernel = _kernels.get(key)
    if kernel is None:
        T = MSL_TYPES[dtype]
        kernel = _kernels[key] = mx.fast.metal_kernel(
            name=f"qmm_rows_b{bits}_g{group_size}_m{chunk}_t{int(tail)}_c{clamp}_{T}",
            input_names=["w", "scales", "biases", "x"], output_names=["y"],
            source=_source(bits=bits, group=group_size, chunk=chunk, pair=pair, tail=tail, clamp=clamp, T=T))
    template = [("K", K), ("N", N)] + ([("MT", rows)] if clamp else [])  # MT: the real row count
    return kernel, {"template": template, "grid": (LANES * chunks, 2 * (N // (2 * RPS)), 1),
                    "threadgroup": (LANES, 2, 1), "output_shapes": [(*shape[:-1], N)], "output_dtypes": [dtype]}


def matmul_rows(x: mx.array, w: mx.array, scales: mx.array, biases: mx.array, *, bits: int = 4,
                group_size: int = 64) -> mx.array:
    """y = x @ W.T for x of shape [..., K] with 2 to MAX_ROWS rows in total, W in MLX's quantized form.

    The caller checks `supported(...)` and the row count; x, scales and biases share one dtype.
    The output keeps x's leading shape.
    """
    kernel, fixed = prepare(tuple(x.shape), x.dtype, w.shape[0], bits=bits, group_size=group_size)
    return kernel(inputs=[w, scales, biases, x], **fixed)[0]


def _dense(layer: nn.Module):
    """x -> x @ W.T through the decoded weights. The gradient for x decodes them again rather than storing them."""

    def decoded() -> mx.array:
        return mx.dequantize(layer["weight"], layer["scales"], layer["biases"], group_size=layer.group_size,
                             bits=layer.bits)

    @mx.custom_function
    def forward(x: mx.array) -> mx.array:
        return x @ decoded().T

    @forward.vjp
    def backward(x: mx.array, cotangent: mx.array, output: mx.array) -> mx.array:
        return cotangent @ decoded()

    return forward


def _fast(layer: nn.Module, x: mx.array) -> mx.array | None:
    """The fast path for a quantized layer, or None when MLX's own kernel should run."""
    rows = x.size // layer._row_k
    if rows >= DENSE_MIN_ROWS and layer._row_dense is not None and x.dtype == layer["scales"].dtype:  # decode once, dense multiply
        return layer._row_dense(x)
    if rows < 2 or rows > MAX_ROWS or layer.training:  # checked first: one row is the common single-prompt case
        return None
    key = (x.shape, x.dtype)
    call = layer._row_calls.get(key)  # prepared once per input shape (only shapes of 2-18 rows are remembered)
    if call is None:
        ok = x.dtype == layer["scales"].dtype and x.shape[-1] == layer._row_k
        call = layer._row_calls[key] = prepare(tuple(x.shape), x.dtype, layer["weight"].shape[0], bits=layer.bits,
                                               group_size=layer.group_size) if ok else False
    if call is False:
        return None
    kernel, fixed = call
    return kernel(inputs=[layer["weight"], layer["scales"], layer["biases"], x], **fixed)[0]


class RowsQuantizedLinear(nn.QuantizedLinear):
    """nn.QuantizedLinear that uses the multi-row kernel for 2-18 rows and the dense path for 256+."""

    def __call__(self, x: mx.array) -> mx.array:
        y = _fast(self, x)
        if y is None:
            return nn.QuantizedLinear.__call__(self, x)
        return y + self["bias"] if "bias" in self else y


class RowsQuantizedEmbedding(nn.QuantizedEmbedding):
    """nn.QuantizedEmbedding whose tied output projection uses the multi-row kernel."""

    def as_linear(self, x: mx.array) -> mx.array:
        y = _fast(self, x)
        return nn.QuantizedEmbedding.as_linear(self, x) if y is None else y


_SWAP = {nn.QuantizedLinear: RowsQuantizedLinear, nn.QuantizedEmbedding: RowsQuantizedEmbedding}
_checked: set[tuple] = set()


def self_test(*, bits: int, group_size: int, dtype: mx.Dtype) -> None:
    """Run the kernel on a small random layer and compare with MLX's kernel. Raises if they disagree.

    Covers one full pass (5 rows) and a split with a partial last chunk (11 rows). Uses its own
    random key, so the caller's random state is untouched.
    """
    if (bits, group_size, dtype) in _checked:
        return
    keys = mx.random.split(mx.random.key(0), 3)
    w, scales, biases = mx.quantize(mx.random.normal((16, 2 * BLOCK), key=keys[0]) * 0.05, group_size=group_size, bits=bits)
    scales, biases = scales.astype(dtype), biases.astype(dtype)
    for rows, key in ((5, keys[1]), (11, keys[2])):
        x = mx.random.normal((rows, 2 * BLOCK), key=key).astype(dtype)
        want = mx.quantized_matmul(x, w, scales, biases, transpose=True, group_size=group_size, bits=bits).astype(mx.float32)
        got = matmul_rows(x, w, scales, biases, bits=bits, group_size=group_size).astype(mx.float32)
        if not mx.all(mx.abs(got - want) <= 0.02 * mx.abs(want) + 0.01 * mx.abs(want).max()).item():
            raise RuntimeError(f"lab kernel disagrees with MLX's kernel ({bits}-bit, group {group_size}, {dtype}, "
                               f"{rows} rows) on this machine; nothing was switched")
    _checked.add((bits, group_size, dtype))


def layer_inputs(layer: nn.Module) -> int:
    """K, the number of inputs of a quantized layer (its weights are stored packed, 32 bits per word)."""
    return layer["weight"].shape[1] * 32 // layer.bits


def takes(layer: nn.Module) -> bool:
    """True if `install` would switch this layer to the multi-row kernel."""
    if type(layer) not in _SWAP or "biases" not in layer:
        return False
    N = layer["weight"].shape[0]
    return N >= MIN_OUTPUTS and supported(bits=layer.bits, group_size=layer.group_size, mode=getattr(layer, "mode", "affine"),
                                          K=layer_inputs(layer), N=N, dtype=layer["scales"].dtype)


def install(model: nn.Module, many_rows: bool | None = None) -> int:
    """Switch every supported quantized layer of `model` to the lab's paths. Returns how many were switched.

    many_rows says whether layers also decode their weights and use the dense multiply for 256 or more
    rows. None decides by itself: yes, except for a float16 model whose weights take more than a third
    of the Mac's memory (see DENSE_FLOAT16_MEMORY_SHARE).
    """
    chosen = [m for _, m in model.named_modules() if takes(m)]
    for combo in {(m.bits, m.group_size, m["scales"].dtype) for m in chosen}:  # check first, switch after
        self_test(bits=combo[0], group_size=combo[1], dtype=combo[2])
    big = sum(m["weight"].nbytes for m in chosen) > DENSE_FLOAT16_MEMORY_SHARE * mx.device_info()["memory_size"]
    for module in chosen:
        module.__class__ = _SWAP[type(module)]  # same object and weights; only the call path changes
        # plain attributes (not parameters): the prepared calls per input shape, and K to spot a single row
        object.__setattr__(module, "_row_calls", {})
        object.__setattr__(module, "_row_k", layer_inputs(module))
        dense = many_rows if many_rows is not None else not (big and module["scales"].dtype == mx.float16)
        fits = module["weight"].shape[0] * layer_inputs(module) * 2 <= DENSE_MAX_BYTES
        object.__setattr__(module, "_row_dense", _dense(module) if dense and fits else None)
    return len(chosen)


def uninstall(model: nn.Module) -> int:
    """Undo `install`. Returns how many layers were switched back."""
    back = {fast: plain for plain, fast in _SWAP.items()}
    switched = 0
    for _, module in model.named_modules():
        if type(module) in back:
            module.__class__ = back[type(module)]
            for name in ("_row_calls", "_row_k", "_row_dense"):
                module.__dict__.pop(name, None)
            switched += 1
    return switched
