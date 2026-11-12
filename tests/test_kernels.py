"""The multi-row quantized matmul kernel must give the same numbers as MLX's own kernels.

Both are checked against an exact reference: the weights fully decoded to 32-bit floats and
multiplied in 32-bit. The outputs are 16-bit, so "the same" means equal up to rounding.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from lab.kernels import qmm

ROWS = (2, 3, 4, 5, 7, 8, 9, 12, 15, 16, 17, 18)


def make_layer(n: int, k: int, bits: int, dtype: mx.Dtype, group: int = 64) -> tuple[mx.array, mx.array, mx.array]:
    w, scales, biases = mx.quantize(mx.random.normal((n, k)) * 0.05, group_size=group, bits=bits)
    return w, scales.astype(dtype), biases.astype(dtype)


def reference(x: mx.array, w: mx.array, scales: mx.array, biases: mx.array, bits: int, group: int = 64) -> mx.array:
    full = mx.dequantize(w, scales.astype(mx.float32), biases.astype(mx.float32), group_size=group, bits=bits)
    return x.astype(mx.float32) @ full.T


def close(y: mx.array, ref: mx.array) -> bool:
    """Within 16-bit rounding: 1% of each value plus a little of the largest one."""
    return mx.all(mx.abs(y.astype(mx.float32) - ref) <= 0.01 * mx.abs(ref) + 0.003 * mx.abs(ref).max()).item()


@pytest.mark.parametrize("bits", [3, 4, 5, 6, 8])
@pytest.mark.parametrize("shape", [(64, 512), (40, 256), (32, 320), (16, 1024), (48, 832)])
def test_matches_reference_and_mlx(bits: int, shape: tuple[int, int]) -> None:
    # (40, 256): outputs not a multiple of 16.  (32, 320) and (48, 832): a partial last block.
    mx.random.seed(bits * 1000 + shape[0])
    n, k = shape
    w, scales, biases = make_layer(n, k, bits, mx.bfloat16)
    for rows in ROWS:
        x = mx.random.normal((rows, k)).astype(mx.bfloat16)
        ref = reference(x, w, scales, biases, bits)
        builtin = mx.quantized_matmul(x, w, scales, biases, transpose=True, group_size=64, bits=bits)
        ours = qmm.matmul_rows(x, w, scales, biases, bits=bits, group_size=64)
        assert ours.shape == (rows, n) and ours.dtype == mx.bfloat16
        assert close(builtin, ref), f"MLX's own kernel is off at rows={rows}: the reference is wrong"
        assert close(ours, ref), f"rows={rows}"


def test_float16_and_other_group_sizes() -> None:
    mx.random.seed(7)
    for dtype, group in ((mx.float16, 64), (mx.bfloat16, 32), (mx.bfloat16, 128)):
        w, scales, biases = make_layer(32, 768, 4, dtype, group)
        for rows in (2, 5, 8, 11):
            x = mx.random.normal((rows, 768)).astype(dtype)
            ours = qmm.matmul_rows(x, w, scales, biases, bits=4, group_size=group)
            assert close(ours, reference(x, w, scales, biases, 4, group)), f"{dtype} group {group} rows {rows}"


def test_each_row_is_independent() -> None:
    """Row i of a batch must equal the same input run alone: no mixing between prompts."""
    mx.random.seed(11)
    w, scales, biases = make_layer(64, 512, 4, mx.bfloat16)
    x = mx.random.normal((13, 512)).astype(mx.bfloat16)
    together = qmm.matmul_rows(x, w, scales, biases)
    for i in (0, 6, 12):
        pair = mx.stack([x[i], x[i]])
        alone = qmm.matmul_rows(pair, w, scales, biases)[0].astype(mx.float32)
        assert mx.allclose(together[i].astype(mx.float32), alone, rtol=0.01, atol=0.01).item()


def test_install_switches_layers_and_falls_back() -> None:
    mx.random.seed(3)
    net = nn.Sequential(nn.Linear(512, 1024, bias=True), nn.Linear(1024, 96, bias=False))  # second layer: too few outputs
    net.set_dtype(mx.bfloat16)
    nn.quantize(net, group_size=64, bits=4)
    layer = net.layers[0]
    x = mx.random.normal((2, 3, 512)).astype(mx.bfloat16)  # 6 rows, 3-D input like a real decode step
    before = layer(x).astype(mx.float32)

    assert qmm.install(net) == 1 and type(layer) is qmm.RowsQuantizedLinear
    assert type(net.layers[1]) is nn.QuantizedLinear, "an unsupported layer is left alone"
    after = layer(x)
    assert after.shape == (2, 3, 1024)
    assert mx.allclose(after.astype(mx.float32), before, rtol=0.02, atol=0.02).item()

    remembered = len(layer._row_calls)
    for rows in (1, 19, 40):  # 1 row and 19+ rows go to MLX's kernel, and their shapes are not remembered
        other = mx.random.normal((rows, 512)).astype(mx.bfloat16)
        plain = mx.quantized_matmul(other, layer.weight, layer.scales, layer.biases, transpose=True, group_size=64,
                                    bits=4) + layer.bias
        assert mx.array_equal(layer(other), plain).item()
    assert len(layer._row_calls) == remembered

    net.train()  # training needs gradients, which the custom kernel does not have
    grad = mx.grad(lambda inp: layer(inp).astype(mx.float32).sum())(x)
    assert grad.shape == x.shape
    net.eval()

    assert qmm.uninstall(net) == 1 and type(layer) is nn.QuantizedLinear


def test_dense_path_for_many_rows_matches_mlx_and_has_gradients() -> None:
    """256+ rows: decode the weights and use MLX's dense multiply. Same numbers as MLX's kernel, with gradients."""
    mx.random.seed(5)
    net = nn.Sequential(nn.Linear(512, 1024, bias=False))
    net.set_dtype(mx.bfloat16)
    nn.quantize(net, group_size=64, bits=4)
    layer = net.layers[0]
    x = mx.random.normal((4, 96, 512)).astype(mx.bfloat16)  # 384 rows
    plain = layer(x)
    plain_grad = mx.grad(lambda inp: layer(inp).astype(mx.float32).sum())(x)

    assert qmm.install(net) == 1
    assert mx.array_equal(layer(x), plain).item(), "MLX's block kernel rounds decoded weights the same way"
    net.train()  # the dense path is made of MLX's own operations, so it also works in training
    grad = mx.grad(lambda inp: layer(inp).astype(mx.float32).sum())(x)
    assert mx.allclose(grad.astype(mx.float32), plain_grad.astype(mx.float32), rtol=0.02, atol=0.02).item()
    assert not layer._row_calls, "the dense path needs no prepared kernel"
    qmm.uninstall(net)
