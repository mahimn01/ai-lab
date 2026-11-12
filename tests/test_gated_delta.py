"""The faster training path for Qwen3.5's recurrent layers must compute what mlx-lm's loop computes."""

from __future__ import annotations

import mlx.core as mx

from lab.finetune import gated_delta


def inputs(B: int = 2, T: int = 24, Hk: int = 2, Hv: int = 4, D: int = 64) -> dict:
    mx.random.seed(0)
    return {
        "q": (mx.random.normal((B, T, Hk, D)) * 0.1).astype(mx.bfloat16),
        "k": (mx.random.normal((B, T, Hk, D)) * 0.1).astype(mx.bfloat16),
        "v": mx.random.normal((B, T, Hv, D)).astype(mx.bfloat16),
        "a": mx.random.normal((B, T, Hv)).astype(mx.bfloat16),
        "b": mx.random.normal((B, T, Hv)).astype(mx.bfloat16),
        "A_log": mx.random.normal((Hv,)) * 0.1,
        "dt_bias": mx.random.normal((Hv,)) * 0.1,
    }


def close(a: mx.array, b: mx.array, tolerance: float = 0.02) -> bool:
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return (mx.abs(a - b).max() <= tolerance * mx.abs(b).max() + 1e-6).item()


def test_training_path_matches_the_loop_in_outputs_and_gradients() -> None:
    x = inputs()
    rest = (x["a"], x["b"], x["A_log"], x["dt_bias"])
    slow_y, slow_state = gated_delta._original(x["q"], x["k"], x["v"], *rest, use_kernel=False)
    fast_y, fast_state = gated_delta.gated_delta_update(x["q"], x["k"], x["v"], *rest, use_kernel=False)
    assert fast_y.shape == slow_y.shape and fast_y.dtype == slow_y.dtype
    assert fast_state.shape == slow_state.shape and fast_state.dtype == slow_state.dtype
    assert close(fast_y, slow_y) and close(fast_state, slow_state)

    def loss(fn):
        return lambda q, k, v: fn(q, k, v, *rest, use_kernel=False)[0].astype(mx.float32).sum()

    slow_grads = mx.grad(loss(gated_delta._original), argnums=[0, 1, 2])(x["q"], x["k"], x["v"])
    fast_grads = mx.grad(loss(gated_delta.gated_delta_update), argnums=[0, 1, 2])(x["q"], x["k"], x["v"])
    for fast, slow in zip(fast_grads, slow_grads):
        assert close(fast, slow, tolerance=0.05)


def test_install_and_uninstall_swap_the_function_in_mlx_lm() -> None:
    from mlx_lm.models import gated_delta as mlx_lm_module

    gated_delta.uninstall()  # a fine-tuning test that ran earlier leaves it switched on
    assert gated_delta.install() >= 1
    assert mlx_lm_module.gated_delta_update is gated_delta.gated_delta_update
    assert gated_delta.uninstall() >= 1
    assert mlx_lm_module.gated_delta_update is gated_delta._original


def test_chunked_update_matches_the_loop_across_chunk_boundaries() -> None:
    """Longer than one chunk, not a multiple of it, more value heads than key heads, a non-zero starting state."""
    from mlx_lm.models.gated_delta import gated_delta_ops

    mx.random.seed(3)
    B, T, Hk, Hv, D = 2, 150, 2, 4, 64
    q = (mx.random.normal((B, T, Hk, D)) * 0.12).astype(mx.bfloat16)
    k = (mx.random.normal((B, T, Hk, D)) * 0.12).astype(mx.bfloat16)
    v = mx.random.normal((B, T, Hv, D)).astype(mx.bfloat16)
    g = mx.sigmoid(mx.random.normal((B, T, Hv)) + 2.0)
    beta = mx.sigmoid(mx.random.normal((B, T, Hv)))
    state = mx.random.normal((B, Hv, D, D)) * 0.1
    slow_y, slow_state = gated_delta_ops(q, k, v, g, beta, state)
    for chunk in (16, 64):
        fast_y, fast_state = gated_delta.chunked_update(q, k, v, g, beta, state, chunk=chunk)
        assert fast_y.shape == slow_y.shape and close(fast_y, slow_y), f"chunk {chunk}"
        assert close(fast_state, slow_state, tolerance=1e-3), f"chunk {chunk}"


def test_chunked_update_stays_accurate_when_keys_repeat() -> None:
    """The same key at every step is the hardest case for solving a chunk at once."""
    from mlx_lm.models.gated_delta import gated_delta_ops

    mx.random.seed(4)
    B, T, H, D = 1, 64, 2, 64
    key = mx.random.normal((1, 1, H, D))
    k = mx.broadcast_to(key / mx.sqrt((key * key).sum(-1, keepdims=True)), (B, T, H, D)).astype(mx.float32)
    q, v = k, mx.random.normal((B, T, H, D))
    g, beta = mx.full((B, T, H), 0.99), mx.full((B, T, H), 1.0)
    state = mx.zeros((B, H, D, D))
    slow_y, slow_state = gated_delta_ops(q, k, v, g, beta, state)
    fast_y, fast_state = gated_delta.chunked_update(q, k, v, g, beta, state)
    assert close(fast_y, slow_y, tolerance=1e-3) and close(fast_state, slow_state, tolerance=1e-3)


def test_more_than_ten_chunks_match_the_loop_in_outputs_and_gradients() -> None:
    """Above 10 chunks the update cuts its inputs in two levels (MLX's own split changes behaviour there)."""
    from mlx_lm.models.gated_delta import gated_delta_ops

    for n in (1, 10, 11, 19, 101):
        x = mx.arange(n * 6).reshape(2, n, 3)
        parts = gated_delta._split(x, n, 1)
        assert len(parts) == n and all(mx.array_equal(part, x[:, i:i + 1]).item() for i, part in enumerate(parts))

    mx.random.seed(5)
    B, T, H, D = 2, 150, 2, 32
    q, k = mx.random.normal((B, T, H, D)) * 0.15, mx.random.normal((B, T, H, D)) * 0.15
    v = mx.random.normal((B, T, H, D))
    g, beta = mx.sigmoid(mx.random.normal((B, T, H)) + 2.0), mx.sigmoid(mx.random.normal((B, T, H)))
    state = mx.zeros((B, H, D, D))

    def fast(q, k, v):
        return gated_delta.chunked_update(q, k, v, g, beta, state, chunk=8)  # 19 chunks

    def slow(q, k, v):
        return gated_delta_ops(q, k, v, g, beta, state)

    assert close(fast(q, k, v)[0], slow(q, k, v)[0], tolerance=1e-3)
    fast_grads = mx.grad(lambda *a: fast(*a)[0].sum(), argnums=[0, 1, 2])(q, k, v)
    slow_grads = mx.grad(lambda *a: slow(*a)[0].sum(), argnums=[0, 1, 2])(q, k, v)
    for a, b in zip(fast_grads, slow_grads):
        assert close(a, b, tolerance=1e-3)
