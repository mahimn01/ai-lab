"""Fine-tune Qwen3.5-family models with their recurrent layers worked out 64 tokens at a time.

The problem (found 2026-11-12)
  Three of every four layers in a Qwen3.5 model are "gated delta" layers: they carry a
  running summary of the text forward one token at a time. For generation mlx-lm runs
  that update with a fast GPU kernel. In training mode it switches to a plain Python
  loop over the tokens, because its own kernel has no gradient. That loop is why
  fine-tuning these models was 3.5 to 7 times slower per token than Qwen3 models and
  needed gradient checkpointing to fit in memory.

The fix
  `install()` points mlx-lm's training path at `chunked_update` below: the same
  recurrence worked out 64 tokens at a time with ordinary matrix multiplies, which MLX
  can differentiate by itself. It is exact, not an approximation (the algebra is in
  the function's comment), and the outputs and gradients agree with mlx-lm's loop to
  rounding (tests/test_gated_delta.py).

  MLX itself (0.32.4) also ships the update as a built-in with a gradient,
  `mx.fast.gated_delta_update`. Its forward pass is fast everywhere, but on chips
  before the M5 its backward pass walks token by token and stores every intermediate
  state. Measured on one layer of the 4B model's shape, 4 x 512 tokens, forward plus
  backward: mlx-lm's loop 637 ms and 10.2 GB, MLX's built-in 299 ms and 8.4 GB,
  `chunked_update` 25 ms and 1.3 GB (SPEED-RESULTS.md section 13).

Scope
  Only the training path changes. Generation keeps mlx-lm's own kernel. A model that
  uses a variant not covered here (a per-channel decay, or a padding mask) runs
  mlx-lm's original loop as before.

Use:     from lab.finetune import gated_delta; gated_delta.install()   (lab/finetune/lora.py does this)
Changes: replaces one function inside the running mlx-lm, in memory only. `uninstall()` puts it back.
"""

from __future__ import annotations

import sys

import mlx.core as mx
import numpy as np
from mlx_lm.models import gated_delta as _mlx_lm

_original = _mlx_lm.gated_delta_update
CHUNK = 64  # tokens worked out together; must be a power of two
_index_cache: dict[tuple[int, int], mx.array] = {}


def _lower_left_blocks(c: int, h: int) -> mx.array:
    """Positions, in a flattened c x c matrix, of the lower-left h x h corner of every 2h x 2h diagonal block."""
    if (c, h) not in _index_cache:
        j = np.arange(c // (2 * h))[:, None, None] * 2 * h
        a, b = np.arange(h)[None, :, None], np.arange(h)[None, None, :]
        _index_cache[(c, h)] = mx.array(((j + h + a) * c + (j + b)).astype(np.int32))
    return _index_cache[(c, h)]


def _unit_lower_inverse(m: mx.array) -> mx.array:
    """Inverse of matrices [..., c, c] that are lower triangular with ones on the diagonal.

    Built from the bottom up: the inverse of a block [[A, 0], [B, D]] is [[A', 0], [-D' B A', D']],
    where A' and D' are the inverses of the two halves. Every level is one batched step, so the
    whole inverse takes log2(c) steps, and unlike a power series it stays accurate when the rows
    are strongly related.
    """
    c = m.shape[-1]
    flat = m.reshape(*m.shape[:-2], c * c)
    inv = mx.ones((*m.shape[:-2], c, 1, 1), dtype=m.dtype)  # the 1 x 1 diagonal blocks
    h = 1
    while h < c:
        corner = mx.take(flat, _lower_left_blocks(c, h), axis=-1)  # B of every block: [..., c / 2h, h, h]
        first, second = inv[..., 0::2, :, :], inv[..., 1::2, :, :]
        low = -(second @ corner @ first)
        top = mx.concatenate([first, mx.zeros_like(first)], axis=-1)
        inv = mx.concatenate([top, mx.concatenate([low, second], axis=-1)], axis=-2)
        h *= 2
    return inv[..., 0, :, :]


def _split(x: mx.array, n: int, axis: int) -> list[mx.array]:
    """mx.split(x, n, axis) whose backward pass stays cheap for any n.

    MLX (0.32.4, ops.cpp) turns a split into more than 10 parts into separate slices, and the
    backward pass of each slice rebuilds the whole array: the cost then grows with the square of
    n. Splitting into at most 10 groups first, then each group again, keeps every split small.
    """
    if n <= 10:
        return mx.split(x, n, axis=axis)
    per_part = x.shape[axis] // n
    group = -(-n // 10)  # parts per group, rounded up
    cuts = list(range(group * per_part, n * per_part, group * per_part))
    return [piece for part in mx.split(x, cuts, axis=axis) for piece in _split(part, part.shape[axis] // per_part, axis)]


def chunked_update(q: mx.array, k: mx.array, v: mx.array, g: mx.array, beta: mx.array,
                   state: mx.array, chunk: int = CHUNK) -> tuple[mx.array, mx.array]:
    """The gated delta recurrence, `chunk` tokens at a time, in 32-bit maths. Same results as mlx-lm's loop.

    Shapes as in mlx-lm: q, k [B, T, Hk, Dk]; v [B, T, Hv, Dv]; g, beta [B, T, Hv]; state [B, Hv, Dv, Dk].

    One step of the recurrence, for one head, is
        u_t = beta_t * (v_t - g_t * S_{t-1} k_t)        the correction written at step t
        S_t = g_t * S_{t-1} + u_t k_t^T                 the new state
        y_t = S_t q_t                                   the output
    Inside a chunk that starts from state S_0, write gamma_t = g_1 * ... * g_t. Unrolling gives
        S_t = gamma_t S_0 + sum over i <= t of (gamma_t / gamma_i) u_i k_i^T
    and putting that into the first line gives a triangular set of equations for the u's:
        u_t + beta_t * sum over i < t of (gamma_t / gamma_i) (k_i . k_t) u_i = beta_t (v_t - gamma_t S_0 k_t)
    The matrix on the left does not depend on S_0, so it is inverted for all chunks at once.
    Only five matrix multiplies per chunk have to run in order, to carry the state forward.
    """
    if chunk < 1 or chunk & (chunk - 1):
        raise ValueError(f"chunk must be a power of two, not {chunk}")
    B, T, Hk, _ = q.shape
    Hv = v.shape[-2]
    out_dtype = q.dtype
    if Hv // Hk > 1:  # several value heads share one key head
        q, k = mx.repeat(q, Hv // Hk, -2), mx.repeat(k, Hv // Hk, -2)
    pad = (-T) % chunk
    n = (T + pad) // chunk

    def chunks(x: mx.array) -> mx.array:  # [B, T, H, ...] -> [B, H, n, chunk, ...], padded with steps that do nothing
        x = x.astype(mx.float32)
        if pad:
            x = mx.concatenate([x, mx.zeros((B, pad, *x.shape[2:]), dtype=mx.float32)], axis=1)
        x = mx.moveaxis(x, 1, 2)
        return x.reshape(B, Hv, n, chunk, *x.shape[3:])

    q, k, v, beta = chunks(q), chunks(k), chunks(v), chunks(beta)
    log_gamma = mx.cumsum(chunks(mx.log(mx.maximum(g.astype(mx.float32), 1e-26))), axis=-1)  # [B, H, n, chunk]
    gamma = mx.exp(log_gamma)[..., None]
    on_or_below = mx.tril(mx.ones((chunk, chunk), dtype=mx.bool_))
    below = mx.tril(mx.ones((chunk, chunk), dtype=mx.bool_), k=-1)
    # ratio[t, i] = gamma_t / gamma_i for i <= t, else 0 (the difference is zeroed first so exp cannot overflow)
    ratio = mx.where(on_or_below, mx.exp(mx.where(on_or_below, log_gamma[..., :, None] - log_gamma[..., None, :], 0.0)), 0.0)
    kt = mx.swapaxes(k, -1, -2)
    system = mx.where(below, beta[..., :, None] * ratio * (k @ kt), 0.0) + mx.eye(chunk)
    solve = _unit_lower_inverse(system)  # [B, H, n, chunk, chunk]
    read = ratio * (q @ kt)  # read[t, i] = (gamma_t / gamma_i) (q_t . k_i) for i <= t
    carry = mx.exp(log_gamma[..., -1:] - log_gamma)[..., None]  # gamma_end / gamma_i

    state = state.astype(mx.float32)
    outputs = []
    # One piece per chunk, cut once: indexing inside the loop would make the backward pass rebuild
    # a full-size array for every chunk, which grows with the square of the length.
    pieces = zip(*([piece.squeeze(2) for piece in _split(x, n, 2)]
                   for x in (solve, beta[..., None], v, gamma, k, q, read, carry)))
    for solve_c, beta_c, v_c, gamma_c, k_c, q_c, read_c, carry_c in pieces:  # the only part that has to go in order
        st = mx.swapaxes(state, -1, -2)
        u = solve_c @ (beta_c * (v_c - gamma_c * (k_c @ st)))
        outputs.append(gamma_c * (q_c @ st) + read_c @ u)
        state = gamma_c[..., -1:, :] * state + mx.swapaxes(carry_c * u, -1, -2) @ k_c
    y = mx.concatenate(outputs, axis=2)[:, :, :T]
    return mx.moveaxis(y, 1, 2).astype(out_dtype), state


def gated_delta_update(q: mx.array, k: mx.array, v: mx.array, a: mx.array, b: mx.array, A_log: mx.array,
                       dt_bias: mx.array, state: mx.array | None = None, mask: mx.array | None = None, *,
                       use_kernel: bool = True, lower_bound: float | None = None,
                       allow_neg_eigval: bool = False) -> tuple[mx.array, mx.array]:
    """mlx-lm's gated_delta_update, with the training path (use_kernel=False) on `chunked_update`."""
    if use_kernel:  # generation: mlx-lm's own kernel, unchanged
        return _original(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel=True, lower_bound=lower_bound,
                         allow_neg_eigval=allow_neg_eigval)
    # The same preparation as mlx-lm: beta is the update rate, g the decay, both per token and head.
    beta = mx.sigmoid(b) * (2.0 if allow_neg_eigval else 1.0)
    g = (_mlx_lm.compute_g(A_log, a, dt_bias) if lower_bound is None
         else _mlx_lm.compute_lower_bound_g(A_log, a, dt_bias, lower_bound))
    if g.ndim != 3 or mask is not None:  # a decay per channel, or padding: not covered here, so use mlx-lm's loop
        return _original(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel=False, lower_bound=lower_bound,
                         allow_neg_eigval=allow_neg_eigval)
    if state is None:
        state = mx.zeros((q.shape[0], v.shape[-2], v.shape[-1], q.shape[-1]), dtype=mx.float32)
    return chunked_update(q, k, v, g, beta, state)


def _modules_using(fn) -> list:
    return [m for name, m in list(sys.modules.items())
            if name.startswith("mlx_lm.models") and getattr(m, "gated_delta_update", None) is fn]


def install() -> int:
    """Point every loaded mlx-lm model file at the faster training path. Returns how many were changed."""
    changed = _modules_using(_original)
    for module in changed:
        module.gated_delta_update = gated_delta_update
    return len(changed)


def uninstall() -> int:
    changed = _modules_using(gated_delta_update)
    for module in changed:
        module.gated_delta_update = _original
    return len(changed)
