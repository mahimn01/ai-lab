"""Speed experiment 19 (groundwork): what one decode step of a small model costs when the history is long.

Background
  The neural-harness study writes up to 32,768 tokens per problem with Qwen3.5-0.8B (dense
  weights). Its runner (mlx-lm's batch generator) reached 337 tokens a second with 16 rows.
  This script takes one step apart to see what a faster engine could and could not change.

What it measures (mlx-lm's own model and batch caches; one row count and number format per run)
  step       one model pass that writes one token for every row, at several history lengths. The
             history is filled with random keys and values of that length (timing does not depend on
             their content) in mlx-lm's padded block, where every row is as long as the longest.
  attention  MLX's attention operation alone on one layer's block, and how fast it reads the keys
             and values in GB/s.
  top 20     finding the 20 highest of the 248,320 scores per row: full sort (what the usual samplers
             do) against the block method of lab/gen/sampler.py.
  compiled   three recurrent layers written as one function, plain and through mx.compile.
  engine     the row engine (lab/gen) on the same model: one whole step (model pass, sampling, bookkeeping)
             with every slot busy at a short history, and its paged attention kernel against MLX's
             attention on the same keys and values, for rows of equal length and of mixed lengths.

Run:     uv run python -m bench.long_step --model ~/Documents/Dev/neural-harness/models/base --rows 16
         (one process per row count: 16, 32, 64; add --dtype float16 for the other number format)
Changes: writes bench/results/long_step.json. Nothing else.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import ArraysCache, BatchKVCache

from bench.common import environment, save
from lab.gen.engine import Engine, Row, _Live
from lab.gen.kv import PAGE, PagePool
from lab.gen.sampler import top_values

LENGTHS = (64, 4096, 16384, 32768)
KV_BUDGET_GB = 7.0     # lengths whose padded block would need more than this are skipped


def timed(fn, reps: int = 7) -> float:
    """Median milliseconds of fn() over reps runs, after two warm-up runs."""
    fn()
    fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2] * 1e3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", type=Path, default=Path.home() / "Documents/Dev/neural-harness/models/base")
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = ap.parse_args()
    dtype, rows = getattr(mx, args.dtype), args.rows

    model, _ = load(str(args.model.expanduser()))
    if dtype != mx.bfloat16:
        model.apply(lambda v: v.astype(dtype) if v.dtype == mx.bfloat16 else v)
        mx.eval(model.parameters())
    text = model.language_model.model
    attention = next(layer.self_attn for layer in text.layers if not layer.is_linear)
    kv_heads, dim, layers = attention.num_key_value_heads, attention.head_dim, sum(not layer.is_linear for layer in text.layers)
    bytes_per_token = 2 * layers * kv_heads * dim * 2                      # keys and values, 16-bit numbers
    entry = {"environment": environment(), "model": str(args.model), "rows": rows, "dtype": args.dtype, "by_length": {}}

    # A real short pass first, so the recurrent layers hold states of the right shape and type.
    cache = [ArraysCache(size=2) if layer.is_linear else BatchKVCache([0] * rows) for layer in text.layers]
    mx.random.seed(0)
    mx.eval(model(mx.random.randint(0, 200_000, (rows, 64)), cache=cache))
    y = mx.random.randint(0, 200_000, (rows,))
    q = (mx.random.normal((rows, attention.num_attention_heads, 1, dim)) * 0.5).astype(dtype)
    for length in LENGTHS:
        if rows * length * bytes_per_token > KV_BUDGET_GB * 2**30:
            continue
        for c in cache:
            if isinstance(c, BatchKVCache):
                c.keys = (mx.random.normal((rows, kv_heads, (length // 256 + 2) * 256, dim)) * 0.5).astype(dtype)
                c.values = (mx.random.normal((rows, kv_heads, (length // 256 + 2) * 256, dim)) * 0.5).astype(dtype)
                c._idx, c.offset = length, mx.array([length] * rows)
                mx.eval(c.keys, c.values)
        block = next(c for c in cache if isinstance(c, BatchKVCache))

        def step() -> None:
            mx.eval(model(y[:, None], cache=cache), [c.state for c in cache])

        def attend() -> None:
            mx.eval(mx.fast.scaled_dot_product_attention(q, block.keys[..., :length, :], block.values[..., :length, :],
                                                         scale=dim ** -0.5, mask=None))

        step_ms, attention_ms = timed(step), timed(attend)
        layer_gb = rows * length * 2 * kv_heads * dim * 2 / 2**30
        entry["by_length"][str(length)] = {"step_ms": step_ms, "tokens_per_s": rows / step_ms * 1e3, "attention_one_layer_ms": attention_ms,
                                           "attention_gb_per_s": layer_gb / (attention_ms / 1e3)}
        print(f"{rows} rows, {args.dtype}, history {length:6,d}: step {step_ms:6.1f} ms ({rows / step_ms * 1e3:6.0f} tok/s); "
              f"attention alone, one layer {attention_ms:5.2f} ms ({layer_gb / (attention_ms / 1e3):4.0f} GB/s)", flush=True)
    cache = block = None      # let the padded blocks go before the next measurements

    # Top 20 of the vocabulary: full sort against the block method.
    scores = mx.random.normal((rows, text.embed_tokens.weight.shape[0]))
    mx.eval(scores)
    sort_ms = timed(lambda: mx.eval(mx.argpartition(-scores, kth=19, axis=-1)[:, :20]))
    block_ms = timed(lambda: mx.eval(top_values(scores, 20)))
    same = bool(mx.array_equal(top_values(scores, 20)[0], mx.sort(scores, axis=-1)[:, ::-1][:, :20]).item())
    entry["top_20"] = {"full_sort_ms": sort_ms, "block_method_ms": block_ms, "same_values": same}
    print(f"top 20 of {scores.shape[1]:,} scores for {rows} rows: full sort {sort_ms:.2f} ms, block method {block_ms:.2f} ms (same values: {same})", flush=True)

    # Three recurrent layers as one function: plain and compiled.
    h = (mx.random.normal((rows, 1, text.embed_tokens.weight.shape[1])) * 0.5).astype(dtype)
    probe = [ArraysCache(size=2) for _ in range(3)]
    for layer, c in zip(text.layers[:3], probe):
        layer.linear_attn(layer.input_layernorm(h), None, c)
    states = [[mx.zeros_like(s) for s in c.cache] for c in probe]
    mx.eval(h, states)

    def three_layers(x: mx.array, all_states: list) -> tuple[mx.array, list]:
        out = []
        for layer, layer_states in zip(text.layers[:3], all_states):
            c = ArraysCache(size=2)
            c.cache = list(layer_states)
            x = x + layer.linear_attn(layer.input_layernorm(x), None, c)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
            out.append(list(c.cache))
        return x, out

    compiled = mx.compile(three_layers)
    plain_ms = timed(lambda: mx.eval(three_layers(h, states)), reps=30)
    compiled_ms = timed(lambda: mx.eval(compiled(h, states)), reps=30)
    entry["three_recurrent_layers"] = {"plain_ms": plain_ms, "compiled_ms": compiled_ms}
    print(f"three recurrent layers, {rows} rows: plain {plain_ms:.2f} ms, compiled {compiled_ms:.2f} ms ({plain_ms / compiled_ms:.2f}x)", flush=True)
    cache = states = h = scores = None

    # The row engine: a whole step at a short history.
    engine = Engine(model, slots=rows, pool_gb=0.5, max_tokens=1024)
    for slot in range(rows):
        engine._admit(_Live(Row(prompt=[1, 2, 3], max_tokens=1024, seed=slot), f"row {slot}"))
    engine_ms = timed(engine._step, reps=30)
    engine.close()
    engine = None

    # Its attention kernel against MLX's attention, one layer: rows of 12,000 tokens, then rows spread from 0 to 24,000.
    kernel = {}
    for name, lengths in (("equal", [12_000] * rows), ("mixed", [max(1, round(24_000 * i / max(rows - 1, 1))) for i in range(rows)])):
        pool = PagePool(layers=1, slots=rows, pages=sum(n // PAGE for n in lengths) + 1, kv_heads=kv_heads, head_dim=dim, dtype=dtype,
                        max_row_tokens=25_000)
        dense_k = (mx.random.normal((rows, kv_heads, max(lengths), dim)) * 0.5).astype(dtype)
        dense_v = (mx.random.normal((rows, kv_heads, max(lengths), dim)) * 0.5).astype(dtype)
        for slot, n in enumerate(lengths):
            pool.load(slot, [(dense_k[slot:slot + 1, :, :n], dense_v[slot:slot + 1, :, :n])])
        mx.eval(pool.keys, pool.values, pool.buf_keys, pool.buf_values, dense_k, dense_v)
        active, query = np.ones(rows, bool), q[:, :, 0, :]
        new = mx.zeros((rows, kv_heads, dim), dtype)
        pool.write(0, new, new)
        paged_ms = timed(lambda: mx.eval(pool.attend(0, query, active, dim ** -0.5)))
        mlx_ms = timed(lambda: mx.eval(mx.fast.scaled_dot_product_attention(q, dense_k, dense_v, scale=dim ** -0.5, mask=None)))
        held_gb, padded_gb = sum(lengths) * 2 * kv_heads * dim * 2 / 2**30, rows * max(lengths) * 2 * kv_heads * dim * 2 / 2**30
        kernel[name] = {"history_gb": held_gb, "paged_ms": paged_ms, "paged_gb_per_s": held_gb / (paged_ms / 1e3),
                        "padded_gb": padded_gb, "mlx_on_padded_block_ms": mlx_ms, "mlx_gb_per_s": padded_gb / (mlx_ms / 1e3)}
        print(f"attention, {rows} rows of {name} length: paged kernel {paged_ms:.2f} ms for {held_gb:.2f} GB ({held_gb / (paged_ms / 1e3):.0f} GB/s); "
              f"MLX on the padded block {mlx_ms:.2f} ms for {padded_gb:.2f} GB ({padded_gb / (mlx_ms / 1e3):.0f} GB/s)", flush=True)
        pool = dense_k = dense_v = None
    entry["engine"] = {"whole_step_ms_at_short_history": engine_ms, "tokens_per_s_at_short_history": rows / engine_ms * 1e3, "attention": kernel}
    print(f"row engine, {rows} slots, short history: whole step {engine_ms:.1f} ms ({rows / engine_ms * 1e3:,.0f} tok/s)", flush=True)
    save("long_step", f"{args.dtype}, {rows} rows", entry)


if __name__ == "__main__":
    main()
