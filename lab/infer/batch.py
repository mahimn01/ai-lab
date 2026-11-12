"""Fast inference for research workloads: answer many prompts at once.

Why
  One prompt at a time leaves most of the GPU idle. Running many prompts together
  ("batching") gave 7-9x more tokens per second on this Mac (neural-harness, 2026-11-03;
  bench/results/inference.json). Evals, sweeps and data generation are all "many
  prompts", so this is the main speed lever for research.

What it adds on top of mlx-lm's batch generator
  - Shared-prefix reuse: if every prompt starts with the same tokens (a long system
    prompt, few-shot examples, a document), that part is read by the model once and its
    cache is shared, instead of once per prompt.
  - Length sorting (on by default), so prompts of similar length are read together.
  - A cap on MLX's buffer cache, which otherwise grew to ~20 GB in long batched runs.

Read-only: nothing is written to disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import batch_generate
from mlx_lm.models.cache import make_prompt_cache

GB = 2**30
# MLX keeps freed GPU buffers for reuse. Uncapped, that cache grew to ~20 GB in a
# neural-harness run (2026-11-03) and thrashed the Mac. The cap never changes results.
CACHE_LIMIT = 2 * GB
PREFILL_STEP = 512


@dataclass
class BatchResult:
    texts: list[str]
    token_ids: list[list[int]]
    prompt_tps: float  # tokens/s while reading prompts
    generation_tps: float  # tokens/s while writing, all sequences added together
    generation_tokens: int
    wall_s: float
    peak_memory_gb: float
    prefix_tokens_reused: int  # length of the shared prefix that was read only once


def chat_tokens(tokenizer, user: str, system: str | None = None, **template_args) -> list[int]:
    """Token ids for one chat turn, using the model's own chat template."""
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template_args)
    return tokenizer.encode(text, add_special_tokens=False)


def common_prefix_len(prompts: list[list[int]]) -> int:
    """Number of leading tokens shared by every prompt."""
    shortest = min(prompts, key=len)
    for i, token in enumerate(shortest):
        if any(p[i] != token for p in prompts):
            return i
    return len(shortest)


def prefix_cache(model: nn.Module, prefix: list[int]) -> list:
    """Read the prefix once and return the model's cache (its memory of those tokens)."""
    cache = make_prompt_cache(model)
    for i in range(0, len(prefix), PREFILL_STEP):
        model(mx.array(prefix[i:i + PREFILL_STEP])[None], cache=cache)
        mx.eval([c.state for c in cache])
    return cache


def generate_batch(model: nn.Module, tokenizer, prompts: list[list[int]], *, max_tokens: int = 256,
                   batch_size: int = 32, sampler: Callable | None = None, reuse_prefix: bool = True,
                   min_prefix: int = 64, sort_by_length: bool = True) -> BatchResult:
    """Generate a completion for every prompt (lists of token ids). Results come back in input order.

    sampler=None means greedy decoding (always the most likely token), which is repeatable.
    sort_by_length reads prompts of similar length together (less padding). Measured once on
    64 maths questions at batch 32: 306 against 281 tokens/s end to end (SPEED-RESULTS.md, 1).
    """
    if not prompts:
        raise ValueError("no prompts given")
    mx.set_cache_limit(CACHE_LIMIT)

    # Shared prefix: keep at least one token of every prompt outside the cached part.
    shared = min(common_prefix_len(prompts), min(len(p) for p in prompts) - 1) if reuse_prefix and len(prompts) > 1 else 0
    if shared < min_prefix:
        shared = 0
    caches = None
    todo = prompts
    if shared:
        cache = prefix_cache(model, prompts[0][:shared])
        caches = [cache] * len(prompts)  # batch_generate copies them; the shared one is not changed
        todo = [p[shared:] for p in prompts]

    order = sorted(range(len(todo)), key=lambda i: -len(todo[i])) if sort_by_length else list(range(len(todo)))
    kwargs = {"sampler": sampler} if sampler is not None else {}
    out = batch_generate(model, tokenizer, [todo[i] for i in order],
                         prompt_caches=[caches[i] for i in order] if caches else None,
                         max_tokens=max_tokens, return_token_ids=True,
                         completion_batch_size=batch_size, prefill_batch_size=min(8, batch_size),
                         prefill_step_size=PREFILL_STEP, **kwargs)
    texts, ids = [""] * len(prompts), [[] for _ in prompts]
    for position, i in enumerate(order):
        texts[i], ids[i] = out.texts[position], out.token_ids[position]
    s = out.stats
    return BatchResult(texts, ids, s.prompt_tps, s.generation_tps, s.generation_tokens, s.wall_time,
                       s.peak_memory, shared)
