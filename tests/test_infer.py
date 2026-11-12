"""Batched inference: prefix detection (fast) and agreement with one-at-a-time generation (slow, real model)."""

from __future__ import annotations

import mlx.core as mx
import pytest

from lab.edit import hooks
from lab.infer import batch

MODEL = "mlx-community/Qwen3.5-0.8B-4bit"
SYSTEM = ("You are a careful assistant for a small research lab. Answer in one short sentence. "
          "Do not add greetings, apologies or follow-up questions. If you are not sure, say so plainly. ") * 3
QUESTIONS = ["What is the capital of France?", "How many legs does a spider have?",
             "What colour do you get by mixing blue and yellow?", "What is 12 times 12?"]


def test_common_prefix_len():
    assert batch.common_prefix_len([[1, 2, 3, 4], [1, 2, 9], [1, 2, 3]]) == 2
    assert batch.common_prefix_len([[1, 2], [1, 2]]) == 2
    assert batch.common_prefix_len([[5], [6]]) == 0


@pytest.fixture(scope="module")
def loaded():
    from mlx_lm import load

    return load(MODEL)


@pytest.mark.slow
def test_batched_and_prefix_reuse_match_single_generation(loaded):
    """Greedy decoding is the same computation three ways, so the first tokens must agree.

    Only the first 12 tokens are compared: tiny rounding differences between batch shapes
    can flip a near-tie later on, after which two texts legitimately drift apart.
    """
    from mlx_lm import stream_generate

    model, tokenizer = loaded
    prompts = [batch.chat_tokens(tokenizer, q, system=SYSTEM, enable_thinking=False) for q in QUESTIONS]
    assert batch.common_prefix_len(prompts) >= 64, "the shared system prompt should give a long common prefix"

    single = []
    for p in prompts:
        # stream_generate also yields the end-of-text token; the batch path leaves it out.
        ids = [r.token for r in stream_generate(model, tokenizer, p, max_tokens=12)]
        single.append([t for t in ids if t not in tokenizer.eos_token_ids])
    plain = batch.generate_batch(model, tokenizer, prompts, max_tokens=12, reuse_prefix=False)
    reused = batch.generate_batch(model, tokenizer, prompts, max_tokens=12, reuse_prefix=True)

    assert plain.prefix_tokens_reused == 0 and reused.prefix_tokens_reused >= 64
    n = len(prompts)
    assert sum(a == b for a, b in zip(single, plain.token_ids)) >= n - 1
    assert sum(a == b for a, b in zip(plain.token_ids, reused.token_ids)) >= n - 1
    assert all(t.strip() for t in reused.texts), "every prompt got an answer"


@pytest.mark.slow
def test_hooks_work_on_a_real_quantized_model(loaded):
    model, tokenizer = loaded
    tokens = mx.array(batch.chat_tokens(tokenizer, QUESTIONS[0], enable_thinking=False))[None]
    base = model(tokens)
    logits, acts = hooks.capture(model, tokens)
    n_layers = len(hooks.layer_list(model))
    assert sorted(acts) == list(range(n_layers))
    assert acts[0].shape[:2] == tokens.shape
    assert mx.allclose(logits, base, atol=1e-3).item(), "recording activations does not change the output"
    with hooks.hook(model, n_layers // 2, lambda out: mx.zeros_like(out)):
        assert mx.abs(model(tokens) - base).max().item() > 0.1, "zeroing a middle layer changes the output"
    assert mx.allclose(model(tokens), base, atol=1e-3).item(), "the model is restored afterwards"
