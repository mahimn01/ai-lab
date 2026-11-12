"""Inference: prefix detection, lookahead and the float16 switch (fast), and checks on a real model (slow)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
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


def test_lookup_finds_what_followed_a_repeat() -> None:
    from lab.infer.lookahead import _Lookup

    seen = _Lookup(3)
    seen.extend([5, 6, 7, 8, 9, 1, 2, 5, 6, 7])  # "5 6 7" appeared before and was followed by 8 9 1 2
    assert seen.guess(3) == [8, 9, 1]
    assert seen.guess(10) == [8, 9, 1, 2, 5, 6, 7], "a guess can run up to the present, not past it"
    seen.extend([8])
    assert seen.guess(2) == [9, 1], "after 6 7 8 the earlier text continued with 9 1"
    seen.extend([4])
    assert seen.guess(4) == [], "7 8 4 never appeared before"


@pytest.mark.slow
def test_lookahead_writes_the_same_text_in_fewer_passes() -> None:
    """A copy-heavy prompt on a real model: guessing must not change the text, and must save passes."""
    from mlx_lm import load

    from lab.infer.lookahead import generate
    from lab.kernels import qmm

    model, tokenizer = load("mlx-community/Qwen3-0.6B-4bit")
    passage = ("The lab measures before it changes anything. A job that needs more memory than is free pushes "
               "the Mac into swap and runs five to ten times slower. Models stay on the internal disk.")
    prompt = batch.chat_tokens(tokenizer, f"Repeat this passage twice, word for word, and add nothing.\n\n{passage}")
    plain = generate(model, tokenizer, prompt, max_tokens=120, lookahead=False)
    ahead = generate(model, tokenizer, prompt, max_tokens=120)
    assert plain.passes == len(plain.token_ids) and plain.guessed == 0
    assert ahead.token_ids == plain.token_ids, "with MLX's kernels the checked text is exactly the plain text"
    assert ahead.accepted > 20 and ahead.passes < 0.7 * len(ahead.token_ids)

    qmm.install(model)  # the multi-row kernel rounds differently, so allow a late divergence but not an early one
    fast = generate(model, tokenizer, prompt, max_tokens=120)
    qmm.uninstall(model)
    same = next((i for i, (a, b) in enumerate(zip(fast.token_ids, plain.token_ids)) if a != b), len(plain.token_ids))
    assert same >= 40 and fast.accepted > 20


def test_float16_convert_checks_first_and_refuses_values_near_the_limit():
    from lab.infer import float16
    from lab.train.model import GPT, GPTConfig

    mx.random.seed(0)
    model = GPT(GPTConfig(vocab_size=256, n_layer=2, n_head=2, n_embd=32))
    model.wte.weight = mx.random.normal(model.wte.weight.shape)  # wide scores, so the top token is rarely a near-tie
    for block in model.blocks:
        block.attn_out.weight = mx.random.normal(block.attn_out.weight.shape, scale=0.05)
        block.mlp_out.weight = mx.random.normal(block.mlp_out.weight.shape, scale=0.05)
    model.set_dtype(mx.bfloat16)
    prompts = [list(range(1, 60)), np.arange(100, 180)]  # plain lists and numpy arrays are both accepted
    before = model(mx.array(prompts[0])[None]).astype(mx.float32)

    report = float16.convert(model, prompts=prompts)
    assert report["converted_parameters"] > 0 and report["headroom"] > 4 and report["same_top_token"] >= 0.95
    assert model.wte.weight.dtype == mx.float16
    after = model(mx.array(prompts[0])[None]).astype(mx.float32)
    assert mx.abs(after - before).max().item() < 0.05 * mx.abs(before).max().item()
    assert float16.revert(model) == report["converted_parameters"] and model.wte.weight.dtype == mx.bfloat16

    # A model whose values come near float16's limit is refused and left exactly as it was.
    model.blocks[0].mlp_out.weight = model.blocks[0].mlp_out.weight * 1e6
    kept = model(mx.array(prompts[0])[None]).astype(mx.float32)
    with pytest.raises(ValueError, match="not converted"):
        float16.convert(model, prompts=prompts)
    assert model.wte.weight.dtype == mx.bfloat16
    assert mx.array_equal(model(mx.array(prompts[0])[None]).astype(mx.float32), kept).item()


@pytest.mark.slow
def test_float16_convert_on_a_real_model_keeps_its_answers_and_the_lab_kernel(loaded):
    from lab.infer import float16
    from lab.kernels import qmm

    model, tokenizer = loaded
    prompts = mx.array([batch.chat_tokens(tokenizer, QUESTIONS[0], enable_thinking=False)] * 4)  # 4 rows: the lab kernel's range
    qmm.install(model)
    try:
        before = mx.argmax(model(prompts)[:, -1, :], axis=-1)
        report = float16.convert(model, tokenizer)
        assert report["same_top_token"] >= 0.95 and report["headroom"] >= 4 and report["converted_parameters"] > 0
        scores = model(prompts)[:, -1, :]
        assert scores.dtype == mx.float16 and mx.isfinite(scores).all().item()
        assert mx.array_equal(mx.argmax(scores, axis=-1), before).item(), "the same answer token in both formats"
    finally:
        float16.revert(model)
        qmm.uninstall(model)
