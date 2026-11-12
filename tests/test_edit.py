"""Weight editing and hooks, checked on a tiny random GPT (runs in a second, no downloads)."""

from __future__ import annotations

import json

import mlx.core as mx
import numpy as np
import pytest

from lab.edit import hooks, weights
from lab.train.model import GPT, GPTConfig

CFG = GPTConfig(vocab_size=256, n_layer=3, n_head=2, n_embd=32)


def tiny_model(seed: int = 0) -> GPT:
    mx.random.seed(seed)
    model = GPT(CFG)
    # The lab GPT starts its block outputs at zero; give them values so every layer matters.
    for block in model.blocks:
        block.attn_out.weight = mx.random.normal(block.attn_out.weight.shape, scale=0.05)
        block.mlp_out.weight = mx.random.normal(block.mlp_out.weight.shape, scale=0.05)
    mx.eval(model.parameters())
    return model


@pytest.fixture
def tokens() -> mx.array:
    return mx.array(np.random.default_rng(0).integers(0, CFG.vocab_size, (2, 12)))


def test_delta_to_lora_finds_the_rank_and_rebuilds_the_change():
    rng = np.random.default_rng(1)
    delta = rng.normal(size=(128, 2)) @ rng.normal(size=(2, 32))  # a rank-2 change, shape [out, in]
    a, b, info = weights.delta_to_lora(delta)
    assert info["rank"] == 2
    assert a.shape == (32, 2) and b.shape == (2, 128)
    assert info["relative_residual"] < 1e-5


def test_merged_edit_and_attached_lora_give_the_same_logits(tokens):
    rng = np.random.default_rng(2)
    path = "blocks.1.fc"
    delta = 0.1 * rng.normal(size=(128, 1)) @ rng.normal(size=(1, 32))
    base = tiny_model()(tokens)

    merged = tiny_model()
    log = weights.EditLog(model="tiny")
    record = weights.apply_delta(merged, path, delta, log=log, note="test edit")
    assert record.sha256_before != record.sha256_after
    assert 0 < record.relative_change < 1

    attached = tiny_model()
    a, b, _ = weights.delta_to_lora(delta)
    undo = weights.attach_lora(attached, path, a, b, scale=1.0)

    out_merged, out_attached = merged(tokens), attached(tokens)
    assert mx.abs(out_merged - base).max().item() > 1e-3, "the edit should change the output"
    assert mx.allclose(out_merged, out_attached, atol=1e-4).item()

    undo()
    assert mx.allclose(attached(tokens), base, atol=1e-6).item(), "undo should restore the original layer"


def test_edit_log_and_lora_files_round_trip(tmp_path):
    model = tiny_model()
    delta = np.zeros((128, 32), dtype=np.float32)
    delta[3, 5] = 0.5
    log = weights.EditLog(model="tiny")
    weights.apply_delta(model, "blocks.0.fc", delta, log=log)
    log.save(tmp_path / "edits.json")
    saved = json.loads((tmp_path / "edits.json").read_text())
    assert saved["edits"][0]["path"] == "blocks.0.fc"

    a, b, info = weights.delta_to_lora(delta)
    weights.save_lora(tmp_path / "adapter", {"blocks.0.fc": (a, b)}, info)
    loaded = weights.load_lora(tmp_path / "adapter")
    assert np.allclose(loaded["blocks.0.fc"][0], a) and np.allclose(loaded["blocks.0.fc"][1], b)


def test_apply_delta_rejects_a_wrong_shape():
    with pytest.raises(ValueError, match="does not match"):
        weights.apply_delta(tiny_model(), "blocks.0.fc", np.zeros((3, 3)))


def test_capture_reproduces_the_logits(tokens):
    model = tiny_model()
    logits, acts = hooks.capture(model, tokens)
    assert sorted(acts) == [0, 1, 2]
    assert acts[2].shape == (2, 12, CFG.n_embd)
    rebuilt = model.wte.as_linear(model.norm(acts[2]))
    assert mx.allclose(rebuilt, logits, atol=1e-5).item()
    assert mx.allclose(model(tokens), logits, atol=1e-6).item(), "the model is unchanged after capture"


def test_hook_and_steer_change_the_output_only_inside_the_block(tokens):
    model = tiny_model()
    stack = hooks.layer_list(model)
    originals = list(stack)
    base = model(tokens)

    with hooks.hook(model, 1, lambda out: out):
        assert mx.allclose(model(tokens), base, atol=1e-6).item(), "an identity hook changes nothing"
    with hooks.steer(model, 1, np.zeros(CFG.n_embd)):
        assert mx.allclose(model(tokens), base, atol=1e-6).item(), "a zero steering vector changes nothing"
    with hooks.steer(model, 1, np.ones(CFG.n_embd), scale=2.0):
        assert mx.abs(model(tokens) - base).max().item() > 1e-3
    with hooks.hook(model, 0, lambda out: mx.zeros_like(out)):
        assert mx.abs(model(tokens) - base).max().item() > 1e-3, "zeroing a layer changes the output"

    assert all(a is b for a, b in zip(hooks.layer_list(model), originals)), "layers are restored"
    assert mx.allclose(model(tokens), base, atol=1e-6).item()


def test_mean_difference_is_a_unit_vector():
    rng = np.random.default_rng(3)
    a, b = mx.array(rng.normal(size=(2, 5, 8)) + 1.0), mx.array(rng.normal(size=(2, 5, 8)))
    d = hooks.mean_difference(a, b)
    assert d.shape == (8,) and abs(np.linalg.norm(d) - 1.0) < 1e-6
