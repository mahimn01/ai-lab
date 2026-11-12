"""The from-scratch trainer: schedule and optimizer wiring (fast), plus one real 30-second run (slow)."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import pytest

from lab.train import train
from lab.train.model import GPT, GPTConfig, n_params

ROOT = Path(__file__).resolve().parent.parent


def test_lr_schedule_ramps_up_then_decays_to_ten_percent():
    assert train.lr_scale(0.0) == 0.0
    assert train.lr_scale(train.WARMUP_FRACTION) == pytest.approx(1.0)
    assert train.lr_scale(0.5) < 1.0
    assert train.lr_scale(1.0) == pytest.approx(0.1)
    assert train.lr_scale(2.0) == pytest.approx(0.1), "past the budget it stays at the floor"


def test_muon_gets_block_matrices_and_adamw_gets_the_rest():
    args = argparse.Namespace(optimizer="muon", lr=3e-3, muon_lr=0.02, weight_decay=0.0)
    optimizer, schedule = train.make_optimizers(args)
    muon_filter = optimizer.filters[0]
    assert muon_filter("blocks.0.fc.weight", mx.zeros((8, 4)))
    assert not muon_filter("wte.weight", mx.zeros((8, 4))), "the embedding table goes to AdamW"
    assert not muon_filter("blocks.0.norm1.weight", mx.zeros((4,))), "1-D weights go to AdamW"
    assert [peak for _, peak in schedule] == [0.02, 3e-3]


def test_model_shapes_and_weight_sharing():
    model = GPT(GPTConfig(vocab_size=100, n_layer=2, n_head=2, n_embd=16))
    logits = model(mx.zeros((3, 7), dtype=mx.int32))
    assert logits.shape == (3, 7, 100)
    # Embedding (100x16, shared with the output layer) + 2 blocks x (12 x 16^2 + 2 norms x 16) + final norm.
    assert n_params(model) == 100 * 16 + 2 * (12 * 16 * 16 + 2 * 16) + 16


@pytest.mark.slow
def test_thirty_second_run_learns_something():
    if not (train.DATA_DIR / "meta.json").exists():
        pytest.skip("run `uv run python -m lab.train.data` first")
    out = subprocess.run([sys.executable, "-m", "lab.train.train", "--name", "pytest-smoke", "--budget-s", "30"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    result = json.loads(out.split("RESULT ", 1)[1])
    untrained = math.log(json.loads((train.DATA_DIR / "meta.json").read_text())["vocab_size"])
    assert result["val_loss"] < untrained - 2.0, "30 seconds should cut the loss well below a random model's"
    assert result["steps"] > 20
