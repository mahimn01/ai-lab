"""Fine-tuning: the batching helpers (fast), and whole runs on real small models (slow: each loads a model and trains).

The training text of the slow tests is the lab's own README and results write-up, cut into short passages.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    """A folder in mlx-lm's format (train.jsonl, valid.jsonl) holding 40 short passages of real text."""
    words = ((ROOT / "README.md").read_text() + (ROOT / "SPEED-RESULTS.md").read_text()).split()
    passages = [" ".join(words[i:i + 40]) for i in range(0, 40 * 40, 40)]
    assert len(words) >= 40 * 40 and all(passages)
    for name, rows in (("train", passages[:32]), ("valid", passages[32:])):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps({"text": p}) + "\n" for p in rows))
    return tmp_path


class FakeDataset:
    """Stands in for an mlx-lm dataset: records are dicts, process() turns one into (tokens, answer offset)."""

    def __init__(self, lengths: list[int], offset: int = 0) -> None:
        self.records = [{"text": n} for n in lengths]
        self.offset = offset

    def __getitem__(self, i: int) -> dict:
        return self.records[i]

    def __len__(self) -> int:
        return len(self.records)

    def process(self, record: dict) -> tuple[list[int], int]:
        return list(range(1, record["text"] + 1)), self.offset


def test_by_length_makes_mlx_lm_group_examples_of_similar_length():
    from mlx_lm.tuner.datasets import CacheDataset
    from mlx_lm.tuner.trainer import iterate_batches

    from lab.finetune.lora import ByLength

    lengths = [40, 200, 41, 199, 42, 198, 43, 197]

    def widths(dataset) -> list[int]:
        return sorted(batch.shape[1] for batch, _ in iterate_batches(CacheDataset(dataset), batch_size=4, max_seq_length=512))

    assert widths(FakeDataset(lengths)) == [225, 225], "mlx-lm alone keeps file order: every batch holds a long example"
    by_length = ByLength(FakeDataset(lengths))
    assert widths(by_length) == [65, 225], "short examples now share a batch"
    assert by_length.longest_batch_tokens(4, 512) == 800 and by_length.longest_batch_tokens(4, 128) == 512


def test_make_batches_sorts_pads_like_mlx_lm_and_refuses_prompt_masking():
    from lab.finetune.cached import make_batches

    lengths = [40, 200, 41, 199, 42, 198, 43, 197]
    batches = make_batches(FakeDataset(lengths), batch_size=4, max_seq_length=512)
    assert [tokens.shape for tokens, _ in batches] == [(4, 65), (4, 225)]
    assert batches[0][1].tolist() == [40, 41, 42, 43] and batches[0][0][0, 39] == 40 and batches[0][0][0, 40] == 0
    in_file_order = make_batches(FakeDataset(lengths), batch_size=4, max_seq_length=512, sort=False)
    assert [tokens.shape for tokens, _ in in_file_order] == [(4, 225), (4, 225)]
    with pytest.raises(ValueError, match="mask_prompt"):
        make_batches(FakeDataset(lengths, offset=5), batch_size=4, max_seq_length=512)


def test_keeps_layer_class_undoes_mlx_lm_checkpointing():
    import mlx.nn as nn
    from mlx_lm.tuner.trainer import grad_checkpoint

    from lab.finetune.lora import keeps_layer_class

    class Layer(nn.Module):
        def __call__(self, x):
            return x

    layer, original = Layer(), Layer.__call__
    with keeps_layer_class(layer):
        grad_checkpoint(layer)
        assert Layer.__call__ is not original
    assert Layer.__call__ is original


@pytest.mark.slow
def test_cached_prefix_lora_matches_the_normal_forward_pass_and_saves_a_loadable_adapter(tiny_dataset: Path, tmp_path: Path):
    from mlx_lm import load

    from lab.finetune.cached import finetune_cached

    out, cache_dir = tmp_path / "adapter", tmp_path / "cache"
    result = finetune_cached("mlx-community/Qwen3-0.6B-4bit", data=str(tiny_dataset), adapter_path=out, iters=12,
                             num_layers=4, learning_rate=1e-4, val_batches=2, cache_dir=cache_dir)
    assert result["loss_check_split"] == pytest.approx(result["loss_check_normal"], abs=0.02)
    assert result["val_loss_after"] < result["val_loss_before"], "twelve steps should already lower the held-out loss"
    weights = mx.load(str(out / "adapters.safetensors"))
    top = result["layers_total"] - 1
    assert any(f"layers.{top}." in name for name in weights), "adapters are saved under the real layer numbers"
    assert not any("layers.0." in name for name in weights), "the first layer was frozen"
    model, _ = load("mlx-community/Qwen3-0.6B-4bit", adapter_path=str(out))  # mlx-lm can load what was saved
    assert len(model.layers) == result["layers_total"]

    # A second run on the same data and split point loads the cached layer results from disk.
    again = finetune_cached("mlx-community/Qwen3-0.6B-4bit", data=str(tiny_dataset), adapter_path=tmp_path / "second",
                            iters=2, num_layers=4, learning_rate=3e-4, val_batches=2, cache_dir=cache_dir)
    assert again["cache_from_disk"] and not result["cache_from_disk"]
    assert again["val_loss_before"] == pytest.approx(result["val_loss_before"], abs=1e-4)


@pytest.mark.slow
def test_qwen35_lora_trains_with_the_chunked_recurrent_update(tiny_dataset: Path, tmp_path: Path):
    from mlx_lm.models import qwen3_5

    from lab.finetune import gated_delta
    from lab.finetune.lora import finetune

    result = finetune("mlx-community/Qwen3.5-0.8B-4bit", data=str(tiny_dataset), adapter_path=tmp_path / "adapter",
                      iters=12, num_layers=4, learning_rate=1e-4, val_batches=2)
    assert qwen3_5.gated_delta_update is gated_delta.gated_delta_update, "the faster training path was switched on"
    assert result["grad_checkpoint"] is False, "a small model does not need gradient checkpointing any more"
    assert result["val_loss_after"] < result["val_loss_before"]
    assert result["peak_memory_gb"] < 8


@pytest.mark.slow
def test_checkpointing_is_switched_off_again_after_a_run(tiny_dataset: Path, tmp_path: Path):
    from mlx_lm.models import qwen3_5

    from lab.finetune.lora import finetune

    before = qwen3_5.DecoderLayer.__call__
    result = finetune("mlx-community/Qwen3.5-0.8B-4bit", data=str(tiny_dataset), adapter_path=tmp_path / "adapter",
                      iters=2, num_layers=4, val_batches=1, grad_checkpoint=True)
    assert result["grad_checkpoint"] is True
    assert qwen3_5.DecoderLayer.__call__ is before, "mlx-lm's class-wide switch was undone"
