"""Fine-tune an existing model cheaply with LoRA, and report how fast and how heavy it was.

What it does
  Freezes the model and trains only small add-on tables (LoRA) in its last layers, using
  mlx-lm's own trainer. If the model is quantized (4-bit / 8-bit) this is "QLoRA": the
  frozen part stays small in memory. At the end it prints the tokens/s, the peak memory
  and the training and validation losses, and saves the add-on to --adapter-path.

Plain words
  LoRA        instead of changing a 1,000 x 1,000 table (a million numbers), learn two
              thin tables whose product is added to it (a few thousand numbers)
  rank        how thin those tables are; 8 is the usual start
  validation  loss on examples the training never saw: the number that should go down

Data
  --data is a Hugging Face dataset name or a folder with train.jsonl / valid.jsonl
  (chat, completions or text format; see mlx-lm's LORA.md). The default is the small
  public text-to-SQL set that mlx-lm uses in its own guide.

Run:     uv run python -m lab.finetune.lora --model mlx-community/Qwen3.5-0.8B-4bit --iters 200
Use it:  uv run mlx_lm.generate --model mlx-community/Qwen3.5-0.8B-4bit \\
             --adapter-path runs/lora/example --prompt "..."
Changes: writes the adapter files under --adapter-path (default runs/lora/example).
         Downloads the model and dataset to the Hugging Face cache if needed.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.lora import CONFIG_DEFAULTS, train_model
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.datasets import load_dataset

from lab.infer.batch import CACHE_LIMIT

ROOT = Path(__file__).resolve().parent.parent.parent
GRAD_CHECKPOINT = {"auto": None, "on": True, "off": False}  # None = decide from the model


class _Collect(TrainingCallback):
    def __init__(self) -> None:
        self.train: list[dict] = []
        self.val: list[dict] = []

    def on_train_loss_report(self, train_info: dict) -> None:
        self.train.append(train_info)

    def on_val_loss_report(self, val_info: dict) -> None:
        self.val.append(val_info)


def finetune(model_path: str, *, data: str = CONFIG_DEFAULTS["data"], adapter_path: str | Path = ROOT / "runs/lora/example",
             iters: int = 200, batch_size: int = 4, num_layers: int = 16, max_seq_length: int = 512,
             learning_rate: float = 1e-5, rank: int = 8, fine_tune_type: str = "lora",
             grad_checkpoint: bool | None = None, val_batches: int = 10, seed: int = 0) -> dict:
    """Train and return a summary dict. Settings not listed here keep mlx-lm's defaults.

    grad_checkpoint=None picks it for you: on for models with linear-attention layers (the Qwen3.5
    family), off otherwise. Measured 2026-11-11 on Qwen3.5-0.8B-4bit: 18.3 GB peak without it,
    3.9 GB with it, at the same speed. Attention-only models (Qwen3) do not need it.
    """
    config = copy.deepcopy(CONFIG_DEFAULTS) | {
        "model": model_path, "train": True, "data": data, "adapter_path": str(adapter_path), "iters": iters,
        "batch_size": batch_size, "num_layers": num_layers, "max_seq_length": max_seq_length,
        "learning_rate": learning_rate, "fine_tune_type": fine_tune_type,
        "val_batches": val_batches, "seed": seed, "steps_per_report": 10,
        "steps_per_eval": iters, "save_every": iters,  # score and save once, at the end
    }
    config["lora_parameters"]["rank"] = rank
    args = argparse.Namespace(**config)

    np.random.seed(seed)
    # Cap MLX's reuse cache: batches of different lengths otherwise leave many freed buffers held.
    mx.set_cache_limit(CACHE_LIMIT)
    model, tokenizer = load(model_path)
    if grad_checkpoint is None:
        grad_checkpoint = any(getattr(layer, "is_linear", False) for layer in model.layers)
    args.grad_checkpoint = grad_checkpoint
    train_set, valid_set, _ = load_dataset(args, tokenizer)
    mx.reset_peak_memory()
    seen = _Collect()
    train_model(args, model, train_set, valid_set, seen)

    # The first report includes one-off GPU compilation, so leave it out of the speed.
    speeds = [r["tokens_per_second"] for r in seen.train[1:]] or [r["tokens_per_second"] for r in seen.train]
    return {
        "model": model_path, "data": data, "fine_tune_type": fine_tune_type, "iters": iters,
        "batch_size": batch_size, "num_layers": num_layers, "max_seq_length": max_seq_length, "rank": rank,
        "grad_checkpoint": grad_checkpoint,
        "tokens_per_second": statistics.mean(speeds),
        "peak_memory_gb": mx.get_peak_memory() / 2**30,
        "train_loss_first": seen.train[0]["train_loss"], "train_loss_last": seen.train[-1]["train_loss"],
        "val_loss_before": seen.val[0]["val_loss"] if seen.val else None,
        "val_loss_after": seen.val[-1]["val_loss"] if len(seen.val) > 1 else None,
        "adapter_path": str(adapter_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=CONFIG_DEFAULTS["data"])
    ap.add_argument("--adapter-path", default=str(ROOT / "runs/lora/example"))
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-layers", type=int, default=16, help="how many of the last layers get LoRA tables (-1 = all)")
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--fine-tune-type", choices=["lora", "dora", "full"], default="lora")
    ap.add_argument("--grad-checkpoint", choices=list(GRAD_CHECKPOINT), default="auto",
                    help="recompute instead of storing intermediate results: much less memory on Qwen3.5-family models")
    args = ap.parse_args()
    summary = finetune(args.model, data=args.data, adapter_path=args.adapter_path, iters=args.iters,
                       batch_size=args.batch_size, num_layers=args.num_layers, max_seq_length=args.max_seq_length,
                       learning_rate=args.learning_rate, rank=args.rank, fine_tune_type=args.fine_tune_type,
                       grad_checkpoint=GRAD_CHECKPOINT[args.grad_checkpoint])
    print("RESULT " + json.dumps(summary))


if __name__ == "__main__":
    main()
