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

What the lab adds to mlx-lm's trainer (each can be switched off for comparison)
  - its GPU paths for the frozen layers and for Qwen3.5's recurrent layers
    (--mlx-kernels switches them off; SPEED-RESULTS.md section 13)
  - batches of examples of similar length, so less of each batch is padding
    (--file-order keeps mlx-lm's own order)
  - a choice of when to trade speed for memory, from the model and the batch size
    (--grad-checkpoint on|off overrides it)

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
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np
from mlx_lm import load
from mlx_lm.lora import CONFIG_DEFAULTS, train_model
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.datasets import load_dataset

from lab.finetune import gated_delta
from lab.kernels import qmm

from lab.infer.batch import CACHE_LIMIT

ROOT = Path(__file__).resolve().parent.parent.parent
GRAD_CHECKPOINT = {"auto": None, "on": True, "off": False}  # None = decide from the model
# Without checkpointing, each trained recurrent layer keeps about this many arrays the size of its state
# for every chunk of tokens (measured: 4B and 9B models, 100- and 512-token examples, SPEED-RESULTS.md section 13).
STATES_KEPT_PER_CHUNK = 16
PEAK_BUDGET = 0.45  # share of the Mac's memory a fine-tune may fill before it trades speed for memory


class _Collect(TrainingCallback):
    def __init__(self) -> None:
        self.train: list[dict] = []
        self.val: list[dict] = []

    def on_train_loss_report(self, train_info: dict) -> None:
        self.train.append(train_info)

    def on_val_loss_report(self, val_info: dict) -> None:
        self.val.append(val_info)


@contextmanager
def keeps_layer_class(layer) -> Iterator[None]:
    """Undo, when the block ends, mlx-lm's gradient checkpointing if it was switched on inside it.

    mlx-lm switches it on by replacing __call__ on the layer's class. Left alone, that stays in
    force for every later model in the same Python process, and switching it on again stacks a
    second layer of recomputation on top of the first.
    """
    layer_class, original_call = type(layer), type(layer).__call__
    try:
        yield
    finally:
        layer_class.__call__ = original_call


class _Example(list):
    """One example's tokens. `offset` is where its answer starts (0 unless mlx-lm's prompt masking is on)."""

    def __init__(self, tokens: list[int], offset: int = 0) -> None:
        super().__init__(tokens)
        self.offset = offset


class ByLength:
    """An mlx-lm dataset, tokenized up front, so that mlx-lm's trainer really groups examples of similar length.

    mlx-lm (0.32.2) sorts the examples by len() of the raw record. That is 1 for every
    {"text": ...} or {"messages": ...} row, so its batches come in file order and each is padded
    to its longest example. Here an example is its list of tokens, so len() is its real length.
    """

    def __init__(self, data) -> None:
        self.examples = [_Example(*data.process(data[i])) for i in range(len(data))]

    def __getitem__(self, i: int) -> _Example:
        return self.examples[i]

    def __len__(self) -> int:
        return len(self.examples)

    def process(self, example: _Example) -> tuple[_Example, int]:
        return example, example.offset

    def longest_batch_tokens(self, batch_size: int, max_seq_length: int) -> int:
        """The most tokens one batch can hold."""
        return batch_size * min(max(len(example) for example in self.examples), max_seq_length)


def wants_checkpoint(model, lab_kernels: bool, num_layers: int, tokens_per_step: int) -> bool:
    """Whether to recompute instead of storing intermediate results (only ever needed for the Qwen3.5 family).

    mlx-lm's own training loop for the recurrent layers always needs it. With the lab's chunked
    update it costs speed on short examples and saves a lot of memory on long ones (last 16 layers
    trained, batches of 4; bench/results/lora.json):
      4B, ~100-token examples   10.9 GB and 338 tok/s without it,  7.9 GB and 281 tok/s with it
      4B, 256-token examples    14.8 GB and 405 tok/s without it,  8.5 GB and 341 tok/s with it
      4B, 512-token examples    22.6 GB and 280 tok/s without it,  9.7 GB and 345 tok/s with it
      9B, ~100-token examples   17.6 GB and 201 tok/s without it, 14.1 GB and 166 tok/s with it
    Those fit one estimate of the peak without it: 3 GB + 2.5 x the weights + what the trained
    recurrent layers keep, which is STATES_KEPT_PER_CHUNK arrays the size of the layer's state per
    chunk of tokens (14.5 GB estimated for the second case). Checkpointing is switched on when the
    estimate for the longest batch passes PEAK_BUDGET of the Mac's memory.
    """
    layers = model.layers
    if not any(getattr(layer, "is_linear", False) for layer in layers):
        return False
    if not lab_kernels:
        return True
    trained = layers if num_layers < 0 else layers[-num_layers:]
    try:
        state_bytes = sum(a.num_v_heads * a.head_v_dim * a.head_k_dim * 4
                          for a in (layer.linear_attn for layer in trained if getattr(layer, "is_linear", False)))
    except AttributeError:
        return True  # a variant whose sizes are not where Qwen3.5 keeps them: stay on the safe side
    kept_gb = STATES_KEPT_PER_CHUNK * state_bytes * tokens_per_step / gated_delta.CHUNK / 2**30
    weights_gb = sum(v.nbytes for _, v in tree_flatten(model.parameters())) / 2**30
    return 3 + 2.5 * weights_gb + kept_gb > PEAK_BUDGET * mx.device_info()["memory_size"] / 2**30


def finetune(model_path: str, *, data: str = CONFIG_DEFAULTS["data"], adapter_path: str | Path = ROOT / "runs/lora/example",
             iters: int = 200, batch_size: int = 4, num_layers: int = 16, max_seq_length: int = 512,
             learning_rate: float = 1e-5, rank: int = 8, fine_tune_type: str = "lora",
             grad_checkpoint: bool | None = None, val_batches: int = 10, seed: int = 0,
             lab_kernels: bool = True, sort_by_length: bool = True) -> dict:
    """Train and return a summary dict. Settings not listed here keep mlx-lm's defaults.

    grad_checkpoint=None picks it for you, for Qwen3.5-family models only: always on when
    lab_kernels is off (with mlx-lm's own training loop for their recurrent layers a 0.8B model
    needed 18.3 GB without it and 3.9 GB with it), and with lab_kernels on only when the model and
    the batches are big enough to need it (see wants_checkpoint).

    lab_kernels=True does two things. It routes the frozen quantized layers through
    lab/kernels/qmm.py, whose path for many rows (decode the weights, then MLX's dense multiply) is
    faster than MLX's quantized kernel and gives the same numbers. And for the Qwen3.5 family it
    replaces mlx-lm's token-by-token training loop for the recurrent layers with the same update
    worked out 64 tokens at a time (lab/finetune/gated_delta.py).

    sort_by_length=True puts examples of similar length in the same batch (see ByLength);
    False keeps mlx-lm's own order.
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
    if lab_kernels:
        qmm.install(model)
        gated_delta.install()
    else:
        gated_delta.uninstall()
    train_set, valid_set, _ = load_dataset(args, tokenizer)
    by_length = ByLength(train_set)
    if grad_checkpoint is None:
        grad_checkpoint = wants_checkpoint(model, lab_kernels, num_layers,
                                           by_length.longest_batch_tokens(batch_size, max_seq_length))
    args.grad_checkpoint = grad_checkpoint
    if sort_by_length:
        train_set, valid_set = by_length, ByLength(valid_set)
    mx.reset_peak_memory()
    seen = _Collect()
    with keeps_layer_class(model.layers[0]):
        train_model(args, model, train_set, valid_set, seen)

    # The first report includes one-off GPU compilation, so leave it out of the speed.
    speeds = [r["tokens_per_second"] for r in seen.train[1:]] or [r["tokens_per_second"] for r in seen.train]
    return {
        "model": model_path, "data": data, "fine_tune_type": fine_tune_type, "iters": iters,
        "batch_size": batch_size, "num_layers": num_layers, "max_seq_length": max_seq_length, "rank": rank,
        "grad_checkpoint": grad_checkpoint, "lab_kernels": lab_kernels, "sort_by_length": sort_by_length,
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
    ap.add_argument("--mlx-kernels", action="store_true", help="use MLX's own quantized kernels (for comparison)")
    ap.add_argument("--file-order", action="store_true", help="keep mlx-lm's batch order (for comparison)")
    args = ap.parse_args()
    summary = finetune(args.model, data=args.data, adapter_path=args.adapter_path, iters=args.iters,
                       batch_size=args.batch_size, num_layers=args.num_layers, max_seq_length=args.max_seq_length,
                       learning_rate=args.learning_rate, rank=args.rank, fine_tune_type=args.fine_tune_type,
                       grad_checkpoint=GRAD_CHECKPOINT[args.grad_checkpoint], lab_kernels=not args.mlx_kernels,
                       sort_by_length=not args.file_order)
    print("RESULT " + json.dumps(summary))


if __name__ == "__main__":
    main()
