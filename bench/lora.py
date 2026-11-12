"""LoRA fine-tuning speed and memory for each model size, with mlx-lm as it comes or with the lab's additions.

What it does
  Runs a short LoRA fine-tune (40 steps, batch 4, last 16 layers, up to 512 tokens per
  example, on the public WikiSQL set) for each model and records tokens/s and peak
  memory. These numbers say which model sizes are practical to fine-tune here.

  By default the lab's additions are on (its GPU paths, and batches of similar-length
  examples). `--mlx-kernels --file-order` together give mlx-lm exactly as it comes.
  One model per run is the fair way to time: every run then starts from a fresh process.

  To check that the lab trains the same model as mlx-lm, run both with `--file-order
  --val-batches -1`: they then see the same batches in the same order, and both score
  the whole validation set, so their losses can be compared number for number.

  WikiSQL examples are short (about 100 tokens). --long-text trains on passages cut from
  the lab's held-out text instead, every one longer than the 512-token limit, to check
  the speed and memory on full-length examples.

Run:     uv run python -m bench.lora --model qwen3.5-0.8b-4bit=mlx-community/Qwen3.5-0.8B-4bit
         (repeat --model LABEL=PATH_OR_REPO; add --mlx-kernels or --long-text to compare)
Changes: writes bench/results/lora.json and throwaway adapters under runs/lora-bench/.
         --long-text also writes its passages to runs/long-text/ the first time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.common import environment, free_gpu_memory, save
from bench.quant_frontier import held_out_text
from lab.finetune.lora import GRAD_CHECKPOINT, ROOT, finetune
from mlx_lm.lora import CONFIG_DEFAULTS


def long_text_dataset() -> Path:
    """Write passages of 4,000 characters (well over 512 tokens each) from the lab's held-out text; returns the folder."""
    folder = ROOT / "runs" / "long-text"
    if not (folder / "train.jsonl").exists():
        text = held_out_text()
        passages = [text[i:i + 4000] for i in range(0, len(text) - 4000, 4000)]
        folder.mkdir(parents=True, exist_ok=True)
        for name, part in (("train", passages[:-16]), ("valid", passages[-16:])):
            (folder / f"{name}.jsonl").write_text("".join(json.dumps({"text": passage}) + "\n" for passage in part))
    return folder


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", action="append", required=True, metavar="LABEL=PATH")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-checkpoint", choices=list(GRAD_CHECKPOINT), default="auto")
    ap.add_argument("--mlx-kernels", action="store_true", help="use MLX's own quantized kernels (for comparison)")
    ap.add_argument("--file-order", action="store_true", help="keep mlx-lm's batch order (for comparison)")
    ap.add_argument("--val-batches", type=int, default=2, help="validation batches to score before and after (-1 = all)")
    ap.add_argument("--long-text", action="store_true", help="train on passages longer than --max-seq-length")
    args = ap.parse_args()
    data = str(long_text_dataset()) if args.long_text else CONFIG_DEFAULTS["data"]
    for spec in args.model:
        label, _, path = spec.partition("=")
        label += f" | seq {args.max_seq_length}, batch {args.batch_size}, grad checkpoint {args.grad_checkpoint}"
        label += "" if args.mlx_kernels else ", lab kernels"
        label += ", file order" if args.file_order else ""
        label += ", long text" if args.long_text else ""
        label += f", {args.iters} steps, whole validation set" if args.val_batches == -1 else ""
        env = environment()
        try:
            entry = finetune(path, adapter_path=ROOT / "runs/lora-bench" / label.replace(" | ", "-").replace(", ", "-").replace(" ", ""),
                             data=data, iters=args.iters, val_batches=args.val_batches, max_seq_length=args.max_seq_length,
                             batch_size=args.batch_size,
                             grad_checkpoint=GRAD_CHECKPOINT[args.grad_checkpoint], lab_kernels=not args.mlx_kernels,
                             sort_by_length=not args.file_order)
        except Exception as e:  # one model failing (out of memory) must not hide the others
            entry = {"error": f"{type(e).__name__}: {e}"[:300]}
        entry["environment"] = env
        save("lora", label, entry)
        shown = (f"{entry['tokens_per_second']:,.0f} tok/s, peak {entry['peak_memory_gb']:.1f} GB"
                 if "error" not in entry else entry["error"])
        print(f"{label}: {shown}", flush=True)
        free_gpu_memory()


if __name__ == "__main__":
    main()
