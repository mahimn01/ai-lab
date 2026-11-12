"""LoRA baseline: how fast and how heavy is fine-tuning each model size on this Mac?

What it does
  Runs a short LoRA fine-tune (40 steps, batch 4, last 16 layers, up to 512 tokens per
  example, on the public WikiSQL set) for each model and records tokens/s and peak
  memory. These numbers say which model sizes are practical to fine-tune here.

Run:     uv run python -m bench.lora --model qwen3.5-0.8b-4bit=mlx-community/Qwen3.5-0.8B-4bit
         (repeat --model LABEL=PATH_OR_REPO)
Changes: writes bench/results/lora.json and throwaway adapters under runs/lora-bench/.
"""

from __future__ import annotations

import argparse

import mlx.core as mx

from bench.common import environment, free_gpu_memory, save
from lab.finetune.lora import GRAD_CHECKPOINT, ROOT, finetune


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", action="append", required=True, metavar="LABEL=PATH")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-checkpoint", choices=list(GRAD_CHECKPOINT), default="auto")
    args = ap.parse_args()
    for spec in args.model:
        label, _, path = spec.partition("=")
        label += f" | seq {args.max_seq_length}, batch {args.batch_size}, grad checkpoint {args.grad_checkpoint}"
        env = environment()
        try:
            entry = finetune(path, adapter_path=ROOT / "runs/lora-bench" / label.replace(" | ", "-").replace(", ", "-").replace(" ", ""),
                             iters=args.iters, val_batches=2, max_seq_length=args.max_seq_length,
                             batch_size=args.batch_size,
                             grad_checkpoint=GRAD_CHECKPOINT[args.grad_checkpoint])
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
