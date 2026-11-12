"""Speed experiment 14: LoRA fine-tuning with the frozen bottom layers computed once (lab/finetune/cached.py).

What it measures
  For one model and the public WikiSQL set, with LoRA tables in the last 16, 8 and 4 layers:
    normal   the lab's usual fine-tune: every step runs the whole model forwards
    cached   the bottom layers are run once per batch of examples before training; each
             step then runs only the top layers
  It reports training tokens per second for both, the one-off time to compute the cache,
  after how many training steps the cache has paid for itself, and the loss of both on
  the whole validation set before and after training.

Run:     uv run python -m bench.lora_cached --model LABEL=PATH_OR_REPO [--iters 30]
Changes: writes bench/results/lora_cached.json and throwaway adapters under runs/lora-bench/.
"""

from __future__ import annotations

import argparse
import tempfile

from bench.common import environment, free_gpu_memory, save
from lab.finetune.cached import finetune_cached
from lab.finetune.lora import ROOT, finetune

LAYERS = (16, 8, 4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", action="append", required=True, metavar="LABEL=PATH")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    for spec in args.model:
        label, _, path = spec.partition("=")
        entry = {"environment": environment(), "path": path, "iters": args.iters, "batch_size": args.batch_size, "layers": {}}
        for n in LAYERS:
            # val_batches=-1: both runs score the whole validation set, so their losses can be compared
            common = dict(iters=args.iters, batch_size=args.batch_size, num_layers=n, val_batches=-1)
            normal = finetune(path, adapter_path=ROOT / "runs/lora-bench" / f"{label}-normal-{n}", **common)
            free_gpu_memory()
            # The cache goes to a throwaway folder: building it is part of what is measured, and nothing is left behind.
            with tempfile.TemporaryDirectory() as cache_dir:
                cached = finetune_cached(path, adapter_path=ROOT / "runs/lora-bench" / f"{label}-cached-{n}",
                                         cache_dir=cache_dir, **common)
            free_gpu_memory()
            speedup = cached["tokens_per_second"] / normal["tokens_per_second"]
            # the cache pays back once the time saved per step adds up to the time it took to build
            step_tokens = cached["tokens_per_step"]
            saved_per_step = step_tokens / normal["tokens_per_second"] - step_tokens / cached["tokens_per_second"]
            entry["layers"][str(n)] = {
                "normal_tokens_per_s": normal["tokens_per_second"], "cached_tokens_per_s": cached["tokens_per_second"],
                "speedup": speedup, "cache_seconds": cached["cache_seconds"], "cache_gb": cached["cache_gb"],
                "pays_back_after_steps": cached["cache_seconds"] / saved_per_step if saved_per_step > 0 else None,
                "normal_peak_gb": normal["peak_memory_gb"], "cached_peak_gb": cached["peak_memory_gb"],
                "normal_val_loss": [normal["val_loss_before"], normal["val_loss_after"]],
                "cached_val_loss": [cached["val_loss_before"], cached["val_loss_after"]],
                "loss_check": [cached["loss_check_normal"], cached["loss_check_split"]],
            }
            e = entry["layers"][str(n)]
            # The cached run's GPU peak is measured while training; the cache itself sits in ordinary memory on top of it.
            print(f"{label}, last {n:2d} of {cached['layers_total']} layers: normal {e['normal_tokens_per_s']:6.0f} tok/s, cached "
                  f"{e['cached_tokens_per_s']:6.0f} tok/s ({speedup:.2f}x); cache built in {e['cache_seconds']:.0f} s, pays back "
                  f"after {e['pays_back_after_steps'] or float('nan'):.0f} steps; GPU peak {e['normal_peak_gb']:.1f} -> "
                  f"{e['cached_peak_gb']:.1f} GB plus {e['cache_gb']:.2f} GB of cache", flush=True)
            save("lora_cached", label, entry)


if __name__ == "__main__":
    main()
