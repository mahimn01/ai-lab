"""Speed experiment 5: which training choices give the best model in a fixed time on this Mac?

What it does
  Runs lab/train/train.py several times with the same time budget, changing one thing
  at a time, and records the validation score (val_bpb, lower is better) of each:
    noise floor   the default setup three times with different random seeds. Two runs
                  that differ by less than this spread are not really different.
    optimizer     Muon+AdamW (default) against AdamW alone, each at three learning rates
    number format 32-bit weights against 16-bit (bfloat16, float16)
    compilation   the compiled training step against the uncompiled one
    confirm       the two changes that won on their own, combined, on two seeds

Run:     uv run python -m bench.train_sweep [--budget-s 180] [--only noise,optimizer,dtype,compile,confirm]
Changes: writes bench/results/train_sweep.json and runs/train/sweep-*/result.json.
         Takes about (number of runs) x (budget + 20 s).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from bench.common import environment, save

ROOT = Path(__file__).resolve().parent.parent
GROUPS: dict[str, dict[str, list[str]]] = {
    "noise": {
        # The Muon rate is spelled out so these rows keep their meaning when the trainer's default moves.
        "muon seed 0 (default)": ["--muon-lr", "0.02", "--seed", "0"],
        "muon seed 1": ["--muon-lr", "0.02", "--seed", "1"],
        "muon seed 2": ["--muon-lr", "0.02", "--seed", "2"],
    },
    "optimizer": {
        "adamw lr 1e-3": ["--optimizer", "adamw", "--lr", "1e-3"],
        "adamw lr 3e-3": ["--optimizer", "adamw", "--lr", "3e-3"],
        "adamw lr 6e-3": ["--optimizer", "adamw", "--lr", "6e-3"],
        "muon lr 0.01": ["--muon-lr", "0.01"],
        "muon lr 0.04": ["--muon-lr", "0.04"],
    },
    "dtype": {
        "muon bfloat16": ["--muon-lr", "0.02", "--dtype", "bfloat16"],
        "muon float16": ["--muon-lr", "0.02", "--dtype", "float16"],
    },
    "compile": {
        "muon not compiled": ["--muon-lr", "0.02", "--no-compile"],
    },
    # Follow-up: the two single-change wins (higher Muon rate, bfloat16) together, on two
    # seeds, plus one step further on the rate to see where it stops helping.
    "confirm": {
        "muon lr 0.04 bfloat16 seed 0": ["--muon-lr", "0.04", "--dtype", "bfloat16", "--seed", "0"],
        "muon lr 0.04 bfloat16 seed 1": ["--muon-lr", "0.04", "--dtype", "bfloat16", "--seed", "1"],
        "muon lr 0.08": ["--muon-lr", "0.08"],
    },
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--budget-s", type=float, default=180.0)
    ap.add_argument("--only", default=",".join(GROUPS))
    args = ap.parse_args()
    for group in args.only.split(","):
        for label, extra in GROUPS[group].items():
            name = "sweep-" + label.replace(" ", "-").replace("(", "").replace(")", "")
            run = subprocess.run([sys.executable, "-m", "lab.train.train", "--name", name,
                                  "--budget-s", str(args.budget_s), *extra], cwd=ROOT, capture_output=True, text=True)
            if run.returncode == 0 and "RESULT " in run.stdout:
                entry = json.loads(run.stdout.split("RESULT ", 1)[1])
                shown = f"val_bpb {entry['val_bpb']:.4f}  {entry['train_tokens_per_s']:,.0f} tok/s  {entry['steps']} steps"
            else:  # a diverged or crashed run is a result too: record why
                entry = {"error": (run.stderr or run.stdout).strip().splitlines()[-1][:300]}
                shown = "FAILED " + entry["error"]
            entry |= {"group": group, "budget_s": args.budget_s, "environment": environment()}
            save("train_sweep", label, entry)
            print(f"{label:26s} {shown}", flush=True)


if __name__ == "__main__":
    main()
