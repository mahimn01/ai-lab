"""Run a long job safely: keep the Mac awake, log everything, stop it if the Mac runs out of room.

What it does
  Starts the command you give it under `caffeinate` (so the Mac does not sleep), copies
  its output to runs/logs/<time>-<name>.log, and checks free memory and free disk once
  a minute. It stops the job if free disk falls under the floor, or if free memory
  stays under the floor for two checks in a row (a runaway job must not thrash the Mac).

Why
  Same watchdog as neural-harness/sd1/queue.sh, made reusable for any job.

Run:     uv run python -m lab.run --name bench -- uv run python bench/inference.py
Changes: writes one log file under runs/logs/. Stops only the job it started.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from lab.preflight import disk_free_gb, free_memory_pct

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "runs" / "logs"


def watch(proc: subprocess.Popen, min_free_pct: int, min_disk_gb: float, every_s: float = 60.0) -> str | None:
    """Wait for the job. Return None if it ended by itself, else the reason it was stopped."""
    low = 0
    while True:
        try:
            proc.wait(timeout=every_s)
            return None
        except subprocess.TimeoutExpired:
            pass
        mem, disk = free_memory_pct(), disk_free_gb()
        print(f"  watchdog {datetime.now():%H:%M} memory free {mem}%, disk free {disk:.0f} GB", flush=True)
        if disk < min_disk_gb:
            reason = f"free disk {disk:.0f} GB fell under {min_disk_gb:.0f} GB"
        else:
            low = low + 1 if mem < min_free_pct else 0
            reason = f"free memory stayed under {min_free_pct}% for two checks ({mem}%)" if low >= 2 else None
        if reason:
            # Stop the whole process group: the job is usually a shell or `uv run` whose child
            # does the real work, and stopping only the parent left that child running (2026-11-11).
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
            return reason


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", default="job", help="short label used in the log file name")
    ap.add_argument("--min-free-pct", type=int, default=12, help="stop the job under this free memory (default 12)")
    ap.add_argument("--min-disk-gb", type=float, default=8, help="stop the job under this free disk (default 8)")
    ap.add_argument("command", nargs=argparse.REMAINDER, help="the command to run, after --")
    args = ap.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        ap.error("give the command to run after --")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{args.name}.log"
    print(f"log: {log_path}")
    started = time.time()
    with log_path.open("w") as log:
        # -i keeps the Mac from idle-sleeping, -m keeps the disk awake, for as long as the job lives.
        proc = subprocess.Popen(["caffeinate", "-im", *command], stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
                                start_new_session=True)  # own process group, so the watchdog can stop all of it
        stopped = watch(proc, args.min_free_pct, args.min_disk_gb)
    minutes = (time.time() - started) / 60
    if stopped:
        print(f"STOPPED after {minutes:.1f} min: {stopped}")
        sys.exit(2)
    print(f"finished in {minutes:.1f} min with exit code {proc.returncode}")
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
