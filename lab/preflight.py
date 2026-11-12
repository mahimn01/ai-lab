"""Preflight: is this Mac ready for a heavy model job right now?

What it does
  Prints free memory, swap in use, the apps holding the most memory, free disk, the
  thermal state and how much memory MLX says the GPU may use. Exits with code 1 if
  free memory or free disk is below the floor, so a script can refuse to start.

Why
  On a 32 GB Mac the model shares memory with every open app. A job started with
  little free memory swaps to disk and runs many times slower, or thrashes the Mac.

Run:     uv run python -m lab.preflight [--min-free-pct 30] [--min-disk-gb 20]
Changes: nothing. Read-only. It never closes an app; it tells you what to close.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass

GB = 2**30


@dataclass
class Reading:
    total_gb: float
    free_pct: int  # macOS "system-wide memory free percentage"
    swap_used_gb: float
    disk_free_gb: float
    thermal_ok: bool
    gpu_usable_gb: float
    top_apps: list[tuple[str, float]]  # (app name, GB held), largest first


def _out(*cmd: str) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def free_memory_pct() -> int:
    m = re.search(r"free percentage: (\d+)%", _out("memory_pressure"))
    if not m:
        raise RuntimeError("could not read the free percentage from memory_pressure")
    return int(m.group(1))


def swap_used_gb() -> float:
    m = re.search(r"used = ([\d.]+)M", _out("sysctl", "-n", "vm.swapusage"))
    if not m:
        raise RuntimeError("could not read vm.swapusage")
    return float(m.group(1)) / 1024


def disk_free_gb(path: str = "/") -> float:
    return shutil.disk_usage(path).free / GB


def _app_name(command: str) -> str:
    """'/Applications/Dia.app/Contents/.../Browser Helper' -> 'Dia'; '/usr/bin/python3' -> 'python3'."""
    m = re.search(r"/([^/]+)\.app/", command)
    return m.group(1) if m else command.rsplit("/", 1)[-1]


def top_apps(n: int = 8) -> list[tuple[str, float]]:
    """Memory held per app, with an app's helper processes added together."""
    held: dict[str, float] = defaultdict(float)
    for line in _out("ps", "-axo", "rss=,comm=").splitlines():
        rss, _, command = line.strip().partition(" ")
        if rss.isdigit():
            held[_app_name(command.strip())] += int(rss) * 1024 / GB
    return sorted(held.items(), key=lambda kv: -kv[1])[:n]


def thermal_ok() -> bool:
    """True unless macOS has recorded a thermal or performance warning."""
    text = _out("pmset", "-g", "therm")
    limits = [int(v) for v in re.findall(r"(?:CPU_Speed_Limit|CPU_Scheduler_Limit)\s*=\s*(\d+)", text)]
    return all(v == 100 for v in limits)


def read() -> Reading:
    import mlx.core as mx

    info = mx.device_info()
    return Reading(
        total_gb=info["memory_size"] / GB,
        free_pct=free_memory_pct(),
        swap_used_gb=swap_used_gb(),
        disk_free_gb=disk_free_gb(),
        thermal_ok=thermal_ok(),
        gpu_usable_gb=info["max_recommended_working_set_size"] / GB,
        top_apps=top_apps(),
    )


def report(r: Reading, min_free_pct: int, min_disk_gb: float) -> list[str]:
    """Print the reading; return the list of problems (empty = ready)."""
    free_gb = r.total_gb * r.free_pct / 100
    print(f"Memory   {r.free_pct}% free (about {free_gb:.1f} of {r.total_gb:.0f} GB); floor {min_free_pct}%")
    print(f"Swap     {r.swap_used_gb:.1f} GB in use (memory already pushed to disk)")
    print(f"Disk     {r.disk_free_gb:.0f} GB free on the internal disk; floor {min_disk_gb:.0f} GB")
    print(f"GPU      MLX may use up to {r.gpu_usable_gb:.1f} GB for models")
    print(f"Thermal  {'ok' if r.thermal_ok else 'macOS is slowing the CPU because of heat'}")
    print("Apps holding the most memory:")
    for name, gb in r.top_apps:
        print(f"  {gb:5.1f} GB  {name}")
    problems = []
    if r.free_pct < min_free_pct:
        problems.append(f"free memory {r.free_pct}% is under the {min_free_pct}% floor: close some of the apps above")
    if r.disk_free_gb < min_disk_gb:
        problems.append(f"free disk {r.disk_free_gb:.0f} GB is under the {min_disk_gb:.0f} GB floor")
    if not r.thermal_ok:
        problems.append("the Mac is thermally throttled: speed numbers taken now would be too low")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--min-free-pct", type=int, default=30)
    ap.add_argument("--min-disk-gb", type=float, default=20)
    args = ap.parse_args()
    problems = report(read(), args.min_free_pct, args.min_disk_gb)
    if problems:
        print("\nNOT READY:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("\nReady.")


if __name__ == "__main__":
    main()
