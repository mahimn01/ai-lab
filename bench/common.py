"""Shared by the benchmark scripts: record the machine state and save results.

Every result file is JSON keyed by a short label, so a benchmark can be run again for
one model or one setting and only that entry is replaced.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import platform
import statistics
from datetime import datetime
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def environment() -> dict:
    """What the numbers were measured on: chip, OS, library versions, free memory."""
    import mlx.core as mx

    from lab.preflight import free_memory_pct, swap_used_gb

    info = mx.device_info()
    return {
        "when": datetime.now().isoformat(timespec="seconds"),
        "chip": info["device_name"],
        "memory_gb": round(info["memory_size"] / 2**30),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "mlx": md.version("mlx"),
        "mlx_lm": md.version("mlx-lm"),
        "free_memory_pct_at_start": free_memory_pct(),
        "swap_used_gb_at_start": round(swap_used_gb(), 2),
    }


def free_gpu_memory() -> None:
    """Drop what the last model left behind, so the next one starts from a clean GPU."""
    import gc

    import mlx.core as mx

    gc.collect()
    mx.clear_cache()


def mean_sd(values: list[float]) -> dict:
    return {"mean": statistics.mean(values), "sd": statistics.stdev(values) if len(values) > 1 else 0.0, "n": len(values)}


def save(name: str, label: str, entry: dict) -> Path:
    """Merge one labelled entry into bench/results/<name>.json and return the path.

    Paths under the home folder are written with ~, so a result file never carries a user name.
    """
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[label] = entry
    path.write_text((json.dumps(data, indent=2) + "\n").replace(str(Path.home()), "~"))
    return path
