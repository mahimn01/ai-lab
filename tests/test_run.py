"""The job runner's watchdog: a job over the memory floor is stopped, children included."""

from __future__ import annotations

import subprocess
import sys
import time

from lab import run


def test_watchdog_stops_the_whole_process_tree(monkeypatch):
    # A shell that starts a long-running child, like `uv run python ...` does.
    proc = subprocess.Popen(["sh", "-c", "sleep 300 & echo $!; wait"], stdout=subprocess.PIPE, text=True,
                            start_new_session=True)
    child_pid = int(proc.stdout.readline())
    monkeypatch.setattr(run, "free_memory_pct", lambda: 1)  # pretend memory is almost gone
    monkeypatch.setattr(run, "disk_free_gb", lambda: 500.0)

    reason = run.watch(proc, min_free_pct=12, min_disk_gb=8, every_s=0.05)

    assert reason is not None and "free memory" in reason
    time.sleep(0.2)
    gone = subprocess.run(["ps", "-p", str(child_pid)], capture_output=True).returncode != 0
    assert gone, "the child process must be stopped too, not left running"


def test_watchdog_lets_a_healthy_job_finish(monkeypatch):
    proc = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    monkeypatch.setattr(run, "free_memory_pct", lambda: 80)
    monkeypatch.setattr(run, "disk_free_gb", lambda: 500.0)
    assert run.watch(proc, min_free_pct=12, min_disk_gb=8, every_s=0.05) is None
    assert proc.returncode == 0


def test_watchdog_stops_a_job_that_runs_past_its_time_limit(monkeypatch):
    proc = subprocess.Popen(["sleep", "300"], start_new_session=True)
    monkeypatch.setattr(run, "free_memory_pct", lambda: 80)
    monkeypatch.setattr(run, "disk_free_gb", lambda: 500.0)
    reason = run.watch(proc, min_free_pct=12, min_disk_gb=8, every_s=0.05, max_hours=0.1 / 3600)
    assert reason is not None and "limit" in reason and proc.poll() is not None
