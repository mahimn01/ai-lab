"""Automated experiment loop: an AI agent proposes one training change per round; only real wins are kept.

What it does (the protocol of Karpathy's autoresearch, with a noise check added)
  1. Makes a new git branch, so the main branch is never touched.
  2. Measures the starting point: runs the current trainer --baseline-runs times. Their
     average is the score to beat; their spread is the noise.
  3. Each round: gives the agent lab/train/program.md plus the results so far. The agent
     edits lab/train/train.py or model.py. The loop then runs the fixed-time training.
       - If the score beats the best by more than the noise, it runs once more to
         confirm. If the average of the two still wins, the change is committed.
       - Otherwise the change is undone with git.
  4. Writes every round to results.tsv (score, kept or reverted, what was changed).

Cost
  Each round calls the agent once (Claude tokens) and trains for the time budget, plus
  one extra run when a change looks like a win. It only runs when you start it, and it
  stops after --rounds.

Run:     uv run python -m lab.loop --rounds 10 --budget-s 300 \\
             --agent "claude -p --permission-mode acceptEdits"
Changes: creates a branch loop/<time>, commits kept changes there, edits and restores
         lab/train/train.py and lab/train/model.py, appends to results.tsv, writes
         runs/train/loop-*/result.json. Never pushes.
"""

from __future__ import annotations

import argparse
import json
import shlex
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALLOWED = {"lab/train/train.py", "lab/train/model.py"}
RESULTS = ROOT / "results.tsv"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def train_once(name: str, budget_s: float) -> float | None:
    """Run the trainer; return val_bpb, or None if the run crashed or diverged."""
    run = subprocess.run([sys.executable, "-m", "lab.train.train", "--name", name, "--budget-s", str(budget_s)],
                         cwd=ROOT, capture_output=True, text=True)
    if run.returncode != 0 or "RESULT " not in run.stdout:
        print((run.stderr or run.stdout).strip()[-400:])
        return None
    return json.loads(run.stdout.split("RESULT ", 1)[1])["val_bpb"]


def log(round_label: str, score: float | None, verdict: str, change: str) -> None:
    if not RESULTS.exists():
        RESULTS.write_text("when\tround\tval_bpb\tverdict\tchange\n")
    shown = f"{score:.4f}" if score is not None else "failed"
    with RESULTS.open("a") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M}\t{round_label}\t{shown}\t{verdict}\t{change}\n")
    print(f"[{round_label}] {shown} {verdict}: {change}", flush=True)


def changed_files() -> set[str]:
    """Paths git reports as modified, added or untracked."""
    # Not through git(): its strip() would eat the leading space of the first " M path" line.
    out = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return {line[3:] for line in out.splitlines() if line}


def undo_changes() -> None:
    """Put back every tracked file the agent touched and remove files it created. results.tsv is left alone."""
    git("checkout", "--", ".", f":(exclude){RESULTS.name}")
    git("clean", "-fdq", "-e", RESULTS.name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--budget-s", type=float, default=300.0)
    ap.add_argument("--agent", required=True, help='command that receives the prompt as its last argument')
    ap.add_argument("--baseline-runs", type=int, default=3)
    args = ap.parse_args()

    if changed_files() - {RESULTS.name}:
        sys.exit("Commit or stash your changes first: the loop needs a clean git tree to undo failed ideas safely.")
    branch = f"loop/{datetime.now():%Y%m%d-%H%M%S}"
    git("checkout", "-b", branch)
    print(f"working on branch {branch}")

    # Starting point and noise.
    scores = [train_once(f"loop-baseline-{i}", args.budget_s) for i in range(args.baseline_runs)]
    if any(s is None for s in scores):
        sys.exit("The baseline trainer failed; fix that before starting the loop.")
    best = statistics.mean(scores)
    noise = max(scores) - min(scores) if len(scores) > 1 else 0.0
    log("baseline", best, "start", f"mean of {len(scores)} runs, spread {noise:.4f}")

    program = (ROOT / "lab/train/program.md").read_text()
    for r in range(1, args.rounds + 1):
        history = RESULTS.read_text()
        prompt = (f"{program}\n\n## Results so far (tab-separated)\n\n{history}\n"
                  f"Best val_bpb so far: {best:.4f}. Run-to-run noise: {noise:.4f}. "
                  f"Time budget per run: {args.budget_s:.0f} s.")
        reply = subprocess.run([*shlex.split(args.agent), prompt], cwd=ROOT, capture_output=True, text=True)
        change = next((line[7:].strip() for line in reversed(reply.stdout.splitlines()) if line.startswith("CHANGE:")),
                      "(agent gave no CHANGE line)")
        touched = changed_files() - {RESULTS.name}
        if reply.returncode != 0:
            undo_changes()
            log(f"round {r}", None, "agent failed", (reply.stderr or reply.stdout).strip()[-200:].replace("\n", " "))
            continue
        if not touched:
            log(f"round {r}", None, "no change", change)
            continue
        if not touched <= ALLOWED:
            undo_changes()
            log(f"round {r}", None, "rejected", f"edited files outside the allowed two: {sorted(touched - ALLOWED)}")
            continue

        score = train_once(f"loop-r{r}", args.budget_s)
        if score is not None and score < best - noise:
            # Looks like a win: confirm with a second run before believing it.
            again = train_once(f"loop-r{r}-confirm", args.budget_s)
            confirmed = statistics.mean([score, again]) if again is not None else None
            if confirmed is not None and confirmed < best - noise:
                git("add", *sorted(touched))
                git("commit", "-q", "-m", f"Loop round {r}: {change[:60]}")
                best = confirmed
                log(f"round {r}", confirmed, "KEPT", change)
                continue
            score = confirmed if confirmed is not None else score
        undo_changes()
        log(f"round {r}", score, "reverted", change)

    print(f"done. Best val_bpb {best:.4f} on branch {branch}. Results: {RESULTS}")


if __name__ == "__main__":
    main()
