"""The experiment loop's keep/revert machinery, exercised with a scripted stand-in for the AI agent.

Slow: it trains three times for 15 seconds in a throwaway copy of the repo. The real
agent (`claude -p`) is not called here.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Round 1: breaks the rules by creating a file outside the two allowed ones.
# Round 2: makes a legal one-line change to the trainer.
FAKE_AGENT = '''
import pathlib, sys
counter = pathlib.Path(sys.argv[1])
n = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(n))
assert "Results so far" in sys.argv[-1], "the loop should pass the program and results as the prompt"
if n == 1:
    pathlib.Path("lab/not_allowed.py").write_text("x = 1\\n")
    print("CHANGE: wrote a file I should not have")
else:
    p = pathlib.Path("lab/train/train.py")
    s = p.read_text()
    assert 'default=0.04' in s
    p.write_text(s.replace('default=0.04', 'default=0.05'))
    print("CHANGE: raised the Muon learning rate to 0.05")
'''


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout


@pytest.mark.slow
def test_loop_rejects_bad_edits_and_keeps_or_reverts_good_ones(tmp_path):
    data = ROOT / "data" / "fineweb_edu"
    if not (data / "meta.json").exists():
        pytest.skip("run `uv run python -m lab.train.data` first")

    # A throwaway copy of the code with its own git history, pointing at the real token files.
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "lab", repo / "lab", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(ROOT / ".gitignore", repo / ".gitignore")
    (repo / "data" / "fineweb_edu").mkdir(parents=True)
    for f in data.iterdir():
        (repo / "data" / "fineweb_edu" / f.name).symlink_to(f)
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "-c", "user.name=test", "-c", "user.email=test@localhost", "commit", "-q", "-m", "start")

    agent = tmp_path / "fake_agent.py"
    agent.write_text(FAKE_AGENT)
    run = subprocess.run(
        [sys.executable, "-m", "lab.loop", "--rounds", "2", "--budget-s", "15", "--baseline-runs", "1",
         "--agent", f"{sys.executable} {agent} {tmp_path / 'counter'}"],
        cwd=repo, capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin:/opt/homebrew/bin", "HOME": str(Path.home()),
             "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@localhost",
             "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@localhost"})
    assert run.returncode == 0, run.stdout + run.stderr

    rows = [line.split("\t") for line in (repo / "results.tsv").read_text().splitlines()[1:]]
    assert [r[1] for r in rows] == ["baseline", "round 1", "round 2"]
    assert rows[1][3] == "rejected" and not (repo / "lab" / "not_allowed.py").exists()
    assert rows[2][3] in {"KEPT", "reverted"}

    trainer = (repo / "lab" / "train" / "train.py").read_text()
    kept = rows[2][3] == "KEPT"
    assert ("default=0.05" in trainer) == kept, "a kept change stays in the file; a reverted one is undone"
    assert ("Loop round 2" in git(repo, "log", "--oneline")) == kept
    assert git(repo, "status", "--porcelain").split() in ([], ["??", "results.tsv"]), "nothing is left half-changed"
    assert git(repo, "branch", "--show-current").startswith("loop/")
