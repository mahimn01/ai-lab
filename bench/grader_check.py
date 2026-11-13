"""Check of the math-verify grader (bench/grading.py) against published verdicts, and against the study's grader.

Why
  Phase 1 of the neural-harness study is graded with math-verify, as the blog was. Before a
  number graded that way is trusted, the grader here must give the verdicts the published run
  gave, and the size of the change from the study's Phase 0 grader should be known.

What it does
  1. Grades the 40 generations published with the edited model (the first sample of the first 20
     problems, edited and base, in the study's models/edited/eval/) and compares each verdict with
     the published one, which was made with math-verify 0.9.0 by the blog's own script.
  2. Grades every full-length row of the saved timing runs (runs/phase1/, not in the repo) with
     both graders and counts where they disagree.

Run:     uv run python -m bench.grader_check     (CPU only, about a minute)
Changes: writes bench/results/grader_check.json. Reads the study's folder and runs/phase1/.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

from mlx_lm.utils import load_tokenizer

from bench import grading
from bench.common import save
from bench.phase1 import ROOT, plain_text
from lab.gen.ledger import Ledger


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--harness", type=Path, default=Path.home() / "Documents/Dev/neural-harness")
    args = ap.parse_args()
    harness = args.harness.expanduser()
    sys.path.insert(0, str(harness))
    from harness.grader import grade as study_grade

    published = {}
    for arm in ("edited", "base"):
        samples = [json.loads(line) for line in (harness / f"models/edited/eval/samples_{arm}_math500.jsonl").read_text().splitlines()]
        scores = json.loads((harness / f"models/edited/eval/scores_{arm}_math500.json").read_text())
        theirs = {p["id"]: bool(p["correct"][0]) for p in scores["per_problem"]}
        mine = {s["id"]: grading.grade(s["text"], s["answer"])[1] for s in samples}
        published[arm] = {"samples": len(samples), "right_in_the_published_file": sum(theirs[i] for i in mine),
                          "same_verdict": sum(mine[i] == theirs[i] for i in mine),
                          "different": sorted(i for i in mine if mine[i] != theirs[i])}
        print(f"published {arm} samples: {published[arm]['same_verdict']}/{len(samples)} verdicts the same "
              f"({published[arm]['right_in_the_published_file']} right)", flush=True)

    tok = load_tokenizer(harness / "models/base")
    runs = {}
    for folder in sorted((ROOT / "runs/phase1").iterdir()):
        ledger = Ledger(folder)
        rows = [r for r in ledger.records() if r["meta"].get("kind") == "trunk"]
        if not rows:
            continue
        texts = [(plain_text(tok, ledger.tokens(r)), r["meta"]["gold"]) for r in rows]
        both = [(study_grade(text, gold)[1], grading.grade(text, gold)[1]) for text, gold in texts]
        count = {"rows": len(rows), "right_by_the_study_grader": sum(old for old, _ in both), "right_by_math_verify": sum(new for _, new in both),
                 "right_only_by_the_study_grader": sum(old and not new for old, new in both),
                 "right_only_by_math_verify": sum(new and not old for old, new in both)}
        runs[folder.name] = count
        print(f"{folder.name}: {count['rows']} rows, right {count['right_by_the_study_grader']} by the study's grader and "
              f"{count['right_by_math_verify']} by math-verify ({count['right_only_by_the_study_grader']} only by the study's, "
              f"{count['right_only_by_math_verify']} only by math-verify)", flush=True)

    save("grader_check", "math_verify", {
        "versions": {name: version(name) for name in ("math-verify", "latex2sympy2_extended", "antlr4-python3-runtime", "sympy")},
        "published_samples": published, "saved_runs_full_rows": runs})


if __name__ == "__main__":
    main()
