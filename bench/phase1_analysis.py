"""The planned comparisons of a Phase 1 run: accuracy against tokens for every condition, and paired tests between conditions.

Why
  bench/phase1_study.zsh leaves three entries in bench/results/phase1.json (thinking off, edited,
  base), each with a verdict for every row under every cap and every forced exit. The study's
  question is a comparison between arms, which no single entry answers. This script makes the
  comparisons, and it is committed with the plan before the run, so the analysis is fixed before
  any result exists.

What it does
  1. Finds the problems to analyse: in the order they were drawn, up to the first one that is not
     complete in all three arms (every sample finished, every forced exit made). In a finished
     run that is all of them. A run that was cut short is analysed on that front part only,
     because the rows that happen to have finished are not a fair sample (short ones finish first).
  2. For every condition: the share of rows right, a 95% interval, and the mean tokens used.
     With one sample per problem the interval is Wilson's. With more it is the mean of the
     per-problem shares plus and minus 1.96 standard errors, so that the samples of one problem
     are not counted as independent.
  3. For each --compare pair: the difference of the per-problem shares (first minus second), its
     95% interval, and an exact two-sided sign-flip test: if the two conditions are equally good,
     each problem's difference is as likely to have had the other sign. With one sample per
     problem this is the exact McNemar test. The first pair is the primary comparison; the others
     also get their p-value corrected for their number (Holm).

Names of conditions
  edited@32768        the edited model's rows cut at 32,768 tokens
  edited+exit@8192    the edited model with a forced exit at 8,192 tokens
  base@24576, base+exit@16384    the same for the base model
  nothink@8192        the base model with thinking off

Run:     uv run python -m bench.phase1_analysis --name p1 --compare edited@32768,base+exit@24576 --compare ...
Changes: writes one entry in bench/results/phase1_analysis.json. Reads bench/results/phase1.json.
         CPU only, a few seconds. The mean tokens are taken over every finished row of an arm,
         also when the comparison uses only the front part of a run that was cut short.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics

import numpy as np

from bench.common import RESULTS, save

ARMS = ("nothink", "edited", "base")


def conditions(arm: str, entry: dict) -> dict[str, tuple[dict[str, str], float]]:
    """{condition name: ({problem: one verdict character per sample}, mean tokens)} for one arm's entry."""
    c = next(iter(entry["conditions"].values()))
    names = [name.rsplit("#", 1)[0] for name in c["correct_by_problem"]]
    found = {f"{arm}@{cap}": cut for cap, cut in c["at_smaller_caps"].items()}
    found |= {f"{arm}+exit@{b}": forced for b, forced in c["forced_exit"].items()}
    out = {}
    for condition, numbers in found.items():
        marks: dict[str, str] = {}
        for problem, mark in zip(names, numbers["by_row"]):
            marks[problem] = marks.get(problem, "") + mark
        out[condition] = (marks, numbers["mean_tokens"])
    return out


def front_part(order: list[str], marks: dict[str, dict[str, str]], samples: int) -> list[str]:
    """The problems, in the order drawn, before the first one that some condition has not finished."""
    done = lambda problem: all(len(m.get(problem, "")) == samples and "-" not in m[problem] for m in marks.values())
    return order[: next((i for i, problem in enumerate(order) if not done(problem)), len(order))]


def wilson(right: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = right / n
    centre, half = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / (1 + z * z / n), (centre + half) / (1 + z * z / n)


def band(values: list[float]) -> tuple[float, list[float]]:
    """The mean of one value per problem, and the mean plus and minus 1.96 standard errors."""
    mean, half = statistics.mean(values), 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return mean, [mean - half, mean + half]


def share_right(marks: dict[str, str], problems: list[str], samples: int) -> dict:
    """Accuracy of one condition over the analysed problems, with its 95% interval."""
    right = [marks[p].count("1") for p in problems]
    mean, around = band([r / samples for r in right])
    return {"accuracy": mean, "accuracy_95": list(wilson(sum(right), len(right))) if samples == 1 else around}


def sign_flip_p(differences: list[int]) -> float:
    """Exact two-sided p-value: how often the sum of whole-number differences lands at least as far from zero as it did,
    when every difference keeps its size and gets its sign by a fair coin."""
    sizes = [abs(d) for d in differences if d]
    total = sum(sizes)
    chance = np.zeros(2 * total + 1)        # chance[j] belongs to the sum j - total
    chance[total] = 1.0
    for size in sizes:
        moved = np.zeros_like(chance)
        moved[size:] += chance[:-size] / 2
        moved[:-size] += chance[size:] / 2
        chance = moved
    as_far = np.abs(np.arange(-total, total + 1)) >= abs(sum(differences))
    return min(1.0, float(chance[as_far].sum()))


def compare(first: dict[str, str], second: dict[str, str], problems: list[str], samples: int) -> dict:
    """First condition minus second, problem by problem."""
    diffs = [first[p].count("1") - second[p].count("1") for p in problems]       # in samples
    mean, around = band([d / samples for d in diffs])
    return {"difference": mean, "difference_95": around,
            "problems_where_the_first_did_better": sum(d > 0 for d in diffs),
            "problems_where_the_second_did_better": sum(d < 0 for d in diffs), "p": sign_flip_p(diffs)}


def holm(p_values: list[float]) -> list[float]:
    """Holm's correction: each p-value multiplied by the number of tests not yet passed, never falling again."""
    out, worst = [0.0] * len(p_values), 0.0
    for rank, i in enumerate(sorted(range(len(p_values)), key=p_values.__getitem__)):
        worst = max(worst, min(1.0, (len(p_values) - rank) * p_values[i]))
        out[i] = worst
    return out


def analyse(entries: dict[str, dict], pairs: list[tuple[str, str]]) -> dict:
    """Everything the plan asks for, from the three arms' entries."""
    samples = entries["edited"]["samples"]
    found = {name: value for arm in ARMS for name, value in conditions(arm, entries[arm]).items()}
    marks = {name: m for name, (m, _) in found.items()}
    order = entries["edited"]["problems_in_order"]
    problems = front_part(order, marks, samples)
    if len(problems) < 2:
        raise SystemExit(f"only {len(problems)} of {len(order)} problems are complete in all three arms: nothing to compare")
    table = {name: {**share_right(m, problems, samples), "mean_tokens": tokens} for name, (m, tokens) in found.items()}
    compared = [{"first": a, "second": b, **compare(marks[a], marks[b], problems, samples)} for a, b in pairs]
    for c, adjusted in zip(compared[1:], holm([c["p"] for c in compared[1:]])):
        c["p_holm"] = adjusted
    return {"problems_planned": len(order), "problems_analysed": len(problems), "samples_per_problem": samples,
            "conditions": table, "primary": compared[0] if compared else None, "secondary": compared[1:]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True, help="the name the arms were run under: entries <name>-nothink, <name>-edited and <name>-base")
    ap.add_argument("--compare", action="append", default=[], metavar="FIRST,SECOND",
                    help="two condition names; the first --compare is the primary comparison")
    args = ap.parse_args()
    results = json.loads((RESULTS / "phase1.json").read_text())
    out = analyse({arm: results[f"{args.name}-{arm}"] for arm in ARMS}, [tuple(pair.split(",")) for pair in args.compare])
    save("phase1_analysis", args.name, out)

    print(f"{args.name}: {out['problems_analysed']} of {out['problems_planned']} problems analysed, {out['samples_per_problem']} sample(s) each")
    for name, c in sorted(out["conditions"].items(), key=lambda item: item[1]["mean_tokens"]):
        print(f"  {name:<20} {c['accuracy']:6.1%}  ({c['accuracy_95'][0]:.1%} to {c['accuracy_95'][1]:.1%})  {c['mean_tokens']:>8,.0f} tokens a problem")
    for kind, c in [("primary", out["primary"])] * bool(out["primary"]) + [("secondary", c) for c in out["secondary"]]:
        print(f"  {kind}: {c['first']} minus {c['second']}: {c['difference']:+.1%} ({c['difference_95'][0]:+.1%} to {c['difference_95'][1]:+.1%}), "
              f"better on {c['problems_where_the_first_did_better']} problems and worse on {c['problems_where_the_second_did_better']}, p = {c['p']:.3g}"
              + (f" (Holm {c['p_holm']:.3g})" if "p_holm" in c else ""))


if __name__ == "__main__":
    main()
