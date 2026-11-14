"""Checks of a finished Phase 1 run, made from its ledgers after the run.

Why
  bench/phase1_study.zsh leaves three ledgers and their summaries. Before the comparisons between them are
  believed, this script asks three things. Do the ledgers hold what the plan says was run? Are the saved
  verdicts what the saved tokens grade to? And are the forced-exit answers text the model could have written?

What it checks
  ledgers (CPU, half a minute)
    - The plan file's sha256 is in each ledger and each results entry. There is one finished row for each
      planned problem and nothing else.
    - Every row and branch was computed at one batch width. No row itself was set aside or read back in
      (a branch may be read in, and those are counted).
    - Rows stop at their cap or at a stop token. Every row still thinking at an exit point has its branch, which
      starts with the exit phrase, comes after as many tokens as the row had at that point and has the answer
      budget. The cap, the exit points and the budget are the ledger's own settings. They are saved with the
      checks, to be held against the plan by eye: the plan is prose.
    - Every verdict again: each row is decoded and graded here, looked up by the problem's id, and compared
      with the verdict strings the analysis used. Each comparison of the saved analysis is then counted again
      from these (its counts and p-value, not its intervals).
    - The run's own summary found every row and every branch by its key, a hash of the prompt, the tokens up to
      the exit point, the seed and the budget.
    - The run planner (bench/plan_sim.py), replaying the saved row lengths, takes the same number of steps and
      makes the same number of branches wait or be read in as the engine did.
    It also counts, without a verdict: what became of the forced rows at each exit point (right, a wrong boxed
    answer, none, right before the last 500 tokens of the budget, the whole budget written and how many of those
    end in a loop by the study's test, read in from saved tokens), and how many rows are right in their first
    500, 1,000 and 2,000 tokens, to hold against a forced exit's budget.
  answers (GPU, about half an hour with the default sample)
    A forced-exit branch continues from its row's history inside the engine. If that history were damaged, the
    answer would still read like text and only be wrong more often. So a sample of branches is read again by
    mlx-lm alone: the prompt, the row up to the exit point and the branch, through mlx-lm's own cache. At every
    sampled token it asks whether, with mlx-lm's scores and the sampling rule written the slow way, this token
    could have been drawn. Rounding puts a token on the other side of a cut now and then (bench/engine_checks.py
    counted 1 in 8,192). As a yardstick, two answers of each group are also scored after ANOTHER row's text.
    A group fails if more than 1 in 500 of its tokens could not have been drawn. Base rows stop one token
    after the last exit point, so the branches made there are sampled more heavily (--answers-at-row-end).

Run:     HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 uv run python -m bench.phase1_checks --name p1 --plan docs/phase1-prereg.md
         (--answers 0 leaves the GPU part out)
Changes: writes one entry in bench/results/phase1_checks.json. Reads the ledgers under runs/phase1/, the two
         Phase 1 files in bench/results/ and the study's folder, and changes none of them. Exits with 1 if a check fails.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from math import comb
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.utils import load_tokenizer

from bench import grading
from bench.common import RESULTS, save
from bench.phase1 import ANSWER_BUDGETS, plain_text, study, subset
from bench.phase1_analysis import ARMS
from bench.plan_sim import load, pool_pages, simulate
from lab.gen.ledger import Ledger, file_hash
from lab.gen.loops import ends_in_loop

ROOT = Path(__file__).resolve().parent.parent
TAIL = 512              # a row's own last tokens before the exit point, checked along with the answer
MOST_OUTSIDE = 1 / 500  # the largest share of tokens that may fall outside what could be drawn
OUTCOMES = ("rows_forced", "right", "boxed_a_wrong_answer", "no_boxed_answer", "right_before_the_last_500_tokens_of_the_budget",
            "wrote_the_whole_budget", "wrote_the_whole_budget_and_end_in_a_loop", "read_in_from_saved_tokens", "read_in_and_right")


def ledger_checks(tag: str, entry: dict, plan_hash: str, tok, s) -> tuple[dict, dict, dict[str, dict[str, str]]]:
    """One arm: its checks {what: passed}, some counts, and its verdicts {condition: {problem id: "1" or "0"}} graded again."""
    arm = tag.rsplit("-", 1)[1]
    ledger = Ledger(ROOT / "runs/phase1" / tag)
    meta, records = ledger.meta(), ledger.records()
    cap, exits = meta["trunk_tokens"], meta["exits"]
    force_ids = tok.encode(s.force_exit, add_special_tokens=False)
    rows = [r for r in records if r["meta"]["kind"] == "trunk"]
    branches = {(r["meta"]["unique_id"], r["meta"]["exit_at"]): r for r in records if r["meta"]["kind"] == "exit"}
    ok = {
        "the plan's sha256 is in the ledger and in the results entry": meta["plan_sha256"] == plan_hash == entry["plan_sha256"],
        "one finished row for each planned problem, and nothing else in the ledger":
            sorted(r["meta"]["unique_id"] for r in rows) == sorted(entry["problems_in_order"]) and len(records) == len(rows) + len(branches),
        "every row and branch was computed at one batch width": all(r["widths"] == [entry["slots"]] for r in records),
        "no row itself was set aside or read back in": not any(r["set_aside"] or r["read_in"] for r in rows),
        "rows stop at the cap their ledger was started with, or at a stop token":
            all(r["max_tokens"] == cap and (r["finish"], r["written"] == cap) in (("length", True), ("stop", False)) for r in rows),
    }

    # Every verdict again, by problem id. A row that had finished or closed its thinking before an exit point counts as it ran.
    caps = [c for c in sorted({4096, 8192, 16384, 24576, cap}) if c <= cap]
    marks: dict[str, dict[str, str]] = {f"{arm}@{c}": {} for c in caps} | {f"{arm}+exit@{b}": {} for b in exits}
    forced = {b: dict.fromkeys(OUTCOMES, 0) for b in exits}
    right_in_the_first = dict.fromkeys(ANSWER_BUDGETS, 0)    # the rows themselves cut as short as a forced exit's answer may be
    as_planned = 0
    for r in rows:
        uid, gold, tokens = r["meta"]["unique_id"], r["meta"]["gold"], ledger.tokens(r)
        grade = lambda ids: grading.grade(plain_text(tok, ids), gold)
        for c in caps:
            marks[f"{arm}@{c}"][uid] = "01"[grade(tokens[:c])[1]]
        for a in right_in_the_first:
            right_in_the_first[a] += grade(tokens[:a])[1]
        for b in exits:
            mark = marks[f"{arm}+exit@{b}"]
            if len(tokens) <= b or s.think_close in tokens[:b]:
                mark[uid] = marks[f"{arm}@{cap}"][uid]
                continue
            branch = branches.get((uid, b))
            if branch is None:
                mark[uid] = "-"
                continue
            written = ledger.tokens(branch)
            as_planned += (written[: len(force_ids)] == force_ids and branch["max_tokens"] == len(force_ids) + meta["answer_tokens"]
                           and branch["prompt_tokens"] == r["prompt_tokens"] + b and len(written) == branch["written"] <= branch["max_tokens"])
            answer, right = grade(tokens[:b] + written)
            mark[uid] = "01"[right]
            whole, early = branch["finish"] == "length", written[: branch["max_tokens"] - 500]
            for what, yes in zip(OUTCOMES, (True, right, answer is not None and not right, answer is None,
                                            grade(tokens[:b] + early)[1], whole,
                                            whole and ends_in_loop(plain_text(tok, written)), branch["read_in"], branch["read_in"] and right)):
                forced[b][what] += yes
    if exits:
        ok["every row still thinking at an exit point has its branch, and there is no other branch"] = (
            sum(f["rows_forced"] for f in forced.values()) == len(branches) and not any("-" in m.values() for m in marks.values()))
        ok["every branch starts with the exit phrase, comes after as many tokens as its row had at the exit point, "
           "and has the answer budget its ledger was started with"] = as_planned == len(branches)

    saved = next(iter(entry["conditions"].values()))
    order = [key.rsplit("#", 1)[0] for key in saved["correct_by_problem"]]
    strings = ({f"{arm}@{c}": v["by_row"] for c, v in saved["at_smaller_caps"].items()}
               | {f"{arm}+exit@{b}": v["by_row"] for b, v in saved["forced_exit"].items()})
    ok["every verdict graded here again, by problem id, is the saved one"] = (
        order == entry["problems_in_order"] and strings.keys() == marks.keys() and all("".join(marks[c][u] for u in order) == strings[c] for c in marks))

    ok["the run's own summary found every row and every branch by its key"] = (
        saved["rows_not_finished"] == 0 and not any(f["branches_missing"] for f in saved["forced_exit"].values()))

    replay = simulate(load(tag), slots=entry["slots"], pool_pages=pool_pages(entry["pool_gb"]), keep_free=entry["keep_free"], exits=bool(exits),
                      narrow=not entry["kept_width"])
    ok["the planner's replay takes the same steps and makes the same branches wait or be read in"] = (
        [replay[k] for k in ("steps", "set_aside", "branches_sharing", "branches_that_waited", "branches_read_in", "tokens_read")]
        == [entry[k] for k in ("steps", "rows_set_aside", "branches_sharing_history", "branches_that_waited_for_a_slot", "branches_read_in", "tokens_read")])
    counts = {"settings": {k: meta[k] for k in ("arm", "thinking", "n", "samples", "run_seed", "max_tokens", "trunk_tokens", "exits", "answer_tokens")},
              "rows": len(rows), "rows_at_the_cap": sum(r["written"] == cap for r in rows), "branches": len(branches),
              "forced_exits": forced, "rows_right_in_their_first_tokens": right_in_the_first,
              "minutes": entry["wall_s"] / 60, "minutes_by_the_planner": replay["minutes"]}
    return ok, counts, marks


def recount(first: dict[str, str], second: dict[str, str]) -> dict:
    """First minus second: the problems only one of the two got right, and the exact two-sided binomial p-value."""
    a, b = sum(first[u] > second[u] for u in first), sum(first[u] < second[u] for u in first)
    return {"problems_where_the_first_did_better": a, "problems_where_the_second_did_better": b,
            "p": min(1.0, 2 * sum(comb(a + b, i) for i in range(min(a, b) + 1)) / 2 ** (a + b))}


def outside(model, prompt: list[int], row: list[int], answer: list[int], forced: int) -> dict:
    """How many tokens could not have been drawn, by mlx-lm's own scores: among the last TAIL of `row`, and among `answer` after its forced start."""
    text = model.language_model.model
    full = prompt + row + answer
    first = len(prompt) + len(row) - TAIL                           # the first token that is checked
    cache = model.language_model.make_cache()
    for start in range(0, first - 1, 2048):                         # what comes before it is only read
        text(mx.array(full[start:min(start + 2048, first - 1)])[None], cache=cache)
        mx.eval([c.state for c in cache])
    seen = set(row[:-TAIL])                                         # the presence penalty counts every token written so far
    out = {"row_tokens": 0, "row_outside": 0, "answer_tokens": 0, "answer_outside": 0}
    for start in range(first - 1, len(full) - 1, 512):
        piece = full[start:min(start + 512, len(full) - 1)]
        scores = np.array(text.embed_tokens.as_linear(text(mx.array(piece)[None], cache=cache)[0]).astype(mx.float32))
        mx.eval([c.state for c in cache])
        for j, x in enumerate(scores):
            at = start + j + 1 - len(prompt) - len(row)             # these scores are for the answer's token `at`, or for a row token if negative
            token = full[start + j + 1]
            if at < 0 or at >= forced:
                x[list(seen)] -= 1.5
                x /= 0.6
                best = np.argpartition(-x, 20)[:20]
                best = best[np.argsort(-x[best], kind="stable")]
                probs = np.exp(x[best] - x[best[0]])
                probs /= probs.sum()
                part = "row" if at < 0 else "answer"
                out[f"{part}_tokens"] += 1
                out[f"{part}_outside"] += token not in best[(np.cumsum(probs) - probs) < 0.95]
            seen.add(token)
    return out


def answer_checks(name: str, harness: Path, s, per_exit: int, at_row_end: int) -> dict:
    """The GPU part: sampled forced-exit branches read again by mlx-lm alone. {group: counts}"""
    groups: dict[str, dict] = {}
    for arm in ARMS:
        ledger = Ledger(ROOT / "runs/phase1" / f"{name}-{arm}")
        meta, records = ledger.meta(), ledger.records()
        if not meta["exits"]:
            continue
        model, tok = s.load_arm(meta["arm"])
        force_ids = tok.encode(s.force_exit, add_special_tokens=False)
        problems = {p["unique_id"]: p["problem"] for p in subset(harness, meta["n"], meta["subset_seed"])}
        rows = {r["meta"]["unique_id"]: r for r in records if r["meta"]["kind"] == "trunk"}
        picked: dict[str, list[dict]] = {}
        for b in meta["exits"]:
            for read_in in (False, True):
                found = [r for r in records if r["meta"]["kind"] == "exit" and r["meta"]["exit_at"] == b and r["read_in"] == read_in]
                at_end = meta["trunk_tokens"] == b + 1
                label = f"{arm}, exit at {b:,}" + (", the row stops one token later" * at_end) + (", read in from saved tokens" * read_in)
                if found:
                    picked[label] = random.Random(0).sample(found, min(len(found), at_row_end if at_end else per_exit))
        for label, sample in picked.items():
            total = Counter(branches=len(sample))
            for n, branch in enumerate(sample):
                uid, b = branch["meta"]["unique_id"], branch["meta"]["exit_at"]
                prompt = s.chat_prompt(tok, problems[uid], thinking=meta["thinking"])
                assert len(prompt) == rows[uid]["prompt_tokens"], "the prompt built here is not the one the run used"
                answer = ledger.tokens(branch)
                total.update(outside(model, prompt, ledger.tokens(rows[uid])[:b], answer, len(force_ids)))
                if n < 2:       # the yardstick: the same answer after the first long enough other row's text
                    other = next(r for u, r in rows.items() if u != uid and r["written"] > b)
                    wrong = outside(model, prompt, ledger.tokens(other)[:b], answer, len(force_ids))
                    total.update(answer_tokens_after_another_rows_text=wrong["answer_tokens"], outside_after_another_rows_text=wrong["answer_outside"])
                mx.clear_cache()
            groups[label] = dict(total)
            print(f"  {label}: {total['branches']} branches, {total['answer_outside']} of {total['answer_tokens']:,} answer tokens could not have been drawn, "
                  f"{total['row_outside']} of {total['row_tokens']:,} of the rows' last tokens; after another row's text "
                  f"{total['outside_after_another_rows_text']} of {total['answer_tokens_after_another_rows_text']:,}", flush=True)
        del model
        mx.clear_cache()
    return groups


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True, help="the name the arms were run under: ledgers <name>-nothink, <name>-edited and <name>-base")
    ap.add_argument("--plan", type=Path, required=True, help="the plan file the run was started with")
    ap.add_argument("--harness", type=Path, default=Path.home() / "Documents/Dev/neural-harness")
    ap.add_argument("--answers", type=int, default=8, help="branches read again by mlx-lm for each exit point of each arm (0: leave the GPU part out)")
    ap.add_argument("--answers-at-row-end", type=int, default=60, help="the same for an exit point one token before the row's cap")
    args = ap.parse_args()

    harness = args.harness.expanduser()
    s = study(harness)
    tok = load_tokenizer(harness / "models/base")
    results = json.loads((RESULTS / "phase1.json").read_text())
    analysis = json.loads((RESULTS / "phase1_analysis.json").read_text())[args.name]
    out: dict = {"plan_sha256": file_hash(args.plan), "ledgers": {}, "checks": {}}
    marks: dict[str, dict[str, str]] = {}
    for arm in ARMS:
        tag = f"{args.name}-{arm}"
        ok, out["ledgers"][tag], verdicts = ledger_checks(tag, results[tag], out["plan_sha256"], tok, s)
        marks |= verdicts
        out["checks"] |= {f"{tag}: {what}": passed for what, passed in ok.items()}

    # The comparisons of the saved analysis, counted again from the verdicts above.
    planned = [analysis["primary"], *analysis["secondary"]]
    out["comparisons"] = [{"first": c["first"], "second": c["second"], **recount(marks[c["first"]], marks[c["second"]])} for c in planned]
    out["checks"]["every comparison of the saved analysis, counted again, gives its counts and p-value"] = all(
        all(mine[k] == c[k] for k in ("problems_where_the_first_did_better", "problems_where_the_second_did_better"))
        and abs(mine["p"] - c["p"]) <= 1e-9 * c["p"] for mine, c in zip(out["comparisons"], planned))
    for what, passed in out["checks"].items():
        print(("  ok     " if passed else "  WRONG  ") + what, flush=True)

    if args.answers:
        out["answers"] = answer_checks(args.name, harness, s, args.answers, args.answers_at_row_end)
        out["checks"]["no group of forced-exit answers has more than 1 token in 500 that could not have been drawn"] = all(
            g["answer_outside"] <= MOST_OUTSIDE * g["answer_tokens"] and g["row_outside"] <= MOST_OUTSIDE * g["row_tokens"] for g in out["answers"].values())
    save("phase1_checks", args.name, out)
    wrong = [what for what, passed in out["checks"].items() if not passed]
    print(f"{args.name}: " + (f"{len(wrong)} of {len(out['checks'])} checks WRONG: " + "; ".join(wrong) if wrong else f"all {len(out['checks'])} checks passed"))
    raise SystemExit(bool(wrong))


if __name__ == "__main__":
    main()
