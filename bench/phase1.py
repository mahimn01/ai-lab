"""Speed experiment 19: a condition of the neural-harness Phase 1 workload, run through the row engine.

Background
  The neural-harness study (~/Documents/Dev/neural-harness) evaluates Qwen3.5-0.8B on a fixed
  100-problem subset of MATH-500 with thinking on, the blog's sampling and a cap of up to 32,768
  new tokens. Its own runner needed 1.74 hours for one such condition. This script runs the
  same condition through lab/gen (design: docs/design-row-engine.md).

Same workload by construction
  The prompts, the forced-exit phrase, the stop tokens, the sampling settings, the model loader
  and the grader are imported from the study's own code (--harness), not copied. The subset is
  drawn the study's way and checked against its runs/subset_ids.json.
  --grader math-verify grades as the blog does instead (bench/grading.py), which is what the
  study names for Phase 1. Grades are made from the saved tokens, so --report can regrade a
  finished ledger with either grader (give it another --save-as).

What it does
  1. Builds one row per problem and sample. With --exits, each row also branches at those token
     counts when its thinking is still open: the branch takes the study's exit phrase and answers.
     A branch shares its trunk's history; if no slot is free it waits for one, and only when
     that is not possible is it read in later from the saved tokens.
     A branch may write --answer-tokens tokens after the phrase. (The study's own forced exit ran
     on to the overall cap; this is the same exit with a stated answer budget, and the report says
     how many branches used the budget up.)
     --trunk-tokens stops the rows themselves earlier than --max-tokens, which then only limits
     a branch. A row branches only while it is still running, so one past the last exit point is
     the least that still takes every branch.
  2. Runs the rows through the engine, writing each finished row to a ledger at once. A second
     start with the same --tag continues where the first stopped; exit points may be added later
     and are then made from the saved rows.
  3. Grades with the study's grader and reports: wall-clock, tokens per second, accuracy, rows at
     the cap, rows ending in a loop (the study's test), accuracy at smaller caps (the same rows cut
     short), each forced-exit condition, and what a loop-stopping rule would have changed.
  4. Compares with saved runs of the study's own runner for the same condition (--reference).
     Those runs drew other random numbers, so this compares rates and distributions; with 100
     problems it can only show large differences. bench/engine_checks.py does the direct checks.
  With --decide A,B (two conditions in one run) it stops starting rows once a test that may be
  looked at after every problem has decided which of the two is right more often. Problems are
  taken in the subset's own order, drawn at random before the run; each problem counts once.

Run:     uv run python -m bench.phase1 --harness ~/Documents/Dev/neural-harness --arm edited --tag edited-32k
         (add --report to recompute the summary from a finished ledger without running anything)
Changes: writes runs/phase1/<tag>/ (ledger) and one entry in bench/results/phase1.json.
         Reads the study's folder, never writes to it. Downloads MATH-500 if it is not cached.
         An entry's `environment` (time, free memory, swap) is taken as the run starts. The nine
         entries saved up to 2026-11-13 took it as the run ended, whatever their field names say.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx

from bench import grading
from bench.common import environment, save
from lab.gen.engine import Engine, Row, seed_for
from lab.gen.ledger import Ledger, file_hash
from lab.gen.loops import audited, ends_in_loop, stop_table
from lab.stats.sequential import PairedTest, mcnemar_exact

ROOT = Path(__file__).resolve().parent.parent
STOP_TEXT = ("<|im_end|>", "<|endoftext|>")
ANSWER_BUDGETS = (500, 1000, 2000)      # smaller answer budgets a forced-exit branch is also graded at, by cutting it
FIXED = ("arm", "max_tokens", "trunk_tokens", "thinking", "run_seed", "subset_seed", "sampling", "stop_after_loop", "audit", "answer_tokens",
         "plan_sha256")


def plain_text(tok, tokens: list[int]) -> str:
    """Tokens as the text the study grades: special tokens kept, a trailing stop string removed."""
    out = tok.decode(tokens, skip_special_tokens=False)
    for stop in STOP_TEXT:
        out = out.removesuffix(stop)
    return out


def study(harness: Path) -> SimpleNamespace:
    """The study's own pieces, imported from its folder."""
    sys.path.insert(0, str(harness))
    from harness.generate import FORCE_EXIT, STOP_IDS, THINK_CLOSE, chat_prompt
    from harness.grader import grade
    from harness.model import load_adapter_weights, load_arm
    from harness.sampling import BLOG_SAMPLING

    return SimpleNamespace(force_exit=FORCE_EXIT, stop_ids=tuple(STOP_IDS), think_close=THINK_CLOSE, chat_prompt=chat_prompt,
                           grade=grade, load_arm=load_arm, load_adapter_weights=load_adapter_weights, sampling=BLOG_SAMPLING)


def subset(harness: Path, n: int, seed: int) -> list[dict]:
    """The study's problems: random.Random(seed).sample(range(500), n), in that order."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = [dict(ds[i], index=i) for i in random.Random(seed).sample(range(len(ds)), n)]
    recorded = json.loads((harness / "runs/subset_ids.json").read_text())
    if (recorded["seed"], recorded["n"]) == (seed, n) and recorded["unique_ids"] != [r["unique_id"] for r in rows]:
        raise SystemExit("the subset differs from the study's runs/subset_ids.json")
    return rows


def ks_two_sample(a: list[float], b: list[float]) -> tuple[float, float]:
    """Kolmogorov-Smirnov distance between two samples and its large-sample p-value."""
    a, b = sorted(a), sorted(b)
    cdf = lambda xs, v: sum(x <= v for x in xs) / len(xs)
    d = max(abs(cdf(a, v) - cdf(b, v)) for v in sorted(set(a) | set(b)))
    ne = len(a) * len(b) / (len(a) + len(b))
    lam = (math.sqrt(ne) + 0.12 + 0.11 / math.sqrt(ne)) * d
    if lam < 0.3:       # the series below does not converge this close to zero, where the p-value is 1
        return d, 1.0
    return d, min(1.0, max(0.0, 2 * sum((-1) ** (j - 1) * math.exp(-2 * j * j * lam * lam) for j in range(1, 101))))


def summarize(ledger: Ledger, s: SimpleNamespace, tok, cap: int, exits: list[int], rows: list[Row], model_tag: str,
              branch_of, stop_limit: int | None, answer_tokens: int) -> dict:
    """Grade the finished rows of this run's conditions and gather their numbers.

    Only the rows this start asked for are counted, looked up by the key they must have, so a ledger
    that also holds other rows (an earlier, different start) cannot leak into the numbers.
    """
    text = lambda tokens: plain_text(tok, tokens)
    records = {r["key"]: r for r in ledger.records()}
    out = {}
    for condition in dict.fromkeys(row.meta["condition"] for row in rows):
        wanted = [(row, row.key(model_tag)) for row in rows if row.meta["condition"] == condition]
        trunks = [(row, records[key]) for row, key in wanted if key in records]
        if not trunks:
            continue
        tokens = [ledger.tokens(r) for _, r in trunks]
        texts = [text(t) for t in tokens]
        graded = [s.grade(t, r["meta"]["gold"])[1] for t, (_, r) in zip(texts, trunks)]
        counts = [r["written"] for _, r in trunks]
        n = len(trunks)
        name = lambda r: f"{r['meta']['unique_id']}#{r['meta']['sample']}"
        cut = {c: [s.grade(text(t[:c]), r["meta"]["gold"])[1] for t, (_, r) in zip(tokens, trunks)]
               for c in sorted({4096, 8192, 16384, 24576, cap}) if c <= cap}
        result = {
            "rows": n, "rows_not_finished": len(wanted) - n, "correct": sum(graded), "accuracy": sum(graded) / n,
            "at_cap_share": sum(r["finish"] == "length" and r["written"] >= cap for _, r in trunks) / n,
            "ends_in_loop_share": sum(map(ends_in_loop, texts)) / n,
            "closed_thinking_share": sum(s.think_close in t for t in tokens) / n,
            "looped_at_some_point_share": sum(r.get("loop_first_onset") is not None for _, r in trunks) / n,
            "stopped_by_loop_rule_share": sum(r["finish"] == "loop" for _, r in trunks) / n,
            "tokens": sum(counts), "median_tokens": statistics.median(counts), "mean_tokens": statistics.mean(counts),
            "correct_by_problem": {name(r): bool(g) for (_, r), g in zip(trunks, graded)},
            "tokens_by_problem": {name(r): r["written"] for _, r in trunks},
            # The same rows cut short: what a smaller cap would have reported, and the tokens it would have used.
            # `by_row` has one character per row in the order of correct_by_problem: 1 right, 0 wrong.
            "at_smaller_caps": {str(c): {"accuracy": sum(v) / n, "mean_tokens": statistics.mean(min(k, c) for k in counts),
                                         "by_row": "".join("01"[x] for x in v)} for c, v in cut.items()},
        }
        # Forced exit at b: a row still thinking at b tokens takes the exit phrase and answers (its branch);
        # a row that had finished or closed its thinking by then is the row itself.
        result["forced_exit"] = {}
        for b in exits:
            budgets = {a: 0 for a in ANSWER_BUDGETS if a < answer_tokens}
            used = forced = used_up = 0
            by_row = []                     # as in at_smaller_caps, and "-" for a row whose branch is missing
            for (row, r), mine, g in zip(trunks, tokens, graded):
                branch = branch_of(row, mine[:b]) if len(mine) > b else None
                if branch is None:          # finished before b, or thinking already closed
                    used += len(mine)
                    by_row.append("01"[g])
                    for a in budgets:
                        budgets[a] += g
                    continue
                record = records.get(branch.key(model_tag))
                if record is None:
                    by_row.append("-")
                    continue
                answer = ledger.tokens(record)
                forced += 1
                used_up += record["finish"] == "length"
                by_row.append("01"[s.grade(text(mine[:b] + answer), r["meta"]["gold"])[1]])
                used += b + record["written"]
                for a in budgets:
                    budgets[a] += s.grade(text(mine[:b] + answer[: len(branch.forced) + a]), r["meta"]["gold"])[1]
            result["forced_exit"][str(b)] = {
                "accuracy": by_row.count("1") / n, "mean_tokens": used / n, "rows_forced": forced, "branches_missing": by_row.count("-"),
                "branches_that_used_up_their_answer_budget": used_up, "by_row": "".join(by_row),
                "accuracy_with_a_smaller_answer_budget": {str(a): k / n for a, k in budgets.items()}}
        if stop_limit is None:
            result["if_stopped_on_loops"] = stop_table(
                zip(tokens, graded), lambda ids: tok.decode(ids, skip_special_tokens=False),
                lambda i, cut: s.grade(text(tokens[i][:cut]), trunks[i][1]["meta"]["gold"])[1])
        else:
            # The audit: rows that were exempt from the rule ran to the full budget. What would the rule have done to them?
            exempt = [(t, g, r) for t, g, (row, r) in zip(tokens, graded, trunks) if row.stop_after_loop is None]
            table = stop_table([(t, g) for t, g, _ in exempt], lambda ids: tok.decode(ids, skip_special_tokens=False),
                               lambda i, cut: s.grade(text(exempt[i][0][:cut]), exempt[i][2]["meta"]["gold"])[1], limits=(stop_limit,))
            result["audit_of_the_loop_rule"] = {"rows_run_to_the_full_budget": len(exempt), **table[0]}
        out[condition] = result
    return out


def compare(summary: dict, reference: Path, s: SimpleNamespace) -> dict:
    """This run against a saved run of the study's own runner on the same problems (different random draws)."""
    theirs = [json.loads(line) for line in reference.read_text().splitlines()]
    texts = {r["unique_id"]: r["text"] for r in map(json.loads, gzip.open(str(reference).replace(".jsonl", ".text.jsonl.gz"), "rt"))}
    mine_right = {k.split("#")[0]: v for k, v in summary["correct_by_problem"].items() if k.endswith("#0")}
    mine_tokens = {k.split("#")[0]: v for k, v in summary["tokens_by_problem"].items() if k.endswith("#0")}
    shared = [r for r in theirs if r["unique_id"] in mine_right]
    n = len(shared)
    only_mine = sum(mine_right[r["unique_id"]] and not r["correct"] for r in shared)
    only_theirs = sum(r["correct"] and not mine_right[r["unique_id"]] for r in shared)
    gap = (only_mine - only_theirs) / n
    half = 1.96 * math.sqrt(max(only_mine + only_theirs - (only_mine - only_theirs) ** 2 / n, 0)) / n
    d, p = ks_two_sample([mine_tokens[r["unique_id"]] for r in shared], [r["tokens"] for r in shared])
    meta = json.loads(Path(str(reference).replace(".jsonl", ".meta.json")).read_text())
    return {"reference": str(reference), "problems": n,
            "reference_accuracy": sum(r["correct"] for r in shared) / n,
            "reference_at_cap_share": sum(r["hit_cap"] for r in shared) / n,
            "reference_ends_in_loop_share": sum(ends_in_loop(texts[r["unique_id"]]) for r in shared) / n,
            "reference_tokens": sum(r["tokens"] for r in shared), "reference_median_tokens": statistics.median(r["tokens"] for r in shared),
            "reference_wall_s": meta["wall_s"], "reference_tokens_per_s": meta["agg_tok_s"], "reference_rows_at_once": meta["batch"],
            "right_only_here": only_mine, "right_only_in_reference": only_theirs, "mcnemar_p": mcnemar_exact(only_mine, only_theirs),
            "accuracy_difference": gap, "accuracy_difference_95": [gap - half, gap + half],
            "token_count_ks_distance": d, "token_count_ks_p": p}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--harness", type=Path, default=Path.home() / "Documents/Dev/neural-harness")
    ap.add_argument("--arm", default="edited", help="weights: base or edited (the study's merged tensors)")
    ap.add_argument("--adapter-scales", default="", help="with --arm base: also run the unmerged edit at these scales, in the same batch (e.g. 1.0,1.5)")
    ap.add_argument("--only-adapters", action="store_true", help="with --adapter-scales: leave out the plain base rows")
    ap.add_argument("--tag", required=True, help="names the ledger folder runs/phase1/<tag>")
    ap.add_argument("--save-as", help="name of the entry in bench/results/phase1.json (default: the tag)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--run-seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--trunk-tokens", type=int, help="most tokens a row itself may write (default: --max-tokens). A forced-exit branch "
                    "may still reach --max-tokens in all")
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--exits", default="", help="token counts at which a still-thinking row branches into a forced exit, e.g. 4096,8192,16384,24576")
    ap.add_argument("--answer-tokens", type=int, default=4000, help="most tokens a forced-exit branch may write after the phrase")
    ap.add_argument("--keep-free", type=int, default=0, help="slots held back so that branches start at once (see lab/gen/engine.py)")
    ap.add_argument("--stop-after-loop", type=int, help="stop a row after this many looping tokens (off by default; see lab/gen/loops.py)")
    ap.add_argument("--audit", type=float, default=0.0, help="share of rows exempt from --stop-after-loop, chosen by row key")
    ap.add_argument("--keep-width", action="store_true", help="do not narrow the batch as rows run out: the end of a run is a little slower, "
                    "and every row is computed at one batch width, so it comes out the same in any run of that width unless it is set aside "
                    "or read back in on the way")
    ap.add_argument("--slots", type=int, default=48)
    ap.add_argument("--pool-gb", type=float, default=10.0)
    ap.add_argument("--plain-attention", action="store_true", help="use MLX's attention row by row instead of the paged kernel")
    ap.add_argument("--reference", type=Path, action="append", default=[],
                    help="a saved <tag>.jsonl of the study's own runner for the same condition (may be given more than once)")
    ap.add_argument("--decide", default="", help="two condition names, e.g. base,adapter@1: stop starting rows once their accuracy difference is decided")
    ap.add_argument("--longest-first", type=Path, help="a saved <tag>.jsonl of an earlier run of these problems: start the rows that were longest "
                                                         "there first (a run ends sooner. Each row keeps its own random stream, but another order changes "
                                                         "which rows meet a narrowing of the batch or are set aside, and those can come out differently)")
    ap.add_argument("--plan", type=Path, help="the plan written before the run; its sha256 goes into the ledger, and a ledger "
                    "started under one plan is not continued or reported under another")
    ap.add_argument("--grader", choices=("study", "math-verify"), default="study",
                    help="study: the last box anywhere, compared as strings. math-verify: the blog's protocol (bench/grading.py)")
    ap.add_argument("--report", action="store_true", help="only recompute the summary from the ledger")
    args = ap.parse_args()

    harness = args.harness.expanduser()
    s = study(harness)
    if args.grader == "math-verify":
        s.grade = grading.grade
    exits = sorted(int(x) for x in args.exits.split(",") if x)
    trunk_cap = args.trunk_tokens or args.max_tokens
    if not 0 < trunk_cap <= args.max_tokens:
        raise SystemExit(f"--trunk-tokens must be between 1 and --max-tokens ({args.max_tokens:,})")
    if exits and trunk_cap <= exits[-1]:
        raise SystemExit(f"a row that stops at {trunk_cap:,} tokens never takes the exit at {exits[-1]:,}: use at least {exits[-1] + 1:,}")
    scales = [float(x) for x in args.adapter_scales.split(",") if x]
    problems = subset(harness, args.n, args.subset_seed)
    model, tok = s.load_arm(args.arm)
    model_tag = f"{args.arm}|{model.language_model.model.embed_tokens.weight.dtype}"
    force_ids = tuple(tok.encode(s.force_exit, add_special_tokens=False))
    if exits and args.max_tokens - exits[-1] - len(force_ids) < 1:
        raise SystemExit(f"an exit at {exits[-1]:,} leaves no room for an answer under --max-tokens {args.max_tokens:,}")
    sampling = {"temperature": s.sampling["temperature"], "top_k": s.sampling["top_k"], "top_p": s.sampling["top_p"],
                "presence_penalty": s.sampling["presence_penalty"]}
    settings = {"tag": args.tag, "arm": args.arm, "n": args.n, "samples": args.samples, "max_tokens": args.max_tokens,
                "trunk_tokens": trunk_cap, "thinking": not args.no_thinking, "exits": exits, "answer_tokens": args.answer_tokens, "stop_after_loop": args.stop_after_loop,
                "audit": args.audit, "sampling": sampling, "run_seed": args.run_seed, "subset_seed": args.subset_seed,
                "plan_sha256": file_hash(args.plan) if args.plan else None}

    ledger = Ledger(ROOT / "runs/phase1" / args.tag)
    before = ledger.meta()
    if before:
        before.setdefault("trunk_tokens", before["max_tokens"])      # ledgers from before the option existed
    changed = [name for name in FIXED if before and before.get(name) != settings[name]]
    if changed:
        raise SystemExit(f"runs/phase1/{args.tag} was started with other settings ({', '.join(changed)}); use another --tag")
    resumed = bool(ledger.records())

    # One condition per set of weights: the loaded arm as it is, and (base only) the unmerged edit at each scale.
    conditions = ([] if args.only_adapters else [(args.arm, None, 1.0)]) + [(f"adapter@{x:g}", "astra", x) for x in scales]
    edits = {"astra": s.load_adapter_weights()} if scales else None
    rows, key_of = [], {}
    for p in problems:
        prompt = s.chat_prompt(tok, p["problem"], thinking=not args.no_thinking)
        for sample in range(args.samples):
            for name, edit, scale in conditions:
                row = Row(prompt=prompt, max_tokens=trunk_cap, seed=seed_for(args.run_seed, p["unique_id"], sample),
                          stop_tokens=s.stop_ids, edit=edit, edit_scale=scale, branch_at=tuple(exits), **sampling,
                          meta={"unique_id": p["unique_id"], "index": p["index"], "sample": sample, "condition": name,
                                "gold": p["answer"], "kind": "trunk"})
                if args.stop_after_loop and not audited(row.key(model_tag), args.audit):
                    row.stop_after_loop = args.stop_after_loop
                rows.append(row)
                key_of[name, p["unique_id"], sample] = row.key(model_tag)

    if args.longest_first:
        was = {r["unique_id"]: r["tokens"] for r in map(json.loads, args.longest_first.expanduser().read_text().splitlines())}
        rows.sort(key=lambda row: -was.get(row.meta["unique_id"], 0))      # a stable sort: ties keep the subset's order

    def forced_exit(row: Row, written: list[int]) -> Row | None:
        """The branch for a row that is still thinking at one of the exit points."""
        if s.think_close in written:
            return None
        allowance = min(args.answer_tokens, args.max_tokens - len(written) - len(force_ids))
        return Row(prompt=row.prompt + written, max_tokens=len(force_ids) + allowance, forced=force_ids, seen=tuple(written),
                   seed=seed_for(args.run_seed, row.meta["unique_id"], row.meta["sample"], "exit", len(written)),
                   stop_tokens=row.stop_tokens, edit=row.edit, edit_scale=row.edit_scale, **sampling,
                   meta={**row.meta, "kind": "exit", "exit_at": len(written)})

    text = lambda tokens: plain_text(tok, tokens)

    def new_engine() -> Engine:
        engine = Engine(model, slots=args.slots, pool_gb=args.pool_gb, max_tokens=args.max_tokens, edits=edits, model_tag=model_tag,
                        decode=lambda ids: tok.decode(ids, skip_special_tokens=False), use_kernel=not args.plain_attention)
        engine.narrow_when_idle = not args.keep_width
        return engine

    def branches_still_owed() -> list[Row]:
        """Forced-exit branches of finished rows that are not in the ledger (after a restart, or when --exits was added later)."""
        records = {r["key"]: r for r in ledger.records()}
        owed = []
        for row in rows:
            record = records.get(row.key(model_tag))
            if record is None:
                continue
            written = ledger.tokens(record)
            for b in exits:
                branch = forced_exit(row, written[:b]) if len(written) > b else None
                if branch is not None and branch.key(model_tag) not in records:
                    owed.append(branch)
        return owed

    # Deciding early: grades are kept by row key (so a row is graded once) and start from what the ledger already holds.
    grades: dict[str, bool] = {}
    decision = {"test": None, "graded": 0}
    if args.decide:
        mine = {row.key(model_tag) for row in rows}
        grades = {r["key"]: s.grade(text(ledger.tokens(r)), r["meta"]["gold"])[1] for r in ledger.records() if r["key"] in mine}

    def decided(records: list[dict]) -> bool:
        """Is the accuracy difference between the two --decide conditions settled? Called whenever a row finishes."""
        first, second = args.decide.split(",")
        for r in records[decision["graded"]:]:
            if r["meta"]["kind"] == "trunk":
                grades[r["key"]] = s.grade(text(r["tokens"]), r["meta"]["gold"])[1]
        decision["graded"] = len(records)
        test = PairedTest()
        for p in problems:                      # the subset's order, as far as both conditions have finished every sample
            a = [grades.get(key_of[first, p["unique_id"], k]) for k in range(args.samples)]
            b = [grades.get(key_of[second, p["unique_id"], k]) for k in range(args.samples)]
            if None in a or None in b:
                break
            test.add_problem(sum(a), sum(b))    # one problem, one piece of evidence, however many samples
        decision["test"] = test
        return test.decided

    label = args.save_as or args.tag
    results_file = ROOT / "bench/results/phase1.json"
    entry = json.loads(results_file.read_text()).get(label, {}) if results_file.exists() else {}
    if not args.report:
        ledger.write_meta(settings)
        started = environment()
        engine = new_engine()
        say = lambda line: print(line, flush=True)
        say(f"{label}: {len(rows)} rows, {args.slots} at once, pool {engine.pool.pages} pages "
            f"({engine.pool.pages * engine.pool.bytes_per_page / 2**30:.1f} GB){', continuing an earlier run' if resumed else ''}")
        t0 = time.perf_counter()
        engine.run(rows, ledger, brancher=forced_exit if exits else None, enough=decided if args.decide else None,
                   keep_free=args.keep_free, log=say)
        stats = dict(engine.stats)
        owed = branches_still_owed()
        if owed:
            # A fresh engine: the first one may have narrowed its batch by now.
            engine.close()
            del engine
            mx.clear_cache()
            engine = new_engine()
            say(f"  {len(owed)} branches of finished rows are read in from the ledger")
            engine.run(owed, ledger, log=say)
            stats = {name: stats[name] + engine.stats[name] for name in stats}
        wall = time.perf_counter() - t0
        engine.close()
        entry = {"environment": started, "harness": str(args.harness), **{k: v for k, v in settings.items() if k != "tag"},
                 "slots": args.slots, "pool_gb": args.pool_gb, "keep_free": args.keep_free, "kept_width": args.keep_width,
                 "longest_first_from": str(args.longest_first) if args.longest_first else None,
                 "paged_kernel": not args.plain_attention, "continued_an_earlier_run": resumed, "wall_s": wall,
                 "tokens_written": stats["written"], "tokens_per_s": stats["written"] / wall, "steps": stats["steps"],
                 "tokens_read": stats["read"], "rows_set_aside": stats["set_aside"], "branches_sharing_history": stats["branches_shared"],
                 "branches_that_waited_for_a_slot": stats["branches_waited"],
                 "branches_read_in": stats["branches_read"] + len(owed), "batch_narrowed": stats["narrowed"],
                 "peak_memory_gb": mx.get_peak_memory() / 2**30}
        if decision["test"] is not None:
            test = decision["test"]
            entry["decision"] = {"conditions": args.decide, "problems_analysed": test.pairs, "problems_won_by_first": test.only_a,
                                 "problems_won_by_second": test.only_b, "decided": test.decided, "p_valid_at_every_look": test.p_value}
    entry["grader"] = args.grader
    entry["problems_in_order"] = [p["unique_id"] for p in problems]
    entry["conditions"] = summarize(ledger, s, tok, trunk_cap, exits, rows, model_tag, forced_exit, args.stop_after_loop,
                                    args.answer_tokens)
    if args.reference and entry["conditions"]:
        first = next(iter(entry["conditions"]))
        entry["against_references"] = [compare(entry["conditions"][first], ref.expanduser(), s) for ref in args.reference]
    save("phase1", label, entry)

    if "wall_s" in entry:
        print(f"{label}: {entry['wall_s'] / 60:.1f} min{' (this start only; an earlier run was continued)' if entry['continued_an_earlier_run'] else ''}, "
              f"{entry['tokens_written']:,} tokens written, {entry['tokens_read']:,} read, {entry['tokens_per_s']:,.0f} tok/s, {entry['steps']:,} steps, "
              f"{entry['rows_set_aside']} rows set aside, branches sharing history {entry['branches_sharing_history']} "
              f"({entry.get('branches_that_waited_for_a_slot', 0)} after waiting for a slot), read in {entry['branches_read_in']}, "
              f"peak memory {entry['peak_memory_gb']:.1f} GB", flush=True)
    if "decision" in entry:
        d = entry["decision"]
        print(f"  {d['conditions']}: {'decided' if d['decided'] else 'not decided'} after {d['problems_analysed']} problems "
              f"({d['problems_won_by_first']} won by the first, {d['problems_won_by_second']} by the second; "
              f"p = {d['p_valid_at_every_look']:.3g}, valid at every look)", flush=True)
    for name, c in entry["conditions"].items():
        print(f"  {name} (grader: {args.grader}): {c['correct']}/{c['rows']} correct ({c['accuracy']:.0%}), at the cap {c['at_cap_share']:.0%}, ends in a loop "
              f"{c['ends_in_loop_share']:.0%}, median {c['median_tokens']:,.0f} tokens"
              + (f", {c['rows_not_finished']} rows not finished" if c["rows_not_finished"] else ""), flush=True)
        print("    cut at a smaller cap: " + ", ".join(f"{int(k):,} -> {v['accuracy']:.0%}" for k, v in c["at_smaller_caps"].items()), flush=True)
        for b, e in c["forced_exit"].items():
            print(f"    forced exit at {int(b):,} with up to {args.answer_tokens:,} answer tokens: {e['accuracy']:.0%} with {e['mean_tokens']:,.0f} tokens "
                  f"a problem; {e['rows_forced']} rows forced, {e['branches_that_used_up_their_answer_budget']} used the budget up, "
                  f"{e['branches_missing']} branches missing"
                  + "".join(f"; with a budget of {int(a):,}: {v:.0%}" for a, v in e["accuracy_with_a_smaller_answer_budget"].items()), flush=True)
        for line in c.get("if_stopped_on_loops", []):
            print(f"    stopping after {line['limit']:,} looping tokens would stop {line['stopped']} rows, save {line['saved_share']:.1%} of tokens, "
                  f"lose {line['lost']} correct answers, gain {line['gained']}", flush=True)
        if "audit_of_the_loop_rule" in c:
            a = c["audit_of_the_loop_rule"]
            print(f"    loop rule: {c['stopped_by_loop_rule_share']:.0%} of rows stopped; of {a['rows_run_to_the_full_budget']} exempt rows the rule "
                  f"would have stopped {a['stopped']}, losing {a['lost']} correct answers and gaining {a['gained']}", flush=True)
    for r in entry.get("against_references", []):
        print(f"  against the study's own runner ({Path(r['reference']).name}): accuracy {r['reference_accuracy']:.0%} there, at the cap "
              f"{r['reference_at_cap_share']:.0%}, ends in a loop {r['reference_ends_in_loop_share']:.0%}, median {r['reference_median_tokens']:,.0f} tokens; "
              f"accuracy here minus there {r['accuracy_difference']:+.0%} (95% interval {r['accuracy_difference_95'][0]:+.0%} to "
              f"{r['accuracy_difference_95'][1]:+.0%}); token counts KS {r['token_count_ks_distance']:.2f} (p = {r['token_count_ks_p']:.2f}); "
              f"their wall-clock {r['reference_wall_s'] / 60:.1f} min at {r['reference_tokens_per_s']:.0f} tok/s with {r['reference_rows_at_once']} rows at once", flush=True)


if __name__ == "__main__":
    main()
