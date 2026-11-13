"""Speed experiment 19 (groundwork): three shortcuts checked against the study's saved transcripts, without a GPU.

Background
  Before building the row engine (docs/design-row-engine.md), three ideas for making long-reasoning
  evals cheaper could be tested on transcripts the neural-harness study had already saved: 900
  from Phase 0 (text only, turned back into tokens here) and 2,832 from SD1 (token ids stored).

What it measures
  repeats      Could a second lane guess the next tokens from the text's own earlier repeats and have
               the model check several at once? Each transcript is replayed: whenever its last 8 tokens
               occurred before, what followed then is the guess, and it counts until the first token
               that differs from what was really written.
  loop_rule    What would "stop a row after N looping tokens" have changed? Uses the study's loop test
               (lab/gen/loops.py) and the study's grader on the row cut at the stopping point.
  shorter_cap  Would a cap of 1,024, 2,048 or 4,096 tokens rank the 38 saved edit variants the way the
               8,192 cap did? A shorter cap is the start of the same generation, so no model is needed.
  early_decision  For comparisons the study made on 100 problems: at which problem would a test that may
               be looked at after every problem (lab/stats/sequential.py) have been decided? Problems
               are taken in the study's own order, which was drawn at random before any run.

Run:     uv run python -m bench.transcripts --harness ~/Documents/Dev/neural-harness
Changes: writes bench/results/transcripts.json. Reads the study's folder, never writes to it.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from bench.common import save
from lab.gen.loops import ends_in_loop, stop_table
from lab.stats.sequential import PairedTest, mcnemar_exact

NGRAM, FIRST_GUESS, LONGEST_GUESS = 8, 8, 64
STOPS = ("<|im_end|>", "<|endoftext|>")
STOP_IDS = (248046, 248044)
CAPS = (1024, 2048, 4096, 8192)
CAP_MARGIN, LOOP_MARGIN = 5, 10     # the study's gates: this many fewer rows at the cap / in a loop than base, out of 64


def replay(ids: list[int]) -> dict:
    """Replay one transcript through a guess-and-check lane; count passes and tokens by kind."""
    last: dict[tuple, int] = {}     # the 8 tokens before a position -> the latest such position
    stats = {"tokens": len(ids), "single_steps": 0, "guess_passes": 0, "guess_tokens": 0, "passes_ended_early": 0}
    t, indexed, k = 0, NGRAM, FIRST_GUESS

    def index_upto(end: int) -> None:
        nonlocal indexed
        for e in range(indexed, end + 1):
            last[tuple(ids[e - NGRAM:e])] = e
        indexed = max(indexed, end + 1)

    while t < len(ids):
        source = last.get(tuple(ids[t - NGRAM:t])) if t >= NGRAM else None
        index_upto(t)
        if source is None or source >= t:
            stats["single_steps"] += 1
            t, k = t + 1, FIRST_GUESS
            continue
        guess = ids[source:source + k - 1]
        accepted = 0
        while accepted < len(guess) and t + accepted < len(ids) and guess[accepted] == ids[t + accepted]:
            accepted += 1
        gained = min(accepted + 1, len(ids) - t)    # the model's own next token comes with the pass
        stats["guess_passes"] += 1
        stats["guess_tokens"] += gained
        if accepted == len(guess) == k - 1:
            k = min(2 * k, LONGEST_GUESS)
        else:
            stats["passes_ended_early"] += 1
            k = FIRST_GUESS
        t += gained
        index_upto(t - 1)       # not t itself: the next lookup must find an earlier occurrence, not this position
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--harness", type=Path, default=Path.home() / "Documents/Dev/neural-harness")
    args = ap.parse_args()
    root = args.harness.expanduser()
    sys.path.insert(0, str(root))
    from harness.grader import grade  # the study's own grader, unchanged

    tok = Tokenizer.from_file(str(root / "models/base/tokenizer.json"))
    decode = lambda ids: tok.decode(ids, skip_special_tokens=False)

    def text(ids: list[int]) -> str:
        out = decode(ids)
        for stop in STOPS:
            out = out.removesuffix(stop)
        return out

    def phase0(tag: str) -> list[dict]:
        meta = {json.loads(line)["unique_id"]: json.loads(line) for line in open(root / f"runs/{tag}.jsonl")}
        rows = []
        for line in gzip.open(root / f"runs/{tag}.text.jsonl.gz", "rt"):
            rec = json.loads(line)
            m = meta[rec["unique_id"]]
            rows.append({"ids": tok.encode(rec["text"], add_special_tokens=False).ids, "gold": m["gold"], "correct": m["correct"],
                         "recorded": m["tokens"], "id": rec["unique_id"]})
        return rows

    def sd1(path: Path) -> list[dict]:
        meta = {json.loads(line)["unique_id"]: json.loads(line) for line in open(str(path).replace(".text.jsonl.gz", ".jsonl"))}
        return [{"ids": rec["tokens"], "gold": meta[rec["unique_id"]]["gold"], "correct": meta[rec["unique_id"]]["correct"], "id": rec["unique_id"]}
                for rec in map(json.loads, gzip.open(path, "rt"))]

    groups = {f"Phase 0 {tag}": phase0(tag) for tag in ("edited_32k", "base", "edited", "adapter_s0.5", "adapter_s1.0", "adapter_s1.5", "base_nothink")}
    validation = {p.name.removesuffix(".text.jsonl.gz"): sd1(p) for p in sorted((root / "runs/sd1/gen").glob("validation__*.text.jsonl.gz"))}
    groups["SD1 validation (38 runs)"] = [row for rows in validation.values() for row in rows]
    groups["SD1 test and discovery"] = [row for p in sorted((root / "runs/sd1/gen").glob("*.text.jsonl.gz"))
                                        if not p.name.startswith("validation__") for row in sd1(p)]

    # 1. repeats ---------------------------------------------------------------------------------
    repeats = {}
    for name in ("Phase 0 edited_32k", "Phase 0 base", "Phase 0 edited", "Phase 0 base_nothink"):
        total = {"tokens": 0, "single_steps": 0, "guess_passes": 0, "guess_tokens": 0, "passes_ended_early": 0}
        for row in groups[name]:
            for key, value in replay(row["ids"]).items():
                total[key] += value
        passes = total["single_steps"] + total["guess_passes"]
        repeats[name] = {**total, "tokens_per_pass": total["tokens"] / passes, "guessed_share": total["guess_tokens"] / total["tokens"],
                         "tokens_per_guess_pass": total["guess_tokens"] / max(total["guess_passes"], 1)}
        print(f"repeats, {name}: {repeats[name]['guessed_share']:.1%} of tokens come from accepted guesses, "
              f"{repeats[name]['tokens_per_guess_pass']:.1f} per guessing pass, {total['passes_ended_early'] / max(total['guess_passes'], 1):.0%} of those "
              f"passes ended early; {repeats[name]['tokens_per_pass']:.2f} tokens per model pass overall", flush=True)

    # 2. loop rule -------------------------------------------------------------------------------
    loop_rule = {}
    for name, rows in groups.items():
        regrade_differs = sum(grade(text(r["ids"]), r["gold"])[1] != r["correct"] for r in rows)
        table = stop_table([(r["ids"], r["correct"]) for r in rows], decode, lambda i, cut: grade(text(rows[i]["ids"][:cut]), rows[i]["gold"])[1])
        loop_rule[name] = {"rows": len(rows), "correct": sum(r["correct"] for r in rows), "tokens": sum(len(r["ids"]) for r in rows),
                           "regrade_differs_from_record": regrade_differs, "by_limit": table}
        print(f"loop rule, {name} ({len(rows)} rows, {loop_rule[name]['correct']} correct): " + "; ".join(
            f"after {line['limit']:,}: saves {line['saved_share']:.1%}, loses {line['lost']}, gains {line['gained']}" for line in table), flush=True)

    # 3. shorter cap -----------------------------------------------------------------------------
    def rank(x: list[float]) -> np.ndarray:
        x = np.asarray(x, float)
        r = np.empty(len(x))
        r[np.argsort(x, kind="stable")] = np.arange(len(x))
        for v in np.unique(x):
            r[x == v] = r[x == v].mean()
        return r

    counts = {tag: {c: {"still_running": 0, "looping": 0, "correct": 0, "tokens": 0} for c in CAPS} for tag in validation}
    for tag, rows in validation.items():
        for row in rows:
            for c in CAPS:
                cut = row["ids"][:c]
                cell = counts[tag][c]
                cell["still_running"] += not (len(row["ids"]) <= c and row["ids"][-1] in STOP_IDS)
                cell["looping"] += ends_in_loop(text(cut))
                cell["correct"] += grade(text(cut), row["gold"])[1]
                cell["tokens"] += len(cut)
    tags = list(validation)
    base = [t for t in tags if "__base__" in t]
    full = {key: [counts[t][8192][key] for t in tags] for key in ("still_running", "looping")}
    full_gate = {"looping": [np.mean([counts[t][8192]["looping"] for t in base]) - x >= LOOP_MARGIN for x in full["looping"]],
                 "still_running": [np.mean([counts[t][8192]["still_running"] for t in base]) - x >= CAP_MARGIN for x in full["still_running"]]}
    shorter_cap = {}
    for c in CAPS:
        at = {key: [counts[t][c][key] for t in tags] for key in ("still_running", "looping")}
        gate = {"looping": [np.mean([counts[t][c]["looping"] for t in base]) - x >= LOOP_MARGIN for x in at["looping"]],
                "still_running": [np.mean([counts[t][c]["still_running"] for t in base]) - x >= CAP_MARGIN for x in at["still_running"]]}
        shorter_cap[str(c)] = {
            "tokens_share": sum(counts[t][c]["tokens"] for t in tags) / sum(counts[t][8192]["tokens"] for t in tags),
            "base_looping_of_64": float(np.mean([counts[t][c]["looping"] for t in base])),
            "rank_agreement_looping": float(np.corrcoef(rank(at["looping"]), rank(full["looping"]))[0, 1]),
            "rank_agreement_still_running": float(np.corrcoef(rank(at["still_running"]), rank(full["still_running"]))[0, 1]),
            "loop_gate_passes": int(sum(gate["looping"])), "loop_gate_same_decisions": int(sum(a == b for a, b in zip(gate["looping"], full_gate["looping"]))),
            "cap_gate_passes": int(sum(gate["still_running"])),
            "cap_gate_same_decisions": int(sum(a == b for a, b in zip(gate["still_running"], full_gate["still_running"]))), "runs": len(tags)}
        s = shorter_cap[str(c)]
        print(f"shorter cap {c:,}: {s['tokens_share']:.0%} of the tokens; base shows {s['base_looping_of_64']:.1f} of 64 looping; rank agreement with 8,192 "
              f"{s['rank_agreement_looping']:.2f} (loops), {s['rank_agreement_still_running']:.2f} (rows still running); loop gate passes "
              f"{s['loop_gate_passes']} runs, same decision as at 8,192 for {s['loop_gate_same_decisions']} of {s['runs']}", flush=True)

    # 4. early decision ---------------------------------------------------------------------------
    def decided(first: list[dict], second: list[dict], measure) -> dict:
        other = {row["id"]: row for row in second}
        test, at = PairedTest(), None
        for i, row in enumerate(first):
            test.add(measure(row), measure(other[row["id"]]))
            if at is None and test.decided:
                at = i + 1
        return {"problems": test.pairs, "only_first": test.only_a, "only_second": test.only_b, "decided_at_problem": at,
                "fixed_size_p": mcnemar_exact(test.only_a, test.only_b)}

    in_loop = lambda row: ends_in_loop(text(row["ids"]))
    right = lambda row: row["correct"]
    test_runs = {name: sd1(root / f"runs/sd1/gen/test__{name}__s0.text.jsonl.gz") for name in ("base", "astra")}
    early_decision = {
        "Phase 0, 8,192 cap: base ends in a loop more often than edited": decided(groups["Phase 0 base"], groups["Phase 0 edited"], in_loop),
        "SD1 test: base ends in a loop more often than the edit": decided(test_runs["base"], test_runs["astra"], in_loop),
        "SD1 test: the edit is right more often than base": decided(test_runs["astra"], test_runs["base"], right),
        "Phase 0: edited at 32,768 is right more often than base with thinking off": decided(groups["Phase 0 edited_32k"], groups["Phase 0 base_nothink"], right),
        "Phase 0, forced exit at 6,000: edited is right more often than base": decided(phase0("edited_budget6k"), phase0("base_budget6k"), right),
    }
    for name, d in early_decision.items():
        print(f"early decision, {name}: {d['only_first']} against {d['only_second']} problems; "
              + (f"decided at problem {d['decided_at_problem']} of {d['problems']}" if d["decided_at_problem"] else f"not decided in {d['problems']}")
              + f"; the fixed-size test gives p = {d['fixed_size_p']:.3g}", flush=True)

    path = save("transcripts", "neural-harness", {"harness": str(args.harness), "repeats": repeats, "loop_rule": loop_rule,
                                                  "shorter_cap": shorter_cap, "early_decision": early_decision})
    print(f"saved to {path}")


if __name__ == "__main__":
    main()
