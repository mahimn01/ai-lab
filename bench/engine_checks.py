"""Speed experiment 19 (checks): the row engine against mlx-lm and the study's own code, on the study's real model.

Background
  tests/test_gen.py checks the engine against mlx-lm on a tiny random model, where the two agree
  to 32-bit rounding. The real model computes in bfloat16 (about 3 digits), so two correct ways of
  computing the same score differ in the last digit. This script measures how large those
  differences are on the model the benchmark uses, and whether they change what is written.

What it measures (dense Qwen3.5-0.8B from the neural-harness study; its saved transcripts as text)
  scores      The engine re-writes the first 600 tokens of saved transcripts (forced, so both sides see
              the same text). Its scores at every step are compared with mlx-lm's scores for the same
              text read in one pass: largest difference, and how often the top token agrees.
  yardstick   The same comparison with mlx-lm on both sides: reading the text token by token through
              its own cache, against reading it in one pass. This is how much two correct ways of
              running this model differ, and what the line above should be held against.
  study's run The tokens the study's own runner wrote (a saved 8,192-token transcript) are fed to the
              engine one by one. At each step: is the token the study drew among the tokens the engine
              would keep, given the engine's own scores and its own record of what was written? A
              token the engine gives no chance at all would mean the two do not sample alike. The
              same question is asked of mlx-lm's own scores with the rule written the slow way, to
              show how often rounding alone puts a token on the other side of a cut.
  long        The same on rows forced through --long-tokens tokens (default 16,384; 32,768 is the
              benchmark's cap), compared at a few late positions with mlx-lm reading the text through
              its own cache. This covers many pages of history, which the short check does not.
  greedy      With no randomness, how many tokens the engine and mlx-lm's own generator write
              identically before the first difference, and how close the two candidates were there.
  neighbours  The same sampled row alone in an 8-slot batch, with 7 other rows in it, and with those
              7 while it sits in the last slot instead of the first.
  width       One text forced through a batch of 48 that keeps its width, and through one that narrows
              to 24 when 24 short neighbours end (the first narrowing of a 48-slot run; the row is
              moved to another slot by it). Before the narrowing the two must give the same scores
              exactly. After it they may differ as far as rounding goes (the yardstick), if a batch
              of another width is computed in another order. A larger difference would mean the
              narrowing itself loses something. Then the same with the text in the first slot, which
              the narrowing does not move: a difference there comes from the width and not the move.
  branches    The same sampled branch of one row, taken three ways in the same 8 slots: started at
              once (a slot was free), after waiting for a slot with its trunk's state held, and read
              in later from the trunk's saved tokens. The first two start from the same numbers and
              must write the same tokens. The third starts from mlx-lm's reading of the text, which
              rounds differently, so it may part ways after a while. Both of the first two go through
              the same code, so a second check does not: with no randomness a branch must write the
              tokens its trunk goes on to write, started at once and after a wait that ends while the
              trunk is still running.
  kernel      The same rows, with no randomness, through the paged kernel and through MLX's attention row
              by row: how many tokens are identical, and how close the two candidates were at the first
              difference.
  edit        A row carrying the study's edit unmerged, against the study's own adapter model.
  sampler     On real score vectors with the study's presence penalty: the tokens the engine keeps and
              their probabilities, against those of the study's own sampler code.
  draws       A sampled row is run with the study's settings while every step's scores are kept. Each
              token is then worked out again outside the engine, the slow obvious way (the penalty for
              every token written so far, a full sort, the row's own random number), and compared with
              what the engine wrote. This checks the penalty bookkeeping and the random stream in a
              real run, not only the sampling rule.
  pool        Step time with a 1 GB and an 8 GB history pool, each timed three times in turn with the
              fastest kept. They must match: a step that copied the pool instead of writing into it
              would take seconds. It is timed before the other checks, in a process that holds nothing
              else: at the end of this script, next to 5 GB of kept scores, the 8 GB pool timed 2.4 ms
              a step slower than the 1 GB one.

Run:     uv run python -m bench.engine_checks --harness ~/Documents/Dev/neural-harness [--long-tokens 32768]
Changes: writes bench/results/engine_checks.json. Reads the study's folder, never writes to it.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.generate import generate_step

from bench.common import environment, save
from lab.gen.engine import Engine, Row, seed_for
from lab.gen.sampler import kept_probabilities, row_stream

ROWS, FORCED, GREEDY, SAMPLED = 4, 600, 400, 400
BRANCH_AT = 300         # the branch check's trunk branches here: its history is then full pages and part of a buffer
NARROW_AT = 200         # the width check's 24 short rows end here, and the batch narrows from 48 slots to 24
LATE = (1024, 4096, 8192, 12288, 16383, 24576, 32767)     # written-token counts at which the long check compares scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--harness", type=Path, default=Path.home() / "Documents/Dev/neural-harness")
    ap.add_argument("--long-tokens", type=int, default=16384)
    args = ap.parse_args()
    root = args.harness.expanduser()
    sys.path.insert(0, str(root))
    from harness import sampling as study_sampling
    from harness.model import load_adapter_weights, load_arm

    model, tok = load_arm("base")
    text = model.language_model.model
    saved = [json.loads(line) for line in gzip.open(root / "runs/sd1/gen/test__base__s0.text.jsonl.gz", "rt")]
    extra, saved = saved[10:17], [r for r in saved if len(r["tokens"]) > 8000][:ROWS]
    out = {"environment": environment(), "rows": ROWS}

    def head(hidden: mx.array) -> mx.array:
        return text.embed_tokens.as_linear(hidden).astype(mx.float32)

    def same_until(a: list[int], b: list[int]) -> int:
        """How many tokens two rows share before their first difference."""
        return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))

    def reference_scores(tokens: list[int]) -> mx.array:
        """mlx-lm's scores after every position of one text, read in one pass."""
        return model(mx.array(tokens)[None])[0].astype(mx.float32)

    def parting(prompt: list[int], a: list[int], b: list[int]) -> dict:
        """Where two rows written with no randomness part, and how close the two candidates were there in mlx-lm's own scores."""
        same, gap = same_until(a, b), None
        if same < min(len(a), len(b)):
            last = reference_scores(prompt + a[:same])[-1]
            gap = abs(last[a[same]].item() - last[b[same]].item())
        return {"tokens_identical": same, "score_gap_at_first_difference": gap}

    def partings(rows: list[dict]) -> str:
        return ", ".join(f"{a['tokens_identical']}" + (f" (gap {a['score_gap_at_first_difference']:.3f})" if a["score_gap_at_first_difference"] is not None else "")
                         for a in rows)

    def run(rows: list[Row], slots: int, probe=None, pool_gb: float = 1.0, **kwargs) -> tuple[list[dict], Engine]:
        engine = Engine(model, slots=slots, pool_gb=pool_gb, max_tokens=max(r.max_tokens for r in rows), **kwargs)
        if probe is not None:
            engine.probe = lambda scores: probe(engine, scores)
        records = engine.run(rows, log=lambda line: None)
        engine.close()
        return records, engine

    def forced_scores(r: dict, **kwargs) -> mx.array:
        """The engine's scores at each of the first FORCED steps of one saved transcript, one row alone."""
        seen: list[mx.array] = []

        def probe(engine: Engine, scores: mx.array) -> None:
            seen.append(scores[0].astype(mx.float32))
            mx.eval(seen[-1])

        run([Row(prompt=r["prompt"], max_tokens=FORCED, seed=0, forced=tuple(r["tokens"][:FORCED]), **kwargs.pop("row", {}))], 1, probe, **kwargs)
        return mx.stack(seen)

    def greedy_row(r: dict, n: int) -> Row:
        return Row(prompt=r["prompt"], max_tokens=n, seed=0, temperature=0.0, top_k=1, top_p=1.0, presence_penalty=0.0, meta={"id": r["unique_id"]})

    # pool: a step must cost the same whatever the size of the pool (it is written in place, never copied).
    # Timed first: once this script holds the score arrays of the checks below, the larger pool times up to 2.4 ms slower.
    step_ms: dict[float, list[float]] = {1.0: [], 8.0: []}
    for pool_gb in (1.0, 8.0) * 3:
        engine = Engine(model, slots=8, pool_gb=pool_gb, max_tokens=300)
        t0 = time.perf_counter()
        engine.run([greedy_row(r, 300) for r in saved + extra], log=lambda line: None)
        step_ms[pool_gb].append((time.perf_counter() - t0) / engine.stats["steps"] * 1e3)
        engine = None
        mx.clear_cache()
    out["pool"] = {"step_ms_with_1_gb": min(step_ms[1.0]), "step_ms_with_8_gb": min(step_ms[8.0]),
                   "every_timing_with_1_gb": step_ms[1.0], "every_timing_with_8_gb": step_ms[8.0]}
    print(f"pool: {min(step_ms[1.0]):.1f} ms a step with a 1 GB pool, {min(step_ms[8.0]):.1f} ms with 8 GB (the fastest of 3 each)", flush=True)

    # scores: one row at a time, so step i of the run is position i of the text
    worst, same_top, compared, mine_by_row = 0.0, 0, 0, []
    for r in saved:
        got = forced_scores(r)
        ref = reference_scores(r["prompt"] + r["tokens"][:FORCED - 1])[len(r["prompt"]) - 1:]
        worst = max(worst, mx.abs(got - ref).max().item())
        same_top += int((mx.argmax(got, axis=-1) == mx.argmax(ref, axis=-1)).sum().item())
        compared += got.shape[0]
        mine_by_row.append(got)
    out["scores"] = {"positions": compared, "largest_difference": worst, "same_top_token_share": same_top / compared}
    print(f"scores: {compared:,} positions, largest difference {worst:.3f}, same top token at {same_top / compared:.2%}", flush=True)

    # yardstick: mlx-lm against itself on the same texts, token by token through its cache and in one pass
    worst, same_top = 0.0, 0
    for r in saved:
        cache = model.language_model.make_cache()
        text(mx.array(r["prompt"][:-1])[None], cache=cache)
        mx.eval([c.state for c in cache])
        steps = []
        for token in [r["prompt"][-1]] + r["tokens"][:FORCED - 1]:
            steps.append(head(text(mx.array([[token]]), cache=cache)[0, -1]))
            mx.eval(steps[-1], [c.state for c in cache])
        got = mx.stack(steps)
        ref = reference_scores(r["prompt"] + r["tokens"][:FORCED - 1])[len(r["prompt"]) - 1:]
        worst = max(worst, mx.abs(got - ref).max().item())
        same_top += int((mx.argmax(got, axis=-1) == mx.argmax(ref, axis=-1)).sum().item())
    out["yardstick"] = {"positions": compared, "largest_difference": worst, "same_top_token_share": same_top / compared}
    print(f"yardstick (mlx-lm token by token against mlx-lm in one pass): largest difference {worst:.3f}, same top token at {same_top / compared:.2%}", flush=True)

    # sampler: the engine's kept tokens and probabilities against the study's own sampler code, on real scores
    real = mine_by_row[0][::10]                                   # 60 real score vectors
    seen_tokens = sorted(set(saved[0]["tokens"][:FORCED]))
    penalty = study_sampling.PresencePenalty(1.5, real.shape[1], seed_tokens=seen_tokens)
    captured: list[mx.array] = []
    draw, pick = mx.random.categorical, [0]

    def capture(values: mx.array, axis: int = -1) -> mx.array:
        captured.append(values)
        return mx.full((values.shape[0],), pick[0], dtype=mx.uint32)

    study = study_sampling.make_sampler(0.6, 0.95, 20)
    study_sampling.mx.random.categorical = capture
    try:
        penalised = penalty(mx.array([0]), real.astype(mx.bfloat16))
        logprobs = penalised - mx.logsumexp(penalised, axis=-1, keepdims=True)
        their_tokens = []
        for k in range(20):
            pick[0] = k
            their_tokens.append(study(logprobs))
        their_values = captured[0]
    finally:
        study_sampling.mx.random.categorical = draw
    their_tokens = mx.stack(their_tokens, axis=1)
    their_probs = mx.softmax(their_values, axis=-1)
    table = mx.zeros((real.shape[0], real.shape[1]))
    table[:, mx.array(seen_tokens)] = 1.5
    n = real.shape[0]
    my_tokens, my_probs = kept_probabilities(real.astype(mx.bfloat16), table, temperature=mx.full((n,), 0.6),
                                             top_k=mx.full((n,), 20, dtype=mx.int32), top_p=mx.full((n,), 0.95))
    my_probs = my_probs / my_probs.sum(axis=-1, keepdims=True)
    out["sampler"] = {"score_vectors": n, "largest_probability_difference": mx.abs(my_probs - their_probs).max().item(),
                      "vectors_with_identical_tokens": int((my_tokens == their_tokens).all(axis=-1).sum().item())}
    print(f"sampler: {n} real score vectors, largest difference in a kept token's probability {out['sampler']['largest_probability_difference']:.2e}, "
          f"identical kept tokens in {out['sampler']['vectors_with_identical_tokens']} of {n}", flush=True)

    # draws: every token of a sampled row worked out again from the kept scores, the slow obvious way
    kept_scores: list[mx.array] = []

    def keep(engine: Engine, scores: mx.array) -> None:
        kept_scores.append(scores[0].astype(mx.float32))
        mx.eval(kept_scores[-1])

    seed = seed_for("draws", saved[1]["unique_id"])
    wrote = run([Row(prompt=saved[1]["prompt"], max_tokens=1500, seed=seed)], 1, keep)[0][0]["tokens"]
    stream = np.array(row_stream(seed, len(wrote) + 1))
    seen_so_far, same, tied = set(), 0, 0
    for i, token in enumerate(wrote):
        x = np.array(kept_scores[i])
        x[list(seen_so_far)] -= 1.5
        x = x / 0.6
        order = np.argsort(-x, kind="stable")[:20]
        probs = np.exp(x[order] - x[order[0]])
        probs /= probs.sum()
        keep_mask = (np.cumsum(probs) - probs) < 0.95
        cumulative = np.cumsum(np.where(keep_mask, probs, 0.0))
        expected = int(order[min(int((cumulative <= stream[i] * cumulative[-1]).sum()), int(keep_mask.sum()) - 1)])
        same += expected == token
        tied += expected != token and abs(x[expected] - x[token]) < 1e-6          # equal scores: either token is right
        seen_so_far.add(token)
    out["draws"] = {"tokens": len(wrote), "identical": same, "differing_only_between_equal_scores": tied}
    print(f"draws: {len(wrote):,} sampled tokens worked out again outside the engine: {same:,} identical, {tied} more differ only "
          f"between tokens with equal scores", flush=True)

    # study's run: would the engine have allowed every token the study's runner drew?
    r = saved[2]
    outside, chance = [], []

    def allowed(engine: Engine, scores: mx.array) -> None:
        live = engine.live[0]
        tokens, probs = kept_probabilities(scores[:1], engine.mask[:1], temperature=mx.array([0.6]), top_k=mx.array([20], dtype=mx.int32),
                                           top_p=mx.array([0.95]))
        p = (mx.where(tokens[0] == live.forced[0], probs[0], 0.0).sum() / probs[0].sum()).item()
        chance.append(p)
        if p == 0:
            outside.append(len(live.written))

    run([Row(prompt=r["prompt"], max_tokens=len(r["tokens"]), seed=0, forced=tuple(r["tokens"]))], 1, allowed)
    # the same question with mlx-lm's own scores and the rule written the slow way
    full, outside_ref, seen_ref = r["prompt"] + r["tokens"], 0, set()
    cache = model.language_model.make_cache()
    for start in range(0, len(full) - 1, 512):
        logits = np.array(head(text(mx.array(full[start:start + 512])[None], cache=cache)[0]))
        mx.eval([c.state for c in cache])
        for j in range(logits.shape[0]):
            k = start + j - (len(r["prompt"]) - 1)                # this position's scores are for generated token k
            if k < 0 or k >= len(r["tokens"]):
                continue
            x = logits[j].copy()
            x[list(seen_ref)] -= 1.5
            x /= 0.6
            order = np.argsort(-x, kind="stable")[:20]
            probs = np.exp(x[order] - x[order[0]])
            probs /= probs.sum()
            kept = set(order[(np.cumsum(probs) - probs) < 0.95].tolist())
            outside_ref += r["tokens"][k] not in kept
            seen_ref.add(r["tokens"][k])
    out["studys_run"] = {"tokens": len(r["tokens"]), "outside_what_the_engine_keeps": len(outside), "first_such_positions": outside[:10],
                         "outside_with_mlx_lm_scores_and_the_slow_rule": outside_ref,
                         "mean_log_chance_under_the_engine": float(np.mean(np.log(np.maximum(chance, 1e-12))))}
    print(f"study's run: of {len(r['tokens']):,} tokens the study's runner drew, {len(outside)} are outside what the engine would keep "
          f"(with mlx-lm's own scores and the rule written the slow way: {outside_ref})", flush=True)

    # long: rows forced through many pages, compared at late positions with mlx-lm reading through its own cache
    long_rows = []
    for line in gzip.open(root / "runs/edited_32k.text.jsonl.gz", "rt"):
        ids = tok.encode(json.loads(line)["text"], add_special_tokens=False)
        if len(ids) >= args.long_tokens:
            long_rows.append(ids[:args.long_tokens])
        if len(long_rows) == 2:
            break
    prompts = [r["prompt"] for r in saved[:2]]
    late = [p for p in LATE if p < args.long_tokens]
    kept: dict[tuple[int, int], mx.array] = {}

    def late_probe(engine: Engine, scores: mx.array) -> None:
        for slot, live in engine.live.items():
            if len(live.written) in late:
                kept[live.row.meta["i"], len(live.written)] = scores[slot].astype(mx.float32)
                mx.eval(kept[live.row.meta["i"], len(live.written)])

    t0 = time.perf_counter()
    run([Row(prompt=p, max_tokens=args.long_tokens, seed=0, forced=tuple(ids), meta={"i": i}) for i, (p, ids) in enumerate(zip(prompts, long_rows))],
        2, late_probe, pool_gb=1.2 * args.long_tokens / 32768)
    long_minutes = (time.perf_counter() - t0) / 60
    worst, agree = 0.0, 0
    for i, (p, ids) in enumerate(zip(prompts, long_rows)):
        full = p + ids
        cache = model.language_model.make_cache()
        for start in range(0, len(full), 512):
            hidden = text(mx.array(full[start:start + 512])[None], cache=cache)
            mx.eval(hidden, [c.state for c in cache])
            for at in late:                                        # the pass that reads token `at` of the row gives the scores after it
                position = len(p) - 1 + at
                if start <= position < start + 512:
                    ref = head(hidden[0, position - start])
                    worst = max(worst, mx.abs(kept[i, at] - ref).max().item())
                    agree += int(mx.argmax(kept[i, at]).item() == mx.argmax(ref).item())
    out["long"] = {"rows": len(long_rows), "tokens_each": args.long_tokens, "positions_compared": [int(x) for x in late],
                   "largest_difference": worst, "same_top_token": agree, "of": len(late) * len(long_rows), "minutes": long_minutes}
    print(f"long: {len(long_rows)} rows forced through {args.long_tokens:,} tokens ({long_minutes:.1f} min); at positions {late}: "
          f"largest difference {worst:.3f}, same top token in {agree} of {len(late) * len(long_rows)}", flush=True)

    # greedy: the engine against mlx-lm's own generator
    records, _ = run([greedy_row(r, GREEDY) for r in saved], ROWS)
    mine = {rec["meta"]["id"]: rec["tokens"] for rec in records}
    agreed = []
    for r in saved:
        theirs = [int(token) for token, _ in generate_step(mx.array(r["prompt"]), model, max_tokens=GREEDY, sampler=lambda x: mx.argmax(x, axis=-1))]
        agreed.append(parting(r["prompt"], mine[r["unique_id"]], theirs))
    out["greedy"] = {"tokens_each": GREEDY, "rows": agreed}
    print(f"greedy: identical for {partings(agreed)} of {GREEDY} tokens", flush=True)

    # neighbours: the same sampled row alone in 8 slots and with 7 others
    def sampled_row(r: dict) -> Row:
        return Row(prompt=r["prompt"], max_tokens=SAMPLED, seed=seed_for("checks", r["unique_id"]), meta={"id": r["unique_id"]})

    def keep_width(rows: list[Row]) -> list[dict]:
        engine = Engine(model, slots=8, pool_gb=1.0, max_tokens=SAMPLED)
        engine.narrow_when_idle = False
        return engine.run(rows, log=lambda line: None)

    def tokens_of_first(rows: list[Row]) -> list[int]:
        return next(rec["tokens"] for rec in keep_width(rows) if rec["meta"]["id"] == saved[0]["unique_id"])

    alone = tokens_of_first([sampled_row(saved[0])])
    same = same_until(alone, tokens_of_first([sampled_row(r) for r in [saved[0]] + extra]))
    same_last = same_until(alone, tokens_of_first([sampled_row(r) for r in extra + [saved[0]]]))    # rows take slots in order: it gets the last one
    out["neighbours"] = {"tokens": len(alone), "identical_until": same, "identical_until_in_the_last_slot": same_last}
    print(f"neighbours: alone and among 7 others in the same 8 slots, identical for {same} of {len(alone)} tokens; "
          f"in the last slot instead of the first, for {same_last}", flush=True)

    # width: one forced text in a batch that keeps its 48 slots and in one that narrows to 24 when half its rows end
    def scores_by_step(narrow: bool, forced_first: bool = False) -> tuple[mx.array, int]:
        seen: list[mx.array] = []

        def probe(engine: Engine, scores: mx.array) -> None:
            slot = next(slot for slot, live in engine.live.items() if live.row.forced)
            seen.append(scores[slot].astype(mx.float32))
            mx.eval(seen[-1])

        filler = lambda i, n: Row(prompt=extra[i % len(extra)]["prompt"], max_tokens=n, seed=seed_for("checks", "filler", i))
        forced = Row(prompt=saved[0]["prompt"], max_tokens=FORCED, seed=0, forced=tuple(saved[0]["tokens"][:FORCED]))
        engine = Engine(model, slots=48, pool_gb=1.0, max_tokens=FORCED)
        engine.narrow_when_idle = narrow
        engine.probe = lambda scores: probe(engine, scores)
        # Rows take slots in order: the 24 short rows fill slots below 24, so the narrowing moves every long row behind them. The
        # forced row is last (moved from slot 47 to 23) or first (it stays in slot 0).
        others = [filler(i, NARROW_AT) for i in range(24)] + [filler(i, FORCED) for i in range(24, 47)]
        engine.run([forced] + others if forced_first else others + [forced], log=lambda line: None)
        return mx.stack(seen), engine.slots

    def width_gap(forced_first: bool) -> dict:
        kept_wide, slots_wide = scores_by_step(False, forced_first)
        narrowed, slots_narrow = scores_by_step(True, forced_first)
        assert (slots_wide, slots_narrow) == (48, 24)
        gap = mx.abs(kept_wide - narrowed).max(axis=-1)
        after = (mx.argmax(kept_wide, axis=-1) == mx.argmax(narrowed, axis=-1))[NARROW_AT:]
        return {"largest_difference_before": gap[:NARROW_AT].max().item(), "largest_difference_after": gap[NARROW_AT:].max().item(),
                "same_top_token_share_after": after.astype(mx.float32).mean().item()}

    out["width"] = {"slots": [48, 24], "narrows_after_step": NARROW_AT, "steps": FORCED, **width_gap(False), "row_not_moved": width_gap(True)}
    for name, w in (("the row moved to another slot", out["width"]), ("the row left in its slot", out["width"]["row_not_moved"])):
        print(f"width: 48 slots throughout against 48 narrowing to 24 after step {NARROW_AT}, {name}: largest difference before "
              f"{w['largest_difference_before']:.3f}, after {w['largest_difference_after']:.3f}, same top token after at "
              f"{w['same_top_token_share_after']:.2%}", flush=True)

    # branches: one branch started at once, after waiting for a slot, and read in from saved tokens
    def branch_tokens(others: int, max_waiting: int) -> tuple[list[int], dict]:
        engine = Engine(model, slots=8, pool_gb=1.0, max_tokens=SAMPLED)
        engine.narrow_when_idle = False
        engine.max_waiting = max_waiting
        trunk = Row(prompt=saved[0]["prompt"], max_tokens=SAMPLED, seed=seed_for("checks", "trunk"), branch_at=(BRANCH_AT,))
        records = engine.run([trunk] + [sampled_row(r) for r in extra[:others]], log=lambda line: None,
                             brancher=lambda row, written: Row(prompt=row.prompt + written, max_tokens=SAMPLED, seed=seed_for("checks", "branch"),
                                                               seen=tuple(written), meta={"branch": True}))
        return next(rec["tokens"] for rec in records if rec["meta"].get("branch")), engine.stats

    at_once, stats = branch_tokens(6, 8)
    assert stats["branches_shared"] == 1 and stats["branches_waited"] == 0
    waited, stats = branch_tokens(7, 8)
    assert stats["branches_shared"] == 1 and stats["branches_waited"] == 1
    read_in, stats = branch_tokens(7, 0)
    assert stats["branches_read"] == 1
    out["branches"] = {"branch_at": BRANCH_AT, "tokens": len(at_once), "waited_identical_until": same_until(at_once, waited),
                       "read_in_identical_until": same_until(at_once, read_in)}
    print(f"branches: a branch at token {BRANCH_AT} that waited for a slot writes the same {same_until(at_once, waited)} of {len(at_once)} tokens "
          f"as one started at once; read in from saved tokens, the same {same_until(at_once, read_in)}", flush=True)

    def continues(others: list[Row]) -> tuple[int, dict]:
        """A trunk and its branch at BRANCH_AT, both written with no randomness: how many of the branch's tokens are the trunk's own next ones."""
        engine = Engine(model, slots=8, pool_gb=1.0, max_tokens=BRANCH_AT + GREEDY)
        engine.narrow_when_idle = False
        trunk = replace(greedy_row(saved[0], BRANCH_AT + GREEDY), branch_at=(BRANCH_AT,))
        records = engine.run([trunk] + others, log=lambda line: None,
                             brancher=lambda row, written: replace(row, prompt=row.prompt + written, max_tokens=GREEDY // 2, branch_at=(), meta={"branch": True}))
        branch = next(rec["tokens"] for rec in records if rec["meta"].get("branch"))
        own = next(rec["tokens"] for rec in records if rec["meta"].get("id") == saved[0]["unique_id"])
        return same_until(branch, own[BRANCH_AT:]), engine.stats

    same_at_once, stats = continues([greedy_row(r, BRANCH_AT + GREEDY) for r in extra[:6]])
    assert stats["branches_shared"] == 1 and stats["branches_waited"] == 0
    same_waited, stats = continues([greedy_row(r, BRANCH_AT + 60) for r in extra])       # they end 60 steps after the branch point, the trunk runs on
    assert stats["branches_shared"] == 1 and stats["branches_waited"] == 1
    out["branches"] |= {"tokens_of_the_trunk_continued": GREEDY // 2, "continues_its_trunk_at_once": same_at_once, "continues_its_trunk_after_a_wait": same_waited}
    print(f"branches: with no randomness a branch writes its trunk's own next {GREEDY // 2} tokens for {same_at_once} of them when started at once, "
          f"and for {same_waited} after a wait that ended while the trunk was running", flush=True)

    # kernel: paged kernel against MLX attention row by row
    plain, _ = run([greedy_row(r, GREEDY) for r in saved], ROWS, use_kernel=False)
    plain = {rec["meta"]["id"]: rec["tokens"] for rec in plain}
    kernel_same = [parting(r["prompt"], mine[r["unique_id"]], plain[r["unique_id"]]) for r in saved]
    out["kernel"] = {"tokens_each": GREEDY, "rows": kernel_same}
    print(f"kernel: paged kernel and MLX's attention write the same tokens for {partings(kernel_same)} of {GREEDY}", flush=True)

    # edit: a row with the study's edit unmerged, against the study's own adapter model
    r = saved[0]
    got = forced_scores(r, row={"edit": "astra", "edit_scale": 1.0}, edits={"astra": load_adapter_weights()})
    base_ref = reference_scores(r["prompt"] + r["tokens"][:FORCED - 1])[len(r["prompt"]) - 1:]
    adapter_model, _ = load_arm("adapter@1.0")
    adapter_ref = adapter_model(mx.array(r["prompt"] + r["tokens"][:FORCED - 1])[None])[0].astype(mx.float32)[len(r["prompt"]) - 1:]
    out["edit"] = {"positions": FORCED, "largest_difference_from_the_studys_adapter": mx.abs(got - adapter_ref).max().item(),
                   "same_top_token_share": (mx.argmax(got, axis=-1) == mx.argmax(adapter_ref, axis=-1)).astype(mx.float32).mean().item(),
                   "largest_difference_between_adapter_and_base": mx.abs(adapter_ref - base_ref).max().item()}
    print(f"edit: largest difference from the study's adapter model {out['edit']['largest_difference_from_the_studys_adapter']:.3f} "
          f"(the edit itself moves scores by up to {out['edit']['largest_difference_between_adapter_and_base']:.2f}); "
          f"same top token at {out['edit']['same_top_token_share']:.2%}", flush=True)
    save("engine_checks", f"qwen3.5-0.8b, long check to {args.long_tokens:,}", out)


if __name__ == "__main__":
    main()
