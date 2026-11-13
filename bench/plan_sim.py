"""Planner for row-engine runs: replay saved row lengths through the engine's scheduling rules, without a model.

Why
  A 100-problem timing run takes one to two hours, so settings (rows at once, slots kept free for
  branches, pool size) cannot be tried one by one, and a run ten times larger cannot be tried at
  all before it is started. This script answers "how long would it take" on the CPU in a second.

What it does
  1. Reads the rows of finished timing runs from their ledgers (runs/phase1/<tag>/): how many
     tokens each trunk wrote, where it branched into a forced exit and how long each branch was.
  2. Steps through lab/gen/engine.py's rules for those rows: slots, slots kept free, branches
     that start at once, wait for a slot or are read in later, the page pool, rows set aside and
     replayed, the batch narrowing.
  3. Adds up time from three measured costs: a step's fixed part by batch width and the part
     per token of history read (bench/results/long_step.json), and the time to read a saved row
     back in (SPEED-RESULTS.md section 19).
  4. Prints its answer next to the wall-clock of every timing run that was really made, then
     the settings and sizes that were not run.
  It is a model: it knows nothing about memory pressure from other programs, and its answers
  for runs that were never made are estimates.

Run:     uv run python -m bench.plan_sim        (CPU only, about a second)
Changes: writes bench/results/plan_sim.json. Reads runs/phase1/ and bench/results/phase1.json.
"""

from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from bench.common import RESULTS, save
from lab.gen.engine import GROWTH_PAGES, WIDTHS
from lab.gen.kv import PAGE
from lab.gen.ledger import Ledger

ROOT = Path(__file__).resolve().parent.parent
STEP_MS = ((16, 23.6), (48, 33.0), (96, 53.7))      # one engine step at a short history, by batch width (bench.long_step)
HISTORY_NS = 51.0                                    # per token of history, per row, per step (the 32,768-cap timing run)
READ_S = ((4096, 0.84), (8192, 1.80), (16384, 4.13), (24576, 6.93))    # seconds to read one saved row of that length back in
Rows = list[tuple[int, int, dict[int, int]]]         # per trunk: prompt tokens, written tokens, {exit point: branch length}


def along(points: tuple[tuple[int, float], ...], x: float) -> float:
    """Straight lines through measured points, carried on past both ends."""
    pairs = list(zip(points, points[1:]))
    (x0, y0), (x1, y1) = next((pair for pair in pairs if x <= pair[1][0]), pairs[-1])
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


@dataclass
class Sim:
    """One row of the replay: a trunk, or a branch of one."""

    left: int                                         # tokens still to write
    history: int                                      # tokens it reads at every step (its own and its trunk's)
    own: int                                          # tokens whose pages only this row holds
    trunk: "Sim | None" = None                        # set while a branch shares its trunk's pages
    branches: dict[int, int] = field(default_factory=dict)
    written: int = 0
    started: bool = False
    alive: bool = False


def load(tag: str, exits_from: str | None = None, cap: int | None = None, answer: int | None = None) -> Rows:
    """The trunks of a ledger in the order the run started them, with the forced-exit branches of `exits_from`."""
    ledger = Ledger(ROOT / "runs/phase1" / tag)
    meta = ledger.meta()
    order = {index: at for at, index in enumerate(random.Random(meta["subset_seed"]).sample(range(500), meta["n"]))}
    place = lambda r: (order[r["meta"]["index"]], r["meta"]["sample"])
    trunks = sorted((r for r in ledger.records() if r["meta"]["kind"] == "trunk"), key=place)
    exits: dict[tuple, dict[int, int]] = {}
    for r in Ledger(ROOT / "runs/phase1" / (exits_from or tag)).records():
        if r["meta"]["kind"] == "exit":
            exits.setdefault(place(r), {})[r["meta"]["exit_at"]] = min(r["written"], answer) if answer else r["written"]
    rows = []
    for r in trunks:
        written = min(r["written"], cap) if cap else r["written"]
        rows.append((r["prompt_tokens"], written, {b: n for b, n in exits.get(place(r), {}).items() if b < written}))
    return rows


def simulate(rows: Rows, *, slots: int, pool_pages: int, keep_free: int = 0, waiting: bool = True, exits: bool = True,
             only_exits: bool = False) -> dict:
    """Minutes, steps and counts for one run.

    waiting=False is the engine before branches could wait: no free slot meant reading the branch in later.
    only_exits: the trunks are already in the ledger and every branch is read in from their saved tokens.
    """
    queue: deque[Sim] = deque()
    for prompt, written, branches in rows:
        if only_exits:
            queue.extend(Sim(left=n, history=prompt + b, own=prompt + b, started=True) for b, n in sorted(branches.items()))
        else:
            queue.append(Sim(left=written, history=prompt, own=prompt, branches=dict(branches) if exits else {}))
    live: list[Sim] = []
    held: deque[Sim] = deque()
    width, ms, steps, read_s = slots, 0.0, 0, 0.0
    count = {"set_aside": 0, "branches_sharing": 0, "branches_that_waited": 0, "branches_read_in": 0}

    def pages_used() -> int:
        # A trunk that has ended keeps the pages its remaining branches share, up to the furthest of them.
        kept: dict[int, int] = {}
        for s in [*live, *held]:
            if s.trunk is not None and not s.trunk.alive:
                kept[id(s.trunk)] = max(kept.get(id(s.trunk), 0), s.history - s.own)
        return sum(-(-s.own // PAGE) for s in live) + sum(-(-tokens // PAGE) for tokens in kept.values())

    def room_for(tokens: int) -> bool:
        return pool_pages - pages_used() - tokens // PAGE >= GROWTH_PAGES * (len(live) + 1)

    def start(s: Sim) -> None:
        s.alive = s.started = True
        live.append(s)

    def read_in_later(s: Sim, at_front: bool = False) -> None:
        s.own, s.trunk, s.alive = s.history, None, False
        queue.appendleft(s) if at_front else queue.append(s)

    while queue or live or held:
        while held and len(live) < width and (room_for(0) or not live):
            start(held.popleft())
        while queue and width - len(live) > (0 if queue[0].started else keep_free) and room_for(queue[0].own):
            s = queue.popleft()
            if s.started:
                read_s += max(0.0, along(READ_S, s.own))
            start(s)
        if not live:
            raise SystemExit("the pool is too small for the next row")
        while pages_used() > pool_pages and len(live) > 1:
            orphan = next((s for s in reversed(held) if not s.trunk.alive), None)
            if orphan is not None:                    # a waiting branch whose trunk is gone gives its pages up
                held.remove(orphan)
                read_in_later(orphan)
                count["branches_sharing"] -= 1
                count["branches_read_in"] += 1
                continue
            youngest = min(live, key=lambda s: s.history)
            live.remove(youngest)
            read_in_later(youngest, at_front=True)
            count["set_aside"] += 1
        # Run to the next thing that happens: a row ends, a branch point, or the next look at the pool.
        d = min(min(s.left for s in live), min((b - s.written for s in live for b in s.branches), default=PAGE), PAGE)
        ms += d * along(STEP_MS, width) + HISTORY_NS * 1e-6 * sum(s.history * d + d * (d - 1) / 2 for s in live)
        steps += d
        for s in list(live):
            s.left, s.history, s.own, s.written = s.left - d, s.history + d, s.own + d, s.written + d
            if s.left == 0:
                live.remove(s)
                s.alive = False
            elif s.written in s.branches:
                child = Sim(left=s.branches.pop(s.written), history=s.history, own=0, trunk=s, started=True)
                if len(live) < width:
                    start(child)
                    count["branches_sharing"] += 1
                elif waiting and len(held) < slots:
                    held.append(child)
                    count["branches_sharing"] += 1
                    count["branches_that_waited"] += 1
                else:
                    read_in_later(child)
                    count["branches_read_in"] += 1
        if not queue and live:
            wanted = len(live) + len(held) + (min(len(live), keep_free) if exits else 0)
            narrower = next((w for w in WIDTHS if w >= wanted), width)
            if narrower <= width // 2:
                width = narrower
    return {"minutes": (ms / 1000 + read_s) / 60, "steps": steps, "minutes_reading_back_in": read_s / 60, **count}


def main() -> None:
    measured = json.loads((RESULTS / "phase1.json").read_text())
    gb = lambda n: int(n * 2**30 // (2 * 6 * 2 * PAGE * 256 * 2))        # pool pages for the 0.8B model: 6 layers, 2 heads, 256 numbers, 16-bit
    edited, base = load("edited-32k", "edited-32k-exits"), load("base-32k")
    out: dict[str, dict] = {}

    def show(name: str, result: dict, was: str | None = None) -> None:
        if was:
            result = {**result, "measured_minutes": measured[was]["wall_s"] / 60, "measured_steps": measured[was]["steps"]}
        out[name] = result
        print(f"  {name:62s} {result['minutes']:6.1f} min  {result['steps']:9,} steps  set aside {result['set_aside']:3d}  branches sharing "
              f"{result['branches_sharing']:4d} (waited {result['branches_that_waited']:4d}), read in {result['branches_read_in']:3d}"
              + (f"   measured: {result['measured_minutes']:.1f} min, {result['measured_steps']:,} steps" if was else ""))

    print("Against the runs that were made (branches could not wait yet):")
    show("edited trunks, 48 slots, 12 GB", simulate(edited, slots=48, pool_pages=gb(12), exits=False), "edited-32k")
    show("edited exits read in afterwards, 48 slots, 10 GB", simulate(edited, slots=48, pool_pages=gb(10), only_exits=True), "edited-32k-exits")
    show("base trunks and exits, 32 slots, 10 GB", simulate(base, slots=32, pool_pages=gb(10), waiting=False), "base-32k")
    for tag in ("edited-8k-seed1", "edited-8k-seed2"):
        show(f"{tag}, 48 slots, 6 GB", simulate(load(tag), slots=48, pool_pages=gb(6), exits=False), tag)
    for tag in sorted(set(measured) - {"edited-32k", "edited-32k-exits", "base-32k", "edited-8k-seed1", "edited-8k-seed2"}):
        m = measured[tag]
        if (ROOT / "runs/phase1" / tag).exists() and not m["continued_an_earlier_run"]:
            show(f"{tag}, {m['slots']} slots, {m['pool_gb']:g} GB, keep {m['keep_free']}",
                 simulate(load(tag), slots=m["slots"], pool_pages=gb(m["pool_gb"]), keep_free=m["keep_free"], exits=bool(m["exits"])), tag)

    print("Trunks and exits in one run, with branches that wait (estimates):")
    for slots, keep, pool in ((48, 0, 10), (48, 8, 10), (48, 16, 10), (56, 8, 11), (64, 0, 11)):
        show(f"edited, {slots} slots, {keep} kept free, {pool} GB", simulate(edited, slots=slots, pool_pages=gb(pool), keep_free=keep))
    for slots, keep, pool in ((32, 0, 10), (40, 8, 10), (48, 0, 10), (48, 12, 10), (48, 16, 10), (56, 24, 10)):
        show(f"base, {slots} slots, {keep} kept free, {pool} GB", simulate(base, slots=slots, pool_pages=gb(pool), keep_free=keep))

    def tenfold(name: str, rows: Rows, slots: int, keep: int, pool: int) -> None:
        result = simulate(rows * 10, slots=slots, pool_pages=gb(pool), keep_free=keep)
        show(f"{name} x10, {slots} slots, {keep} kept free, {pool} GB: {result['minutes'] / 10:.1f} per 100", result)

    print("Ten times the rows, as one run (estimates; minutes per 100 problems in the name):")
    tenfold("edited", edited, 48, 0, 10)
    tenfold("edited", edited, 64, 0, 11)
    tenfold("base", base, 48, 12, 10)
    tenfold("base", base, 48, 16, 10)
    print("The two cheaper choices of docs/phase1-plan.md, ten times the rows (estimates):")
    tenfold("edited, answers of 2,000 tokens", load("edited-32k", "edited-32k-exits", answer=2025), 48, 0, 10)
    tenfold("base, answers of 2,000 tokens", load("base-32k", answer=2025), 48, 16, 10)
    tenfold("base, answers of 2,000 and trunks stopped at 24,577", load("base-32k", cap=24577, answer=2025), 48, 16, 10)
    save("plan_sim", "replay", {"step_ms_by_width": STEP_MS, "history_ns_per_token": HISTORY_NS, "seconds_to_read_a_row_back": READ_S, "runs": out})


if __name__ == "__main__":
    main()
