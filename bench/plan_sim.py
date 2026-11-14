"""Planner for row-engine runs: replay saved row lengths through the engine's scheduling rules, without a model.

Why
  A 100-problem timing run takes one to two hours, so settings (rows at once, slots kept free for
  branches, pool size) cannot be tried one by one, and a run ten times larger cannot be tried at
  all before it is started. This script answers "how long would it take" on the CPU in seconds.

What it does
  1. Reads the rows of finished timing runs from their ledgers (runs/phase1/<tag>/): how many
     tokens each trunk wrote, where it branched into a forced exit and how long each branch was.
  2. Steps through lab/gen/engine.py's rules for those rows: slots, slots kept free, branches
     that start at once, wait for a slot or are read in later, the page pool, rows set aside and
     replayed, the batch narrowing (or not). The pool is counted in whole pages and looked at
     before every step, as the engine does, so the replay of a finished run must give that run's
     steps, rows set aside, branch counts and tokens read back in exactly.
  3. Adds up time from three measured costs: a step's fixed part by batch width and the part
     per token of history read (bench/results/long_step.json), and the time to read a saved row
     back in (SPEED-RESULTS.md section 19).
  4. Prints its answer next to every timing run that was really made (the counts, and the
     minutes, which are the only part that is a model), then the settings and sizes not run.
  It is a model: it knows nothing about memory pressure from other programs, and its answers
  for runs that were never made are estimates.

Run:     uv run python -m bench.plan_sim        (CPU only, about 5 seconds)
Changes: writes bench/results/plan_sim.json. Reads bench/results/phase1.json and the ledgers under
         runs/phase1/, which are not in the repository: it runs where the timing runs were made.
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
STEP_MS = ((16, 23.6), (48, 33.0), (96, 53.7))      # one engine step at a short history, by batch width (bench.long_step).
                                                     # Between these it is a straight line: no width from 49 to 95 was measured.
HISTORY_NS = 45.2                                    # per token of history, per row, per step: the second 32,768-cap draw, minutes 9 to 32,
                                                     # all 48 slots busy (the first draw gave 51 over a stretch in which the Mac was also used by something else)
READ_S = ((4096, 0.84), (8192, 1.80), (16384, 4.13), (24576, 6.93))    # seconds to read one saved row of that length back in
Rows = list[tuple[int, int, dict[int, int]]]         # per trunk: prompt tokens, written tokens, {exit point: branch length}


def along(points: tuple[tuple[int, float], ...], x: float) -> float:
    """Straight lines through measured points, carried on past both ends."""
    pairs = list(zip(points, points[1:]))
    (x0, y0), (x1, y1) = next((pair for pair in pairs if x <= pair[1][0]), pairs[-1])
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


@dataclass(eq=False)
class Sim:
    """One row of the replay: a trunk, or a branch of one."""

    left: int                                         # tokens still to write
    history: int                                      # tokens it has in the pool (its own and, for a branch, its trunk's)
    trunk: "Sim | None" = None                        # a branch's trunk
    shared: int = 0                                   # how many of its pages are its trunk's, while it shares them
    lease: list[bool] | None = None                   # the trunk's `hold` when the branch was taken
    hold: list[bool] = field(default_factory=lambda: [False])    # [True] while it runs and has pages; a new one at every start
    branches: dict[int, int] = field(default_factory=dict)
    written: int = 0
    started: bool = False
    slot: int = -1


def pool_pages(gb: float) -> int:
    """The history pool's pages in that many GB, for the 0.8B model: 6 layers, 2 heads, 256 numbers, 16-bit."""
    return int(gb * 2**30 // (2 * 6 * 2 * PAGE * 256 * 2))


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
             only_exits: bool = False, narrow: bool = True, waiting_places: int | None = None) -> dict:
    """Minutes, steps and counts for one run.

    waiting=False is the engine before branches could wait: no free slot meant reading the branch in later.
    only_exits: the trunks are already in the ledger and every branch is read in from their saved tokens.
    narrow=False keeps the batch at full width to the end (bench.phase1 --keep-width).
    waiting_places: how many branches may wait at once (the engine's max_waiting; one per slot if not given).
    """
    places = slots if waiting_places is None else waiting_places
    queue: deque[Sim] = deque()
    for prompt, written, branches in rows:            # the pool holds a row's text less its newest token, which the next step reads
        if only_exits:
            queue.extend(Sim(left=n, history=prompt + b - 1, started=True) for b, n in sorted(branches.items()))
        else:
            queue.append(Sim(left=written, history=prompt - 1, branches=dict(branches) if exits else {}))
    live: list[Sim] = []                              # in the order the engine goes through its rows
    free_slots = list(range(slots - 1, -1, -1))
    held: deque[Sim] = deque()
    width, ms, steps, read_s = slots, 0.0, 0, 0.0
    count = {"set_aside": 0, "branches_sharing": 0, "branches_that_waited": 0, "branches_read_in": len(queue) if only_exits else 0,
             "tokens_read": 0, "narrowed": 0}

    def pages_free() -> int:
        # A row holds whole pages only: its newest tokens sit in a buffer outside the pool. A branch's first `shared` pages
        # are its trunk's. They cost nothing while the trunk holds them, and stay behind for its branches once it does not
        # (it has ended, or was set aside: read in again it gets new pages).
        left_behind: dict[int, int] = {}
        for s in (*live, *held):
            if s.lease is not None and not s.lease[0]:
                left_behind[id(s.lease)] = max(left_behind.get(id(s.lease), 0), s.shared)
        return pool_pages - sum(s.history // PAGE - s.shared for s in live) - sum(left_behind.values())

    def room_for(tokens: int) -> bool:
        return pages_free() - tokens // PAGE >= GROWTH_PAGES * (len(live) + 1)

    def start(s: Sim) -> None:
        s.started, s.hold, s.slot = True, [True], free_slots.pop()
        live.append(s)

    def leave(s: Sim) -> None:
        live.remove(s)
        free_slots.append(s.slot)
        s.hold[0] = False

    def narrower() -> int:
        wanted = len(live) + len(held) + (min(len(live), keep_free) if exits else 0)
        return next((w for w in WIDTHS if w >= wanted), width)

    def read_in_later(s: Sim, at_front: bool = False) -> None:
        s.shared, s.lease = 0, None
        queue.appendleft(s) if at_front else queue.append(s)

    while queue or live or held:
        while held and len(live) < width and (room_for(0) or not live):
            start(held.popleft())
        while queue and width - len(live) > (0 if queue[0].started else keep_free) and room_for(queue[0].history + 1):
            s = queue.popleft()
            count["tokens_read"] += s.history         # the engine reads every row it lets in: its prompt, or all it had written
            if s.started:
                read_s += max(0.0, along(READ_S, s.history + 1))
            start(s)
        if not live:
            raise SystemExit("the pool is too small for the next row")
        # The engine's check before every step: a page for each row about to fill its buffer, or make room.
        made_room = False
        while pages_free() < sum(1 for s in live if s.history % PAGE == PAGE - 1):
            made_room = True
            orphan = next((s for s in reversed(held) if not s.trunk.hold[0]), None)
            if orphan is not None:                    # a waiting branch whose trunk is not running gives its pages up
                held.remove(orphan)
                read_in_later(orphan)
                count["branches_sharing"] -= 1
                count["branches_read_in"] += 1
                continue
            if len(live) == 1:
                raise SystemExit("the pool is too small for a single row")
            youngest = min(live, key=lambda s: s.history)
            leave(youngest)
            read_in_later(youngest, at_front=True)
            count["set_aside"] += 1
        # Run to the next thing that happens: a row ends, a branch point, or a step the pool has no page for.
        # After making room only one step: the slot and pages just freed may let a row in at the very next one.
        d = 1 if made_room else min(min(s.left for s in live), min((b - s.written for s in live for b in s.branches), default=PAGE), PAGE)
        free = pages_free()
        if free < len(live):                          # the step at which one row more fills its buffer than the pool has pages
            d = min(d, sorted(PAGE - 1 - s.history % PAGE for s in live)[free])
        if narrow and not queue and narrower() <= width // 2:
            d = 1                                     # the engine narrows after the very next step
        ms += d * along(STEP_MS, width) + HISTORY_NS * 1e-6 * sum((s.history + 1) * d + d * (d - 1) / 2 for s in live)
        steps += d
        for s in list(live):
            s.left, s.history, s.written = s.left - d, s.history + d, s.written + d
            if s.left == 0:
                leave(s)
            elif s.written in s.branches:
                child = Sim(left=s.branches.pop(s.written), history=s.history, trunk=s, shared=s.history // PAGE, lease=s.hold, started=True)
                if len(live) < width:
                    start(child)
                    count["branches_sharing"] += 1
                elif waiting and len(held) < places:
                    held.append(child)
                    count["branches_sharing"] += 1
                    count["branches_that_waited"] += 1
                else:
                    read_in_later(child)
                    count["branches_read_in"] += 1
        if narrow and not queue and live and narrower() <= width // 2:
            # As the engine does it: rows in the slots given up move into the lowest free ones and go to the end of its order.
            width = narrower()
            count["narrowed"] += 1
            moved = sorted((s for s in live if s.slot >= width), key=lambda s: s.slot)
            taken = {s.slot for s in live}
            empty = [x for x in range(width) if x not in taken]
            for s, slot in zip(moved, empty):
                s.slot = slot
            live[:] = [s for s in live if s not in moved] + moved
            free_slots[:] = empty[len(moved):][::-1]
    return {"minutes": (ms / 1000 + read_s) / 60, "steps": steps, "minutes_reading_back_in": read_s / 60, **count}


def main() -> None:
    measured = json.loads((RESULTS / "phase1.json").read_text())
    gb = pool_pages
    edited, base = load("edited-32k", "edited-32k-exits"), load("base-32k")
    out: dict[str, dict] = {}

    def show(name: str, result: dict, was: str | None = None) -> None:
        counts = lambda r: (f"{r['steps']:9,} steps  set aside {r['set_aside']:3d}  branches sharing {r['branches_sharing']:4d} "
                            f"(waited {r['branches_that_waited']:4d}), read in {r['branches_read_in']:3d}  tokens read {r['tokens_read']:10,}")
        print(f"  {name:62s} {result['minutes']:6.1f} min  {counts(result)}")
        if was:
            m = measured[was]
            real = {"minutes": m["wall_s"] / 60, "steps": m["steps"], "set_aside": m["rows_set_aside"], "branches_sharing": m["branches_sharing_history"],
                    "branches_that_waited": m.get("branches_that_waited_for_a_slot", 0), "branches_read_in": m["branches_read_in"],
                    "tokens_read": m["tokens_read"]}
            same = all(result[key] == real[key] for key in real if key != "minutes")
            print(f"  {'  measured' + ('' if same else ' (the replay differs)'):62s} {real['minutes']:6.1f} min  {counts(real)}")
            result = {**result, "measured": real, "same_steps_and_counts": same}
        out[name] = result

    print("Against the runs made before branches could wait:")
    show("edited trunks, 48 slots, 12 GB", simulate(edited, slots=48, pool_pages=gb(12), exits=False), "edited-32k")
    show("edited exits read in afterwards, 48 slots, 10 GB", simulate(edited, slots=48, pool_pages=gb(10), only_exits=True), "edited-32k-exits")
    show("base trunks and exits, 32 slots, 10 GB", simulate(base, slots=32, pool_pages=gb(10), waiting=False), "base-32k")
    for tag in ("edited-8k-seed1", "edited-8k-seed2"):
        show(f"{tag}, 48 slots, 6 GB", simulate(load(tag), slots=48, pool_pages=gb(6), exits=False), tag)
    print("Against every other run in bench/results/phase1.json, each replayed from its own rows:")
    for tag in sorted(set(measured) - {"edited-32k", "edited-32k-exits", "base-32k", "edited-8k-seed1", "edited-8k-seed2"}):
        m = measured[tag]
        if (ROOT / "runs/phase1" / tag).exists() and not m["continued_an_earlier_run"]:
            show(f"{tag}, {m['slots']} slots, {m['pool_gb']:g} GB, keep {m['keep_free']}",
                 simulate(load(tag), slots=m["slots"], pool_pages=gb(m["pool_gb"]), keep_free=m["keep_free"], exits=bool(m["exits"]),
                          narrow=not m.get("kept_width", False)), tag)

    print("Trunks and exits in one run, with branches that wait (estimates):")
    for slots, keep, pool in ((48, 0, 10), (48, 8, 10), (48, 16, 10), (56, 8, 11), (64, 0, 11)):
        show(f"edited, {slots} slots, {keep} kept free, {pool} GB", simulate(edited, slots=slots, pool_pages=gb(pool), keep_free=keep))
    for slots, keep, pool in ((32, 0, 10), (40, 8, 10), (48, 0, 10), (48, 12, 10), (48, 16, 10), (56, 24, 10)):
        show(f"base, {slots} slots, {keep} kept free, {pool} GB", simulate(base, slots=slots, pool_pages=gb(pool), keep_free=keep))
    if (ROOT / "runs/phase1/base-32k-pack").exists():
        show("base (the 48-slot run's own rows), 48 slots, 12 kept free, 10 GB", simulate(load("base-32k-pack"), slots=48, pool_pages=gb(10), keep_free=12))

    def tenfold(name: str, rows: Rows, slots: int, keep: int, pool: int, **options) -> None:
        result = simulate(rows * 10, slots=slots, pool_pages=gb(pool), keep_free=keep, **options)
        show(f"{name} x10, {slots} slots, {keep} kept free, {pool} GB: {result['minutes'] / 10:.1f} per 100", result)

    print("Ten times the rows, as one run (estimates; minutes per 100 problems in the name):")
    tenfold("edited", edited, 48, 0, 10)
    tenfold("edited", edited, 64, 0, 11)
    tenfold("base", base, 48, 12, 10)
    tenfold("base", base, 48, 16, 10)
    print("The batch kept at full width to the end (bench.phase1 --keep-width; estimates, to hold against the same lines above):")
    show("edited rows alone, 48 slots, 12 GB, full width", simulate(edited, slots=48, pool_pages=gb(12), exits=False, narrow=False))
    show("edited, 48 slots, 0 kept free, 10 GB, full width", simulate(edited, slots=48, pool_pages=gb(10), narrow=False))
    show("base, 48 slots, 16 kept free, 10 GB, full width", simulate(base, slots=48, pool_pages=gb(10), keep_free=16, narrow=False))
    tenfold("edited, full width", edited, 48, 0, 10, narrow=False)
    tenfold("base, full width", base, 48, 16, 10, narrow=False)
    print("Twice as many branches allowed to wait (the engine's max_waiting at 96; estimates, to hold against the same lines above):")
    show("edited, 48 slots, 0 kept free, 10 GB, 96 waiting", simulate(edited, slots=48, pool_pages=gb(10), waiting_places=96))
    tenfold("edited, 96 waiting", edited, 48, 0, 10, waiting_places=96)
    print("The two cheaper choices of docs/phase1-plan.md, ten times the rows (estimates):")
    tenfold("edited, answers of 2,000 tokens", load("edited-32k", "edited-32k-exits", answer=2025), 48, 0, 10)
    tenfold("base, answers of 2,000 tokens", load("base-32k", answer=2025), 48, 16, 10)
    tenfold("base, answers of 2,000 and trunks stopped at 24,577", load("base-32k", cap=24577, answer=2025), 48, 16, 10)
    save("plan_sim", "replay", {"step_ms_by_width": STEP_MS, "history_ns_per_token": HISTORY_NS, "seconds_to_read_a_row_back": READ_S, "runs": out})


if __name__ == "__main__":
    main()
