"""The row engine (lab/gen): sampler, paged attention, the engine against mlx-lm, branches, replay, edits, ledger.

Every engine test uses a tiny random model of the Qwen3.5 shape, built in memory (no download).
The reference is always mlx-lm's own model run the plain way: the whole text in one pass, no cache.

Set LAB_TEST_DEVICE=cpu to run this file without the GPU (the attention kernel is then replaced by
its plain-MLX twin; test_paged_attention_kernel is skipped).
"""

from __future__ import annotations

import os
import random

import mlx.core as mx
import numpy as np
import pytest

from bench import plan_sim
from lab.gen import engine as engine_module
from lab.gen.engine import GB, Engine, Row, seed_for
from lab.gen.kv import PAGE, PagePool
from lab.gen.ledger import Ledger
from lab.gen.loops import LoopWatch, audited, stop_point
from lab.gen.sampler import kept_probabilities, next_tokens, row_stream, top_values

ON_CPU = os.environ.get("LAB_TEST_DEVICE") == "cpu"
VOCAB = 600
PAGE_BYTES = 2 * 2 * 2 * PAGE * 32 * 4     # keys and values, 2 attention layers, 2 heads, 256 tokens, 32 numbers, 4 bytes


@pytest.fixture(autouse=True, scope="module")
def device():
    if ON_CPU:
        mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(mx.gpu)


def tiny_model(seed: int = 0):
    from mlx_lm.models import qwen3_5

    mx.random.seed(seed)
    config = {"model_type": "qwen3_5_text", "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 8,
              "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32, "vocab_size": VOCAB,
              "linear_num_value_heads": 4, "linear_num_key_heads": 4, "linear_key_head_dim": 16, "linear_value_head_dim": 16,
              "linear_conv_kernel_dim": 4, "full_attention_interval": 4, "tie_word_embeddings": True}
    model = qwen3_5.Model(qwen3_5.ModelArgs.from_dict({"model_type": "qwen3_5", "text_config": config}))
    model.eval()
    mx.eval(model.parameters())
    return model


def make_engine(model, **kwargs) -> Engine:
    kwargs.setdefault("slots", 4)
    kwargs.setdefault("pool_gb", 80.5 * PAGE_BYTES / GB)
    kwargs.setdefault("max_tokens", 700)
    return Engine(model, use_kernel=not ON_CPU, **kwargs)


def prompt(length: int, seed: int) -> list[int]:
    return random.Random(seed).choices(range(VOCAB), k=length)


def greedy_reference(model, tokens: list[int], steps: int, forced: tuple[int, ...] = ()) -> list[int]:
    """What mlx-lm writes when it always takes the top token, re-reading the whole text at every step."""
    tokens, written = list(tokens), []
    for i in range(steps):
        top = int(mx.argmax(model(mx.array(tokens)[None])[0, -1]).item())
        written.append(forced[i] if i < len(forced) else top)
        tokens.append(written[-1])
    return written


def greedy(prompt_tokens: list[int], steps: int, **extra) -> Row:
    return Row(prompt=prompt_tokens, max_tokens=steps, seed=1, temperature=0.0, top_k=1, top_p=1.0, presence_penalty=0.0, **extra)


def sampled(prompt_tokens: list[int], steps: int, seed: int, **extra) -> Row:
    """A row that draws at random and carries a presence penalty (the settings two runs of the engine must agree under)."""
    return Row(prompt=prompt_tokens, max_tokens=steps, seed=seed, temperature=0.9, top_k=6, top_p=0.9, presence_penalty=1.0, **extra)


# -- sampler -------------------------------------------------------------------------------------

def reference_probabilities(scores: np.ndarray, penalties: np.ndarray, temperature: float, top_k: int, top_p: float) -> dict[int, float]:
    """The study's rule written the slow, obvious way: full sort, one row."""
    x = (scores.astype(np.float32) - penalties) / temperature
    order = np.argsort(-x, kind="stable")[:top_k]
    p = np.exp(x[order] - x[order].max())
    p /= p.sum()
    keep = (np.cumsum(p) - p) < top_p
    return {int(t): float(q) for t, q in zip(order[keep], p[keep] / p[keep].sum())}


def test_top_values_finds_the_largest_scores_without_sorting_everything():
    mx.random.seed(0)
    scores = mx.random.normal((5, 6000))
    values, positions = top_values(scores, 20)
    assert mx.array_equal(values, mx.sort(scores, axis=-1)[:, ::-1][:, :20])
    assert mx.array_equal(mx.take_along_axis(scores, positions, axis=-1), values)


def test_sampler_gives_the_studys_probabilities():
    mx.random.seed(1)
    rows, vocab = 6, 6000
    scores = (mx.random.normal((rows, vocab)) * 3).astype(mx.bfloat16)
    penalties = mx.where(mx.random.uniform(shape=(rows, vocab)) < 0.05, 1.5, 0.0)
    temperature, top_k, top_p = [0.6, 0.6, 1.0, 0.3, 0.6, 1.5], [20, 20, 5, 20, 1, 12], [0.95, 0.5, 0.95, 1.0, 0.95, 0.8]
    tokens, probs = kept_probabilities(scores, penalties, temperature=mx.array(temperature), top_k=mx.array(top_k), top_p=mx.array(top_p))
    for r in range(rows):
        expected = reference_probabilities(np.array(scores[r].astype(mx.float32)), np.array(penalties[r]), temperature[r], top_k[r], top_p[r])
        got = {int(t): float(p) for t, p in zip(tokens[r].tolist(), (probs[r] / probs[r].sum()).tolist()) if p > 0}
        assert got.keys() == expected.keys()
        assert all(abs(got[t] - expected[t]) < 1e-5 for t in expected)


def test_sampler_draws_in_proportion_and_takes_the_top_token_at_temperature_zero():
    mx.random.seed(2)
    scores = mx.random.normal((1, 3000)) * 2
    no_penalty = mx.zeros_like(scores)
    settings = {"temperature": mx.array([0.8]), "top_k": mx.array([8]), "top_p": mx.array([0.9])}
    tokens, probs = kept_probabilities(scores, no_penalty, **settings)
    expected = {int(t): float(p) for t, p in zip(tokens[0].tolist(), (probs[0] / probs[0].sum()).tolist()) if p > 0}
    draws = 20_000
    u = row_stream(7, draws)
    picked = next_tokens(mx.broadcast_to(scores, (draws, 3000)), mx.broadcast_to(no_penalty, (draws, 3000)), u,
                         temperature=mx.broadcast_to(settings["temperature"], (draws,)), top_k=mx.broadcast_to(settings["top_k"], (draws,)),
                         top_p=mx.broadcast_to(settings["top_p"], (draws,)))
    counts = np.bincount(np.array(picked), minlength=3000)
    assert set(np.flatnonzero(counts)) == set(expected)
    assert all(abs(counts[t] / draws - p) < 0.012 for t, p in expected.items())
    top = next_tokens(scores, no_penalty, mx.array([0.99]), temperature=mx.array([0.0]), top_k=mx.array([8]), top_p=mx.array([0.9]))
    assert top.item() == mx.argmax(scores[0]).item()
    assert mx.array_equal(row_stream(7, 50), row_stream(7, 100)[:50]), "a row's random numbers do not depend on its cap"


# -- paged attention -----------------------------------------------------------------------------

@pytest.mark.skipif(ON_CPU, reason="the kernel needs the GPU")
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_paged_attention_kernel_matches_mlx_attention(dtype):
    mx.random.seed(3)
    slots, heads, kv_heads, dim = 5, 8, 2, 64
    lengths = [0, 3, 255, 700, 9000]                              # an empty slot, a short row, rows across 1 to 35 pages
    pool = PagePool(layers=1, slots=slots, pages=40, kv_heads=kv_heads, head_dim=dim, dtype=dtype, max_row_tokens=9300)
    for slot, n in enumerate(lengths):
        if n:
            k = (mx.random.normal((1, kv_heads, n, dim)) * 0.5).astype(dtype)
            v = mx.random.normal((1, kv_heads, n, dim)).astype(dtype)
            pool.load(slot, [(k, v)])
    active = np.array([n > 0 for n in lengths])
    new_k = (mx.random.normal((slots, kv_heads, dim)) * 0.5).astype(dtype)
    new_v = mx.random.normal((slots, kv_heads, dim)).astype(dtype)
    q = mx.random.normal((slots, heads, dim)).astype(dtype)
    pool.write(0, new_k, new_v)
    fast = pool.attend(0, q, active, dim ** -0.5).astype(mx.float32)
    n_pages, n_buf = np.where(active, pool.n_pages, 0), np.where(active, pool.n_buf + 1, 0)
    slow = pool.reference(0, q, n_pages, n_buf, dim ** -0.5).astype(mx.float32)
    tolerance = 2e-4 if dtype == mx.float32 else 0.02
    assert mx.abs(fast - slow).max().item() < tolerance
    assert mx.abs(fast[0]).max().item() == 0, "a slot with no row in it gets zeros"
    pool.advance(active)
    assert pool.tokens_held(4) == 9001 and pool.n_pages[2] == 1 and pool.n_buf[2] == 0, "a full buffer became a page"


def test_page_pool_shares_and_returns_pages():
    pool = PagePool(layers=1, slots=3, pages=6, kv_heads=2, head_dim=32, dtype=mx.float32, max_row_tokens=1200, use_kernel=False)
    k = mx.random.normal((1, 2, 600, 32))
    pool.load(0, [(k, k)])
    assert len(pool.free) == 4 and pool.tokens_held(0) == 600
    pool.adopt(pool.hold(0), 1)
    same = [mx.array_equal(a, b) for a, b in zip(pool.gather(0, 0), pool.gather(0, 1))]
    assert all(same) and len(pool.free) == 4, "a branch holds the same history without taking pages"
    pool.release(0)
    assert len(pool.free) == 4, "shared pages stay while the branch still points at them"
    held = pool.hold(1)
    pool.release(1)
    assert len(pool.free) == 4, "a held history keeps its pages without a slot"
    pool.adopt(held, 2)
    assert all(mx.array_equal(a, k) for a in pool.gather(0, 2)), "and a slot that takes it over reads the same history"
    pool.drop(pool.hold(2))
    pool.release(2)
    assert len(pool.free) == 6


# -- the engine against mlx-lm ---------------------------------------------------------------------

def test_engine_writes_what_mlx_lm_writes():
    model = tiny_model()
    prompts = [prompt(5, 0), prompt(250, 1), prompt(300, 2), prompt(1, 3)]   # one crosses a page while writing, one starts with pages
    engine = make_engine(model)
    records = engine.run([greedy(p, 30) for p in prompts])
    assert sorted(r["written"] for r in records) == [30] * 4
    by_prompt = {r["prompt_tokens"]: r["tokens"] for r in records}
    for p in prompts:
        assert by_prompt[len(p)] == greedy_reference(model, p, 30)


def test_engine_scores_equal_mlx_lm_scores_at_every_step():
    model = tiny_model(1)
    text, forced = prompt(250, 4), tuple(prompt(20, 5))      # the row crosses into its second page while writing
    seen = []
    engine = make_engine(model, slots=1)
    engine.probe = lambda scores: seen.append(scores[0])
    engine.run([greedy(text, 20, forced=forced)])
    want = model(mx.array(text + list(forced[:-1]))[None])[0, len(text) - 1:]
    # 32-bit rounding through 8 layers: under 1e-4 on the CPU, up to about 2e-4 on the GPU (its kernels round differently)
    assert len(seen) == 20 and mx.abs(mx.stack(seen) - want).max().item() < 5e-4


def test_a_row_does_not_depend_on_its_neighbours():
    model = tiny_model(2)
    def row(i: int) -> Row:
        return Row(prompt=prompt(20 + 7 * i, 10 + i), max_tokens=60, seed=seed_for("run", i), temperature=0.9, top_k=6,
                   top_p=0.9, presence_penalty=1.0, meta={"i": i})
    def run(rows: list[Row]) -> list[dict]:
        engine = make_engine(model, slots=4)
        engine.narrow_when_idle = False                      # the same batch width with and without company
        return engine.run(rows)

    alone = run([row(2)])[0]["tokens"]
    together = {r["meta"]["i"]: r["tokens"] for r in run([row(i) for i in range(6)])}
    assert together[2] == alone
    assert len({tuple(t) for t in together.values()}) == 6


def test_stop_token_cap_and_forced_tokens():
    model = tiny_model(3)
    text = prompt(12, 6)
    free = greedy_reference(model, text, 12)
    stopper = free[5]
    records = make_engine(model).run([greedy(text, 12, stop_tokens=(stopper,), meta={"n": "stop"}),
                                      greedy(text, 8, meta={"n": "cap"}),
                                      greedy(text, 10, forced=(11, 22, 33), meta={"n": "forced"})])
    by = {r["meta"]["n"]: r for r in records}
    first = free.index(stopper)
    assert by["stop"]["finish"] == "stop" and by["stop"]["tokens"] == free[: first + 1]
    assert by["cap"]["finish"] == "length" and by["cap"]["tokens"] == free[:8]
    assert by["forced"]["tokens"] == greedy_reference(model, text, 10, forced=(11, 22, 33)) and by["forced"]["sampled"] == 7


def test_a_branch_shares_its_trunk_and_matches_a_fresh_row():
    model = tiny_model(4)
    trunk = greedy(prompt(270, 7), 40, branch_at=(10, 25))

    def brancher(row: Row, written: list[int]) -> Row:
        return greedy(row.prompt + written, 9, forced=(5, 6, 7), seen=tuple(written), meta={"at": len(written)})

    engine = make_engine(model)
    records = engine.run([trunk], brancher=brancher)
    assert engine.stats["branches_shared"] == 2 and len(records) == 3
    trunk_tokens = next(r["tokens"] for r in records if r["parent"] is None)
    for at in (10, 25):
        branch = next(r for r in records if r["meta"].get("at") == at)
        assert branch["tokens"] == greedy_reference(model, trunk.prompt + trunk_tokens[:at], 9, forced=(5, 6, 7))
    assert trunk_tokens == greedy_reference(model, trunk.prompt, 40), "branching does not disturb the trunk"
    crowded = make_engine(model, slots=1)
    crowded.max_waiting = 0                                                    # no free slot and no waiting: read in later instead
    assert {tuple(r["tokens"]) for r in crowded.run([trunk], brancher=brancher)} == {tuple(r["tokens"]) for r in records}
    assert crowded.stats["branches_read"] == 2 and crowded.stats["branches_shared"] == 0


def test_a_sampled_branch_is_the_same_shared_waiting_or_read_in_and_counts_its_trunk_for_the_penalty():
    model = tiny_model(4)
    trunk = sampled(prompt(270, 7), 40, seed=3, branch_at=(12,))

    def brancher(row: Row, written: list[int]) -> Row:
        return sampled(row.prompt + written, 14, seed=4, forced=(5, 6), seen=tuple(written), meta={"branch": True})

    shared = make_engine(model)
    with_room = shared.run([trunk], brancher=brancher)
    waiting = make_engine(model, slots=1)                # no free slot: the branch waits with its trunk's state
    waited = waiting.run([trunk], brancher=brancher)
    reading = make_engine(model, slots=1)
    reading.max_waiting = 0                              # no waiting either: the branch is read in from the trunk's tokens
    read_in = reading.run([trunk], brancher=brancher)
    assert shared.stats["branches_shared"] == 1 and shared.stats["branches_waited"] == 0
    assert waiting.stats["branches_shared"] == 1 and waiting.stats["branches_waited"] == 1
    assert waiting.stats["read"] == len(trunk.prompt) - 1, "a branch that waited reads nothing again"
    assert reading.stats["branches_read"] == 1 and reading.stats["read"] > 2 * len(trunk.prompt)
    pick = lambda records: next(r["tokens"] for r in records if r["meta"].get("branch"))
    assert pick(with_room) == pick(waited) == pick(read_in)
    assert not waiting.waiting and len(waiting.pool.free) == waiting.pool.pages, "every page came back"
    trunk_tokens = next(r["tokens"] for r in with_room if not r["meta"].get("branch"))
    forgetful = make_engine(model).run([sampled(trunk.prompt + trunk_tokens[:12], 14, seed=4, forced=(5, 6))])[0]["tokens"]
    assert pick(with_room) != forgetful, "without the trunk's tokens in the penalty the branch writes something else"


def test_a_row_set_aside_is_replayed_to_the_same_text(monkeypatch):
    model = tiny_model(5)
    rows = [sampled(prompt(10, 20 + i), 600, seed=i, forced=(9, 8, 7), meta={"i": i}) for i in range(2)]
    roomy = {r["meta"]["i"]: r["tokens"] for r in make_engine(model, slots=2).run(rows)}
    monkeypatch.setattr(engine_module, "GROWTH_PAGES", 0)
    tight = make_engine(model, slots=2, pool_gb=3.5 * PAGE_BYTES / GB)
    squeezed = {r["meta"]["i"]: r["tokens"] for r in tight.run(rows)}
    assert tight.stats["set_aside"] >= 1
    assert squeezed == roomy


def test_per_row_edit_equals_the_merged_weights_and_leaves_other_rows_alone():
    model = tiny_model(6)
    path = "language_model.model.layers.3.mlp.down_proj"
    mx.random.seed(9)
    a, b = mx.random.normal((128, 1)) * 0.3, mx.random.normal((1, 64)) * 0.3
    text = prompt(30, 8)
    plain = greedy_reference(model, text, 25)
    engine = make_engine(model, edits={"e": {path: (a, b)}})
    records = engine.run([greedy(text, 25, meta={"n": "plain"}), greedy(text, 25, edit="e", edit_scale=2.0, meta={"n": "edited"})])
    engine.close()
    by = {r["meta"]["n"]: r["tokens"] for r in records}
    layer = model.language_model.model.layers[3].mlp.down_proj
    original = layer.weight
    layer.weight = original + 2.0 * (a @ b).T
    merged = greedy_reference(model, text, 25)
    layer.weight = original
    assert by["plain"] == plain and by["edited"] == merged and merged != plain


def test_the_batch_narrows_when_rows_run_out_without_changing_what_is_written():
    model = tiny_model(9)
    rows = [sampled(prompt(5 + i, 50 + i), 12 if i < 14 else 300, seed=i, meta={"i": i}) for i in range(16)]   # 14 short rows, 2 long ones
    wide = make_engine(model, slots=16)
    wide.narrow_when_idle = False
    expected = {r["meta"]["i"]: r["tokens"] for r in wide.run(rows)}
    engine = make_engine(model, slots=16)
    records = engine.run(rows)
    assert engine.stats["narrowed"] >= 1 and engine.slots == 4 and wide.slots == 16
    assert {r["meta"]["i"]: r["tokens"] for r in records} == expected
    assert all(r["widths"] == ([16] if r["meta"]["i"] < 14 else [16, 4]) for r in records), "a row's record says which widths it ran at"


def test_a_branch_continues_its_trunk_at_once_or_after_a_wait_that_ends_while_the_trunk_runs():
    model = tiny_model(12)
    trunk = greedy(prompt(270, 7), 60, branch_at=(20,), meta={"n": "trunk"})
    other = greedy(prompt(15, 8), 30, meta={"n": "other"})                 # leaves its slot at step 30, with the trunk half way
    brancher = lambda row, written: greedy(row.prompt + written, 25, seen=tuple(written), meta={"n": "branch"})
    for slots, places, waited, read in ((3, 3, 0, 0), (2, 2, 1, 0), (2, 0, 0, 1)):
        engine = make_engine(model, slots=slots)
        engine.max_waiting = places
        by = {r["meta"]["n"]: r for r in engine.run([trunk, other], brancher=brancher)}
        assert (engine.stats["branches_waited"], engine.stats["branches_read"]) == (waited, read)
        assert by["branch"]["tokens"] == by["trunk"]["tokens"][20:45], "with no randomness a branch writes what its trunk goes on to write"
        assert by["branch"]["read_in"] == bool(read) and by["trunk"]["read_in"] is False and by["branch"]["widths"] == [slots]


def test_slots_kept_free_let_branches_start_at_once_and_the_others_wait():
    model = tiny_model(10)
    trunks = [greedy(prompt(20 + i, 60 + i), 30, branch_at=(8,), meta={"i": i}) for i in range(4)]
    brancher = lambda row, written: greedy(row.prompt + written, 6, forced=(3, 4), seen=tuple(written), meta={"of": row.meta["i"]})
    crowded = make_engine(model, slots=4)
    waited = crowded.run(trunks, brancher=brancher)
    roomy = make_engine(model, slots=4)
    records = roomy.run(trunks, brancher=brancher, keep_free=2)
    assert crowded.stats["branches_shared"] == 4 and crowded.stats["branches_waited"] == 4
    assert roomy.stats["branches_shared"] == 4 and roomy.stats["branches_waited"] == 0
    texts = lambda rows: {(r["meta"].get("of"), r["meta"].get("i"), tuple(r["tokens"])) for r in rows}
    assert len(records) == 8 and texts(records) == texts(waited)


def test_a_waiting_branch_whose_trunk_is_gone_gives_its_pages_up_when_the_pool_runs_short(monkeypatch):
    model = tiny_model(11)
    monkeypatch.setattr(engine_module, "GROWTH_PAGES", 0)
    short = greedy(prompt(300, 90), 20, branch_at=(10,), meta={"n": "short"})     # holds one page; done after 20 tokens
    long = greedy(prompt(10, 91), 600, branch_at=(20,), meta={"n": "long"})       # its branch takes the slot the short row leaves

    def brancher(row: Row, written: list[int]) -> Row:
        return greedy(row.prompt + written, 6 if row.meta["n"] == "short" else 500, forced=(3, 4), seen=tuple(written),
                      meta={"of": row.meta["n"]})

    tight = make_engine(model, slots=2, pool_gb=4.5 * PAGE_BYTES / GB)            # 4 pages: the two long rows need all of them
    records = tight.run([short, long], brancher=brancher)
    assert tight.stats["branches_waited"] == 1 and tight.stats["branches_read"] == 1 and tight.stats["branches_shared"] == 1
    assert sorted(r["written"] for r in records) == [6, 20, 500, 600]
    of_short = next(r["tokens"] for r in records if r["meta"].get("of") == "short")
    short_tokens = next(r["tokens"] for r in records if r["meta"].get("n") == "short")
    assert of_short == greedy_reference(model, short.prompt + short_tokens[:10], 6, forced=(3, 4))
    assert not tight.waiting and len(tight.pool.free) == tight.pool.pages


def test_a_run_stops_starting_rows_once_the_question_is_decided():
    model = tiny_model(8)
    rows = [greedy(prompt(6, 40 + i), 12, meta={"i": i}) for i in range(10)]
    engine = make_engine(model, slots=2)
    records = engine.run(rows, enough=lambda so_far: len(so_far) >= 3)
    assert 3 <= len(records) <= 5, "the rows running at that moment still finish; the rest are never started"


def test_a_row_set_aside_comes_back_even_after_the_run_closed_or_with_slots_kept_free(monkeypatch):
    model = tiny_model(5)
    monkeypatch.setattr(engine_module, "GROWTH_PAGES", 0)
    rows = [greedy(prompt(10, 70), 8, meta={"i": 0})] + [greedy(prompt(10, 71 + i), 600, meta={"i": i + 1}) for i in range(2)]
    tight = make_engine(model, slots=3, pool_gb=3.5 * PAGE_BYTES / GB)
    records = tight.run(rows, enough=lambda so_far: True)            # closed as soon as the short row is done
    assert tight.stats["set_aside"] >= 1 and sorted(r["written"] for r in records) == [8, 600, 600]
    held_back = make_engine(model, slots=16, pool_gb=3.5 * PAGE_BYTES / GB)
    mixed = [greedy(prompt(10, 80 + i), 600 if i < 2 else 10, meta={"i": i}) for i in range(8)]
    assert len(held_back.run(mixed, brancher=lambda row, written: None, keep_free=8)) == 8


# -- the planner (bench/plan_sim.py) against the engine --------------------------------------------

SCHEDULES = (    # by seed // 1000: a mix; long rows in a small pool (rows are set aside); short and long rows in wide batches (they narrow)
    {"slots": (4, 4, 6, 8), "keep": (0, 0, 1, 2), "growth": (0, 1, 2), "pages": (6, 8, 10, 14, 20, 40), "places": (None, None, 1, 0),
     "rows": (3, 9), "written": ((5, 60), (100, 500), (300, 900)), "points": (0, 0, 1, 2, 3), "branch": (50, 400)},
    {"slots": (2, 3, 4, 6), "keep": (0, 0, 1), "growth": (0, 0, 1), "pages": (4, 5, 6, 8, 10, 12), "places": (None, None, 1),
     "rows": (3, 7), "written": ((300, 1200),), "points": (0, 1, 1, 2), "branch": (100, 400)},
    {"slots": (8, 8, 16), "keep": (0, 2, 4), "growth": (0, 1, 2), "pages": (10, 16, 24, 60), "places": (None, None, 2),
     "rows": (5, 12), "written": ((5, 60), (5, 60), (200, 700)), "points": (0, 1, 2, 3), "branch": (50, 500)},
)


def engine_and_planner(model, monkeypatch, seed: int) -> tuple[dict, dict] | None:
    """One random small run through the engine and through the planner: what each counts. None if the pool is too small for both."""
    rng, kind = random.Random(seed), SCHEDULES[seed // 1000]
    slots, keep_free, growth, pages, places = (rng.choice(kind[name]) for name in ("slots", "keep", "growth", "pages", "places"))
    places = slots if places is None else places
    rows = []                                                              # per trunk: prompt tokens, written tokens, {branch point: branch length}
    for _ in range(rng.randint(*kind["rows"])):
        prompt_tokens, written = rng.randint(2, 300), rng.choice([rng.randint(*span) for span in kind["written"]])
        points = sorted(rng.sample(range(1, written), k=min(rng.choice(kind["points"]), written - 1)))
        rows.append((prompt_tokens, written, {at: rng.choice([rng.randint(1, 30), rng.randint(*kind["branch"])]) for at in points}))
    monkeypatch.setattr(engine_module, "GROWTH_PAGES", growth)
    monkeypatch.setattr(plan_sim, "GROWTH_PAGES", growth)
    engine = make_engine(model, slots=slots, pool_gb=(pages + 0.5) * PAGE_BYTES / GB, max_tokens=1400, max_prompt=1400)
    engine.max_waiting = places
    trunks = [sampled(prompt(n, seed + i), written, seed=i, branch_at=tuple(branches), meta={"i": i}) for i, (n, written, branches) in enumerate(rows)]
    brancher = lambda row, written: sampled(row.prompt + written, rows[row.meta["i"]][2][len(written)], seed=7, seen=tuple(written))
    replay = lambda: plan_sim.simulate(rows, slots=slots, pool_pages=engine.pool.pages, keep_free=keep_free, waiting_places=places)
    try:
        engine.run(trunks, brancher=brancher, keep_free=keep_free, log=lambda line: None)
    except MemoryError:
        with pytest.raises(SystemExit):
            replay()
        return None
    stats = engine.stats
    counted = {"steps": stats["steps"], "set_aside": stats["set_aside"], "branches_sharing": stats["branches_shared"], "narrowed": stats["narrowed"],
               "branches_that_waited": stats["branches_waited"], "branches_read_in": stats["branches_read"], "tokens_read": stats["read"]}
    planned = replay()
    return counted, {name: planned[name] for name in counted}


def test_the_planner_counts_what_the_engine_does(monkeypatch):
    model = tiny_model(3)
    happened = {"set_aside": 0, "branches_that_waited": 0, "branches_read_in": 0, "narrowed": 0}
    for seed in (1003, 2000, 2005):
        counted, planned = engine_and_planner(model, monkeypatch, seed)
        assert planned == counted, f"schedule {seed}"
        for name in happened:
            happened[name] += counted[name]
    assert all(happened.values()), "these schedules set rows aside, make branches wait, read branches in and narrow the batch"


@pytest.mark.slow
def test_the_planner_counts_what_the_engine_does_on_many_schedules(monkeypatch):
    model = tiny_model(3)
    for seed in (*range(0, 8), *range(1000, 1010), *range(2010, 2018)):
        both = engine_and_planner(model, monkeypatch, seed)
        assert both is None or both[1] == both[0], f"schedule {seed}"


def test_branch_points_are_not_part_of_a_rows_name():
    plain, with_exits = greedy(prompt(5, 1), 10), greedy(prompt(5, 1), 10, branch_at=(4, 8), meta={"note": "same row"})
    assert plain.key("m") == with_exits.key("m") != greedy(prompt(5, 1), 11).key("m") and plain.key("m") != plain.key("other model")


# -- ledger and loop marks -----------------------------------------------------------------------

def test_ledger_keeps_rows_and_a_second_run_skips_them(tmp_path):
    model = tiny_model(7)
    rows = [greedy(prompt(8, 30 + i), 15, meta={"i": i}) for i in range(3)]
    ledger = Ledger(tmp_path / "run")
    first = make_engine(model, model_tag="tiny").run(rows[:2], ledger)
    assert {r["key"] for r in first} == ledger.done() and len(first) == 2
    second = make_engine(model, model_tag="tiny").run(rows, ledger)
    assert [r["meta"]["i"] for r in second] == [2], "only the row that was missing ran"
    stored = {r["meta"]["i"]: ledger.tokens(r) for r in ledger.records()}
    assert stored[0] == first[0]["tokens"] if first[0]["meta"]["i"] == 0 else stored[1] == first[0]["tokens"]
    with open(ledger.rows_path, "a") as f:
        f.write('{"key": "cut off by a cra')
    assert len(ledger.records()) == 3
    ledger.append({"key": "after the crash", "meta": {}}, [1, 2, 3])
    assert [r["key"] for r in ledger.records()][-1] == "after the crash" and len(ledger.records()) == 4
    assert ledger.tokens(ledger.records()[-1]) == [1, 2, 3]


def test_loop_watch_marks_repeated_text_and_the_stop_point():
    decode = lambda ids: " ".join(f"w{i}" for i in ids)
    rng = random.Random(0)
    fresh = [rng.randrange(50_000) for _ in range(1000)]
    looped = fresh + [1, 2, 3, 4, 5, 6, 7] * 400
    watch = LoopWatch()
    for n in range(1, len(looped) + 1):
        watch.update(looped[:n], decode)
    assert watch.first_onset is not None and 800 <= watch.first_onset <= 1300 and watch.looping_tokens > 2000
    assert stop_point(fresh, decode, 512) is None
    cut = stop_point(looped, decode, 1408)
    assert cut is not None and 2300 <= cut <= 2800
    share = sum(audited(f"row{i}", 0.2) for i in range(5000)) / 5000
    assert 0.17 < share < 0.23 and audited("row1", 0.2) == audited("row1", 0.2)
