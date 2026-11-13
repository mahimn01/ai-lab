"""The row engine: many long generations sharing the GPU, each row with its own settings.

What a row is
  One generation: a prompt, a cap, sampling settings, a random seed, optionally a weight
  edit, optionally tokens it is forced to write first, and rules (stop on a loop, branch
  at a token count). `Row` holds exactly that. A condition is a list of rows.

What the engine does (design and measurements: docs/design-row-engine.md)
  - keeps a fixed number of slots; a waiting row takes a slot as soon as one is free and
    the history pool has room, so the batch stays full
  - runs one model pass per step for all slots: the recurrent layers through mlx-lm's
    own code on one state block for all slots, the attention layers through the paged
    history in lab/gen/kv.py
  - picks every row's next token on the GPU with the row's own settings and its own
    random stream (lab/gen/sampler.py), so a row's text does not depend on its neighbours
    (it does depend on the width of the batch: see the last point)
  - can branch a row: the branch shares the trunk's history pages and costs nothing to start.
    If no slot is free at that moment the branch waits, holding a copy of the trunk's small
    per-row state, and still shares the pages when a slot comes free
  - writes each finished row to a ledger at once (lab/gen/ledger.py); a restarted run
    skips what is there
  - if the pool runs short, sets the youngest row aside and replays it later from its
    saved tokens (reading is far faster than writing)
  - narrows the batch once nothing is waiting and half the slots are empty: a step costs
    less with fewer slots, and the last long rows of a run would otherwise pay for all of them.
    A narrower batch is computed in another order, so the rows still running continue with
    other rounding than at full width. narrow_when_idle = False keeps the width

Supported models: the Qwen3.5 text family as loaded by mlx-lm (recurrent layers plus
attention layers). The model's weights are not changed.

Use:     engine = Engine(model, decode=tokenizer.decode)
         records = engine.run(rows, Ledger("runs/phase1/x"))
Changes: memory, and the ledger folder it is given.
"""

from __future__ import annotations

import hashlib
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.cache import ArraysCache

from lab.gen.kv import PAGE, Held, PagePool
from lab.gen.ledger import Ledger, row_key
from lab.gen.loops import LoopWatch
from lab.gen.sampler import next_tokens, row_stream

GB = 2**30
READ_CHUNK = 512        # tokens read per pass when a row's prompt or saved text is read in
WIDTHS = (4, 8, 16, 24, 32, 48, 64, 96, 128)   # batch widths the engine narrows to when rows run out
GROWTH_PAGES = 4        # pages kept free per running row before a new row is let in (about 1,000 steps of growth)
BUFFER_CACHE_GB = 2     # cap on memory MLX keeps for reuse (uncapped, it grew to 20 GB in the study's first long run)


def seed_for(*parts: object) -> int:
    """A 32-bit seed from anything that identifies a row (run seed, problem id, sample number, branch name)."""
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


@dataclass
class Row:
    """Everything that defines one generation. `meta` is carried into the ledger and does not affect the output."""

    prompt: list[int]                    # tokens read before anything is written
    max_tokens: int                      # cap on written tokens (forced ones count)
    seed: int
    temperature: float = 0.6             # 0 = always the most likely token
    top_k: int = 20
    top_p: float = 0.95
    presence_penalty: float = 1.5
    stop_tokens: tuple[int, ...] = ()
    forced: tuple[int, ...] = ()         # tokens the row writes first, without sampling
    seen: tuple[int, ...] = ()           # tokens that already count for the presence penalty (a branch's trunk)
    edit: str | None = None              # name of a per-row weight edit given to the engine
    edit_scale: float = 1.0
    stop_after_loop: int | None = None   # stop once this many tokens in a row were looping (None: never)
    branch_at: tuple[int, ...] = ()      # written-token counts at which the engine asks for a branch
    meta: dict = field(default_factory=dict)

    def key(self, model_tag: str) -> str:
        """The row's name in a ledger: a hash of everything about the row that decides what it writes (not meta, not branch_at).

        The run decides the rest, down to rounding: the width of the batch at each step, and whether the row was read back in
        on the way. A row's record carries both (`widths`, `set_aside`, `read_in`).
        """
        spec = {k: v for k, v in asdict(self).items() if k not in ("meta", "branch_at")}
        return row_key({"model": model_tag, **spec})


@dataclass(eq=False)
class _Live:
    """A row that is running or waiting, with what it has written so far."""

    row: Row
    key: str
    parent: str | None = None
    written: list[int] = field(default_factory=list)
    sampled: int = 0
    forced: deque = field(default_factory=deque)
    watch: LoopWatch | None = None
    started: float | None = None
    set_aside: int = 0
    steps: int = 0
    read_in: bool = False                                    # a branch that started from saved tokens, not from its trunk's history
    widths: list[int] = field(default_factory=list)          # the batch widths it ran at, in order


@dataclass(eq=False)
class _Waiting:
    """A branch with no slot yet, holding what it needs to start exactly where its trunk stood."""

    live: _Live
    history: Held
    states: dict[int, list[np.ndarray]]      # per recurrent layer: this row's part of every state block
    pending: int


class RowEdit(nn.Module):
    """A linear layer plus a small change that differs per row: y = W x + scale_row * (x A_row) B_row.

    a is [edits + 1, inputs, rank] and b is [edits + 1, rank, outputs]; entry 0 is all zeros ("no edit").
    `rows` (which entry each row uses) and `scales` are set by the engine before every pass.
    """

    def __init__(self, base: nn.Module, a: mx.array, b: mx.array, who: "_EditRows") -> None:
        super().__init__()
        self.base = base
        self.lora_a, self.lora_b = a, b
        self._who = who

    def __call__(self, x: mx.array) -> mx.array:
        y = self.base(x)
        if self._who.rows is None:
            return y
        z = (x @ self.lora_a[self._who.rows]) @ self.lora_b[self._who.rows]
        return y + (self._who.scales[:, None, None] * z).astype(x.dtype)


class _EditRows:
    rows: mx.array | None = None
    scales: mx.array | None = None


def _module_at(model: nn.Module, path: str) -> tuple[nn.Module, str]:
    """(parent module, attribute name) for a dotted path such as language_model.model.layers.15.mlp.down_proj."""
    parts = path.split(".")
    obj = model
    for part in parts[:-1]:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj, parts[-1]


class Engine:
    """See the top of this file.

    slots       rows running at once
    pool_gb     memory for the attention history of all rows together
    max_tokens  the largest cap any row will have (sets the size of per-row tables)
    edits       {name: {module path: (a [inputs, rank], b [rank, outputs])}} per-row weight edits, kept unmerged
    decode      tokens -> text, needed only to watch for loops
    """

    def __init__(self, model: nn.Module, *, slots: int = 48, pool_gb: float = 10.0, max_tokens: int = 32768,
                 max_prompt: int = 1024, edits: dict[str, dict[str, tuple[mx.array, mx.array]]] | None = None,
                 model_tag: str = "", decode: Callable[[list[int]], str] | None = None, use_kernel: bool = True) -> None:
        mx.set_cache_limit(BUFFER_CACHE_GB * GB)
        self.model, self.model_tag, self.decode = model, model_tag, decode
        self.lm = model.language_model if hasattr(model, "language_model") else model
        self.text = self.lm.model
        self.slots, self.max_tokens = slots, max_tokens
        attention = [layer.self_attn for layer in self.text.layers if not layer.is_linear]
        dtype = self.text.embed_tokens.weight.dtype
        per_page = 2 * len(attention) * attention[0].num_key_value_heads * PAGE * attention[0].head_dim * dtype.size
        self.pool = PagePool(layers=len(attention), slots=slots, pages=int(pool_gb * GB // per_page),
                             kv_heads=attention[0].num_key_value_heads, head_dim=attention[0].head_dim, dtype=dtype,
                             max_row_tokens=max_prompt + max_tokens, use_kernel=use_kernel)

        # One state block for all slots per recurrent layer, shaped like what mlx-lm makes for one row.
        probe = self.lm.make_cache()
        self.text(mx.array([[0]]), cache=probe)
        self.rec: dict[int, ArraysCache] = {}
        for i, (layer, c) in enumerate(zip(self.text.layers, probe)):
            if layer.is_linear:
                self.rec[i] = ArraysCache(size=len(c.cache))
                self.rec[i].cache = [mx.zeros((slots, *s.shape[1:]), s.dtype) for s in c.cache]

        self._who = _EditRows()
        self._edit_index = {None: 0}
        self._wrapped: list[tuple[nn.Module, str, nn.Module]] = []
        self._install_edits(edits or {})

        vocab = self.text.embed_tokens.weight.shape[0]
        self.mask = mx.zeros((slots, vocab), mx.float32)            # the presence penalty of every token, per slot
        self.streams = mx.zeros((slots, max_tokens + 1), mx.float32)
        mx.eval(self.mask, self.streams, [c.cache for c in self.rec.values()])
        self.pending = np.zeros(slots, np.int32)                     # the token each slot reads next
        self.settings = {name: np.zeros(slots, kind) for name, kind in
                         (("temperature", np.float32), ("top_k", np.int32), ("top_p", np.float32), ("penalty", np.float32),
                          ("edit", np.int32), ("edit_scale", np.float32))}
        self.settings["top_k"][:] = 1      # harmless values for slots with no row in them
        self.settings["top_p"][:] = 1
        self.live: dict[int, _Live] = {}
        self.free_slots = list(range(slots - 1, -1, -1))
        self.probe: Callable[[mx.array], None] | None = None        # if set, called with every step's scores [slots, vocabulary]
        self.stats = {"steps": 0, "written": 0, "read": 0, "set_aside": 0, "branches_shared": 0, "branches_waited": 0,
                      "branches_read": 0, "narrowed": 0}
        self.narrow_when_idle = True
        self.waiting: deque[_Waiting] = deque()                      # branches holding their trunk's state until a slot is free
        self.max_waiting = slots                                     # each holds about 25 MB with the 0.8B model; more are read in later

    # -- per-row weight edits ------------------------------------------------------------------

    def _install_edits(self, edits: dict) -> None:
        if not edits:
            return
        names = sorted(edits)
        self._edit_index.update({name: i + 1 for i, name in enumerate(names)})
        for path in sorted({p for e in edits.values() for p in e}):
            parent, attr = _module_at(self.model, path)
            base = getattr(parent, attr)
            shapes = next(e[path] for e in edits.values() if path in e)
            zero_a, zero_b = mx.zeros_like(shapes[0]), mx.zeros_like(shapes[1])
            a = mx.stack([zero_a] + [edits[n][path][0] if path in edits[n] else zero_a for n in names])
            b = mx.stack([zero_b] + [edits[n][path][1] if path in edits[n] else zero_b for n in names])
            setattr(parent, attr, RowEdit(base, a, b, self._who))
            self._wrapped.append((parent, attr, base))

    def close(self) -> None:
        """Put the model's own layers back where per-row edits were attached."""
        for parent, attr, base in self._wrapped:
            setattr(parent, attr, base)
        self._wrapped.clear()

    def _edits_for(self, rows: np.ndarray, scales: np.ndarray) -> None:
        if self._wrapped:
            self._who.rows, self._who.scales = mx.array(rows), mx.array(scales)

    # -- letting rows in and out ---------------------------------------------------------------

    def _read(self, tokens: list[int], row: Row) -> list:
        """Read tokens through the model the ordinary way (mlx-lm's own caches) and return those caches."""
        cache = self.lm.make_cache()
        self._edits_for(np.array([self._edit_index[row.edit]], np.int32), np.array([row.edit_scale], np.float32))
        for start in range(0, len(tokens), READ_CHUNK):
            self.text(mx.array(tokens[start:start + READ_CHUNK])[None], cache=cache)
            mx.eval([c.state for c in cache])
        self.stats["read"] += len(tokens)
        return cache

    def _settle(self, slot: int, live: _Live) -> None:
        """Fill in the per-slot settings of a row that has just been given a slot."""
        row = live.row
        s = self.settings
        s["temperature"][slot], s["top_k"][slot], s["top_p"][slot] = row.temperature, row.top_k, row.top_p
        s["penalty"][slot], s["edit"][slot], s["edit_scale"][slot] = row.presence_penalty, self._edit_index[row.edit], row.edit_scale
        self.streams[slot, : row.max_tokens + 1] = row_stream(row.seed, row.max_tokens + 1)
        self.live[slot] = live
        live.started = live.started or time.perf_counter()
        if live.widths[-1:] != [self.slots]:
            live.widths.append(self.slots)
        if live.watch is None and self.decode is not None:
            live.watch = LoopWatch()

    def _penalties(self, slot: int, live: _Live) -> None:
        """Set a slot's presence-penalty table from what its row counts as already written."""
        seen = sorted(set(live.row.seen) | set(live.written))
        self.mask[slot] = mx.zeros_like(self.mask[slot])
        if seen:
            self.mask[slot, mx.array(seen)] = live.row.presence_penalty

    def _admit(self, live: _Live) -> None:
        """Give a waiting row a slot by reading its prompt (and whatever it had already written)."""
        slot = self.free_slots.pop()
        live.read_in = live.read_in or (live.parent is not None and not live.written)
        text = live.row.prompt + live.written
        cache = self._read(text[:-1], live.row) if len(text) > 1 else None
        history = []
        for i, layer in enumerate(self.text.layers):
            if layer.is_linear:
                for j, block in enumerate(self.rec[i].cache):
                    block[slot] = cache[i][j][0] if cache is not None else mx.zeros_like(block[slot])
            elif cache is not None:
                history.append((cache[i].keys[..., : cache[i].offset, :], cache[i].values[..., : cache[i].offset, :]))
        if history:
            self.pool.load(slot, history)
        self._penalties(slot, live)
        self.pending[slot] = text[-1]
        live.forced = deque(live.row.forced[len(live.written) - live.sampled:])
        self._settle(slot, live)
        mx.eval(self.mask, self.streams, [c.cache for c in self.rec.values()], self.pool.keys, self.pool.values,
                self.pool.buf_keys, self.pool.buf_values)

    def _hold(self, parent_slot: int, live: _Live) -> _Waiting:
        """What a new branch needs to start where its trunk stands right now: the trunk's pages (shared), and a copy of
        its buffer and of its recurrent state (in 32-bit, which holds every value exactly)."""
        states = {i: [np.array(block[parent_slot].astype(mx.float32)) for block in cache.cache] for i, cache in self.rec.items()}
        return _Waiting(live, self.pool.hold(parent_slot), states, int(self.pending[parent_slot]))

    def _resume(self, held: _Waiting) -> None:
        """Give a branch a slot, at once or after a wait. It starts from the state its trunk had when it was asked for."""
        slot = self.free_slots.pop()
        self.pool.adopt(held.history, slot)
        for i, cache in self.rec.items():
            for block, state in zip(cache.cache, held.states[i]):
                block[slot] = mx.array(state).astype(block.dtype)
        self._penalties(slot, held.live)
        self.pending[slot] = held.pending
        held.live.forced = deque(held.live.row.forced)
        self._settle(slot, held.live)
        self.stats["branches_shared"] += 1
        mx.eval(self.mask, self.streams, [c.cache for c in self.rec.values()], self.pool.buf_keys, self.pool.buf_values)

    def _leave(self, slot: int) -> _Live:
        live = self.live.pop(slot)
        self.pool.release(slot)
        self.free_slots.append(slot)
        return live

    def _narrow(self, width: int) -> None:
        """Move the running rows into the first `width` slots and give up the rest of every per-slot block."""
        empty = [slot for slot in range(width) if slot not in self.live]
        for src in sorted(slot for slot in self.live if slot >= width):
            dst = empty.pop(0)
            self.pool.move(src, dst)
            for cache in self.rec.values():
                for block in cache.cache:
                    block[dst] = block[src]
            self.mask[dst] = self.mask[src]
            self.streams[dst] = self.streams[src]
            self.pending[dst] = self.pending[src]
            for values in self.settings.values():
                values[dst] = values[src]
            self.live[dst] = self.live.pop(src)
        for cache in self.rec.values():
            cache.cache = [mx.contiguous(block[:width]) for block in cache.cache]
        self.mask, self.streams = mx.contiguous(self.mask[:width]), mx.contiguous(self.streams[:width])
        self.pending = self.pending[:width].copy()
        self.settings = {name: values[:width].copy() for name, values in self.settings.items()}
        self.pool.shrink(width)
        self.slots = width
        self.free_slots = [slot for slot in range(width - 1, -1, -1) if slot not in self.live]
        for live in self.live.values():
            live.widths.append(width)
        self.stats["narrowed"] += 1
        mx.eval(self.mask, self.streams, [c.cache for c in self.rec.values()], self.pool.buf_keys, self.pool.buf_values)

    def _room_for(self, tokens: int) -> bool:
        return len(self.pool.free) - tokens // PAGE >= GROWTH_PAGES * (len(self.live) + 1)

    def _make_room(self, queue: deque) -> None:
        """If the pool cannot give a page to every row about to fill its buffer, free some.

        First waiting branches whose trunk has gone (only they keep its pages alive; they are read in
        later instead), then the youngest running rows, which are set aside.
        """
        def short() -> bool:
            filling = sum(1 for slot in self.live if self.pool.n_buf[slot] == PAGE - 1)
            return len(self.pool.free) < filling

        while short():
            running = {live.key for live in self.live.values()}
            orphan = next((held for held in reversed(self.waiting) if held.live.parent not in running), None)
            if orphan is not None:
                self.waiting.remove(orphan)
                self.pool.drop(orphan.history)
                self.stats["branches_read"] += 1
                queue.append(orphan.live)
                continue
            if len(self.live) == 1:
                raise MemoryError("the history pool is too small for a single row; raise pool_gb")
            slot = min(self.live, key=self.pool.tokens_held)
            live = self._leave(slot)
            live.set_aside += 1
            self.stats["set_aside"] += 1
            queue.appendleft(live)

    # -- one step ------------------------------------------------------------------------------

    def _attention(self, index: int, attn: nn.Module, x: mx.array, positions: mx.array, active: np.ndarray) -> mx.array:
        """One attention layer for one new token per slot (mlx-lm's Qwen3NextAttention, with the history in pages)."""
        slots = x.shape[0]
        queries, gate = mx.split(attn.q_proj(x).reshape(slots, 1, attn.num_attention_heads, -1), 2, axis=-1)
        gate = gate.reshape(slots, 1, -1)
        keys, values = attn.k_proj(x), attn.v_proj(x)
        queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
        keys = attn.k_norm(keys.reshape(slots, 1, attn.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(slots, 1, attn.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        queries, keys = attn.rope(queries, offset=positions), attn.rope(keys, offset=positions)
        self.pool.write(index, keys[:, :, 0, :], values[:, :, 0, :])
        out = self.pool.attend(index, queries[:, :, 0, :], active, attn.scale)
        return attn.o_proj(out.reshape(slots, 1, -1) * mx.sigmoid(gate))

    def scores(self) -> mx.array:
        """One model pass: every slot reads its pending token; returns the scores for the token after it, [slots, vocabulary]."""
        active = np.zeros(self.slots, bool)
        active[list(self.live)] = True
        positions = mx.array((self.pool.n_pages * PAGE + self.pool.n_buf).astype(np.int32))
        self._edits_for(self.settings["edit"], self.settings["edit_scale"])
        x = self.text.embed_tokens(mx.array(self.pending))[:, None, :]
        attention = 0
        for i, layer in enumerate(self.text.layers):
            y = layer.input_layernorm(x)
            if layer.is_linear:
                r = layer.linear_attn(y, None, self.rec[i])
            else:
                r = self._attention(attention, layer.self_attn, y, positions, active)
                attention += 1
            h = x + r
            x = h + layer.mlp(layer.post_attention_layernorm(h))
        x = self.text.norm(x)
        out = self.text.embed_tokens.as_linear(x) if self.lm.args.tie_word_embeddings else self.lm.lm_head(x)
        self._active = active
        return out[:, 0, :]

    def _step(self) -> list[int]:
        """Advance every running row by one token; returns the token each slot wrote."""
        s = self.settings
        scores = self.scores()
        if self.probe is not None:
            self.probe(scores)
        counts = np.zeros(self.slots, np.int32)
        forced = np.zeros(self.slots, bool)
        forced_token = np.zeros(self.slots, np.int32)
        for slot, live in self.live.items():
            counts[slot] = live.sampled
            if live.forced:
                forced[slot], forced_token[slot] = True, live.forced[0]
        u = mx.take_along_axis(self.streams, mx.array(counts)[:, None], axis=1)[:, 0]
        drawn = next_tokens(scores, self.mask, u, temperature=mx.array(s["temperature"]), top_k=mx.array(s["top_k"]),
                            top_p=mx.array(s["top_p"]))
        wrote = mx.where(mx.array(forced), mx.array(forced_token), drawn)
        marked = mx.where(mx.array(self._active)[:, None], mx.array(s["penalty"])[:, None], mx.take_along_axis(self.mask, wrote[:, None], axis=1))
        self.mask = mx.put_along_axis(self.mask, wrote[:, None], marked, axis=1)
        mx.eval(wrote, self.mask, [c.cache for c in self.rec.values()], self.pool.buf_keys, self.pool.buf_values)
        self.pool.advance(self._active)
        self.stats["steps"] += 1
        return wrote.tolist()

    def _check(self, rows: list[Row]) -> None:
        """Refuse rows this engine was not built to hold."""
        room = (self.pool.table.shape[1] - 1) * PAGE
        for row in rows:
            if row.max_tokens > self.max_tokens:
                raise ValueError(f"a row has a cap of {row.max_tokens:,} tokens; this engine was built for at most {self.max_tokens:,}")
            if len(row.prompt) + row.max_tokens > room:
                raise ValueError(f"a row could reach {len(row.prompt) + row.max_tokens:,} tokens with its prompt; this engine holds "
                                 f"{room:,} per row (raise max_prompt or max_tokens)")
            if row.edit not in self._edit_index:
                raise ValueError(f"unknown edit {row.edit!r}")

    # -- a whole run ---------------------------------------------------------------------------

    def run(self, rows: list[Row], ledger: Ledger | None = None, *,
            brancher: Callable[[Row, list[int]], Row | None] | None = None,
            enough: Callable[[list[dict]], bool] | None = None, keep_free: int = 0,
            log: Callable[[str], None] = print, log_every: float = 60.0) -> list[dict]:
        """Run rows to the end and return their records (also written to the ledger as each row finishes).

        brancher(row, written) is called when a row reaches one of its branch_at counts while still
        running; it returns the branch as a Row whose prompt is row.prompt + written, or None.
        enough(records) is called whenever a row finishes; once it returns True no new row is started
        (the question the run was for is decided). Rows that had started finish, with their branches;
        rows that never started are left for a later run.
        keep_free slots are held back for branches, so a branch can start the moment it is asked
        for. A branch that finds no free slot waits (up to max_waiting of them) and takes the next
        slot that comes free, ahead of rows that have not started; it still shares its trunk's
        history. Beyond that, or if the pool runs short while its trunk is gone, a branch is read
        in later from the trunk's saved tokens, which costs a pass over that text.
        """
        self._check(rows)
        if keep_free >= self.slots:
            raise ValueError(f"keep_free ({keep_free}) must be smaller than the number of slots ({self.slots})")
        done = ledger.done() if ledger else set()
        keyed = [(row, row.key(self.model_tag)) for row in rows]
        queue = deque(_Live(row, key) for row, key in keyed if key not in done)
        total, finished, records, seen_by_enough = len(queue), 0, [], 0
        t0 = last = time.perf_counter()
        start_written = self.stats["written"]

        def finish(slot: int, reason: str) -> None:
            nonlocal finished
            live = self._leave(slot)
            row = live.row
            record = {"key": live.key, "meta": row.meta, "finish": reason, "written": len(live.written), "sampled": live.sampled,
                      "prompt_tokens": len(row.prompt), "max_tokens": row.max_tokens, "seed": row.seed, "edit": row.edit,
                      "edit_scale": row.edit_scale, "parent": live.parent, "steps": live.steps, "set_aside": live.set_aside,
                      "read_in": live.read_in, "widths": live.widths,
                      "seconds": round(time.perf_counter() - live.started, 2),
                      "finished_at_s": round(time.perf_counter() - t0, 2), **(live.watch.summary() if live.watch else {})}
            if ledger:
                ledger.append(record, live.written)
            records.append({**record, "tokens": live.written})
            finished += 1

        closed = False
        started = lambda waiting: waiting.parent is not None or waiting.set_aside > 0      # not a fresh row
        while queue or self.live or self.waiting:
            while self.waiting and self.free_slots and (self._room_for(0) or not self.live):
                self._resume(self.waiting.popleft())
            while (queue and len(self.free_slots) > (0 if started(queue[0]) else keep_free)
                   and self._room_for(len(queue[0].row.prompt) + len(queue[0].written))):
                self._admit(queue.popleft())
            if not self.live:
                raise MemoryError("the next row does not fit in the history pool; raise pool_gb")
            self._make_room(queue)
            wrote = self._step()
            for slot, live in list(self.live.items()):
                token = wrote[slot]
                row = live.row
                live.steps += 1
                was_forced = bool(live.forced)
                if was_forced:
                    live.forced.popleft()
                else:
                    live.sampled += 1
                live.written.append(token)
                self.pending[slot] = token
                self.stats["written"] += 1
                if live.watch is not None:
                    live.watch.update(live.written, self.decode)
                if not was_forced and token in row.stop_tokens:
                    finish(slot, "stop")
                elif len(live.written) >= row.max_tokens:
                    finish(slot, "length")
                elif row.stop_after_loop and live.watch is not None and live.watch.looping_tokens >= row.stop_after_loop:
                    finish(slot, "loop")
                elif brancher is not None and len(live.written) in row.branch_at:
                    branch = brancher(row, live.written)
                    if branch is not None:
                        if branch.prompt != row.prompt + live.written or (branch.edit, branch.edit_scale) != (row.edit, row.edit_scale):
                            raise ValueError("a branch must have the prompt row.prompt + written and its trunk's edit")
                        self._check([branch])
                        child = _Live(branch, branch.key(self.model_tag), parent=live.key)
                        total += 1
                        if child.key in done:
                            total -= 1
                        elif self.free_slots:
                            self._resume(self._hold(slot, child))
                        elif len(self.waiting) < self.max_waiting:
                            self.waiting.append(self._hold(slot, child))
                            self.stats["branches_waited"] += 1
                        else:
                            self.stats["branches_read"] += 1
                            queue.append(child)
            if self.narrow_when_idle and (closed or not queue) and self.live:
                wanted = len(self.live) + len(self.waiting) + (min(len(self.live), keep_free) if brancher is not None else 0)
                width = next((w for w in WIDTHS if w >= wanted), self.slots)
                if width <= self.slots // 2:
                    self._narrow(width)
            if enough is not None and not closed and len(records) > seen_by_enough:
                seen_by_enough = len(records)
                closed = enough(records)
                if closed:
                    queue = deque(waiting for waiting in queue if started(waiting))
            now = time.perf_counter()
            if now - last >= log_every:
                last = now
                wrote_now = self.stats["written"] - start_written
                log(f"  {finished}/{total} rows done, {wrote_now:,} tokens, {wrote_now / (now - t0):,.0f} tok/s, "
                    f"{len(self.live)} running, {len(queue) + len(self.waiting)} waiting, "
                    f"pool {self.pool.pages - len(self.pool.free)}/{self.pool.pages} pages")
        return records
