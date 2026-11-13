"""The attention history of many rows, kept in pages, and the attention step that reads it.

Words
  An attention layer stores a "key" and a "value" for every earlier token of a row. To
  write the next token it scores the new token's "query" against every key and blends the
  values by those scores. That reading is the part of a step that grows with the length
  of the row.

Why pages
  mlx-lm keeps all rows of a batch in one block as long as the longest row, so a row
  that has just started pays for a neighbour that is 30,000 tokens in. Here the history
  lives in a pool of pages of 256 tokens:
    - a row owns a list of pages and a buffer holding its newest tokens (up to 255);
    - when the buffer fills, it is copied into a free page;
    - a finished row gives its pages back;
    - a full page is never written again, so a branch can share its trunk's pages:
      `hold` keeps a row's history as it stands (pages shared, buffer copied) and
      `adopt` gives it to a slot, at once or later.
  The attention kernel walks each row's own pages and buffer, so a step reads what the
  rows really hold and nothing else.

The kernel
  Two steps, laid out like MLX's own attention kernel for one query and a long history
  (mlx/backend/metal/kernels/sdpa_vector.h, `sdpa_vector_2pass`, MIT licence, copyright
  Apple Inc.), with the page table and the buffer added:
    1. Each row's tokens are dealt out to 64 groups of GPU threads: a group takes every
       64th token, which is 4 tokens of every page, so it looks at the page table once
       per page. A group holds the query heads that share one set of keys (so the keys
       are read once for all of them), 32 threads per head, each with 8 of the 256
       numbers of a key and a value. Scores are folded in as they are read, with a
       running maximum and running sum, in 32-bit.
    2. A second small kernel brings the 64 partial results of a row to one scale and
       adds them up.
  `reference` computes the same result with MLX's attention, one row at a time; the
  tests compare the two.

Tried and dropped: not reading the value of a token whose weight is under a billionth
  of the largest. It left the scores unchanged and saved no time (30.4 ms a step
  against 29.1 on 16 rows with real 8,000-token histories), because the keys must be
  read either way and reading values only on demand loses the overlap of the reads.

Use:     pool = PagePool(...); pool.write(layer, k, v); out = pool.attend(layer, q, active)
Changes: memory only.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

PAGE = 256
BLOCKS = 64      # each row's keys are split among this many groups of GPU threads (MLX's choice for this chip)
_FIRST = """
    constexpr int BD = 32;
    constexpr int PER = D / BD;
    constexpr int TK = PAGE / BLOCKS;                 // tokens of one page that belong to one group
    const uint simd_lid = thread_index_in_simdgroup;
    const int kv_head = threadgroup_position_in_grid.x;
    const int row = threadgroup_position_in_grid.y;
    const int block = threadgroup_position_in_grid.z;
    const int group = row * H + G * kv_head + thread_position_in_threadgroup.y;     // row and query head

    float qv[PER];
    float o[PER];
    const float scaling = scale[0];
    for (int j = 0; j < PER; j++) {
        qv[j] = scaling * float(q[group * D + simd_lid * PER + j]);
        o[j] = 0;
    }
    float max_score = -3.0e38f;
    float sum_exp = 0;
    const int pages = n_pages[row];
    const int in_buffer = n_buf[row];

    // This group takes the tokens at places block, block + BLOCKS, ... of the buffer and of every page:
    // the buffer first, then the pages from the newest back (recent tokens usually score highest, so the
    // running maximum settles early). One look at the page table per page; a page's TK tokens are read together.
    const ulong lane = block * D + simd_lid * PER;
    {
        const ulong base = (ulong(row) * KVH + kv_head) * PAGE * D + lane;
        for (int t = 0; block + t * BLOCKS < in_buffer; t++) {
            float score = 0;
            for (int j = 0; j < PER; j++) score += qv[j] * float(buf_k[base + t * BLOCKS * D + j]);
            score = simd_sum(score);
            const float new_max = max(max_score, score);
            const float keep = fast::exp(max_score - new_max);
            const float weight = fast::exp(score - new_max);
            max_score = new_max;
            sum_exp = sum_exp * keep + weight;
            for (int j = 0; j < PER; j++) o[j] = o[j] * keep + weight * float(buf_v[base + t * BLOCKS * D + j]);
        }
    }
    for (int unit = pages - 1; unit >= 0; unit--) {
        const ulong base = (ulong(table[row * MAXP + unit]) * KVH + kv_head) * PAGE * D + lane;
        const device T* kp = pool_k + base;
        const device T* vp = pool_v + base;
        T key[TK][PER];
        T val[TK][PER];
        for (int t = 0; t < TK; t++) {
            for (int j = 0; j < PER; j++) key[t][j] = kp[t * BLOCKS * D + j];
        }
        for (int t = 0; t < TK; t++) {
            for (int j = 0; j < PER; j++) val[t][j] = vp[t * BLOCKS * D + j];
        }
        for (int t = 0; t < TK; t++) {
            float score = 0;
            for (int j = 0; j < PER; j++) score += qv[j] * float(key[t][j]);
            score = simd_sum(score);
            const float new_max = max(max_score, score);
            const float keep = fast::exp(max_score - new_max);
            const float weight = fast::exp(score - new_max);
            max_score = new_max;
            sum_exp = sum_exp * keep + weight;
            for (int j = 0; j < PER; j++) o[j] = o[j] * keep + weight * float(val[t][j]);
        }
    }
    if (simd_lid == 0) {
        sums[group * BLOCKS + block] = sum_exp;
        tops[group * BLOCKS + block] = max_score;
    }
    for (int j = 0; j < PER; j++) {
        parts[(ulong(group) * BLOCKS + block) * D + simd_lid * PER + j] = o[j];
    }
"""
_SECOND = """
    const int d = thread_position_in_grid.x;
    const int group = thread_position_in_grid.y;
    float top = -3.0e38f;
    for (int b = 0; b < BLOCKS; b++) top = max(top, tops[group * BLOCKS + b]);
    float total = 0;
    float value = 0;
    for (int b = 0; b < BLOCKS; b++) {
        const float factor = fast::exp(tops[group * BLOCKS + b] - top);
        total += sums[group * BLOCKS + b] * factor;
        value += parts[(ulong(group) * BLOCKS + b) * D + d] * factor;
    }
    out[group * D + d] = T(total == 0 ? 0.0f : value / total);
"""
_kernels: dict[str, object] = {}


def paged_attention(q: mx.array, pool_k: mx.array, pool_v: mx.array, buf_k: mx.array, buf_v: mx.array, table: mx.array,
                    n_pages: mx.array, n_buf: mx.array, scale: float) -> mx.array:
    """Attention of one query per row and head over that row's pages and buffer. q is [rows, heads, dim]; returns the same shape."""
    if not _kernels:
        _kernels["first"] = mx.fast.metal_kernel(
            name="paged_attention_first", input_names=["q", "pool_k", "pool_v", "buf_k", "buf_v", "table", "n_pages", "n_buf", "scale"],
            output_names=["parts", "sums", "tops"], source=_FIRST)
        _kernels["second"] = mx.fast.metal_kernel(
            name="paged_attention_second", input_names=["parts", "sums", "tops"], output_names=["out"], source=_SECOND)
    rows, heads, dim = q.shape
    kv_heads = pool_k.shape[1]
    shared = heads // kv_heads                                    # query heads that read the same keys sit in one thread group
    parts, sums, tops = _kernels["first"](
        inputs=[q, pool_k, pool_v, buf_k, buf_v, table, n_pages, n_buf, mx.array([scale], dtype=mx.float32)],
        template=[("T", q.dtype), ("D", dim), ("H", heads), ("KVH", kv_heads), ("G", shared), ("PAGE", PAGE), ("MAXP", table.shape[1]),
                  ("BLOCKS", BLOCKS)],
        grid=(32 * kv_heads, shared * rows, BLOCKS), threadgroup=(32, shared, 1),
        output_shapes=[(rows, heads, BLOCKS, dim), (rows, heads, BLOCKS), (rows, heads, BLOCKS)], output_dtypes=[mx.float32] * 3)
    return _kernels["second"](inputs=[parts, sums, tops], template=[("T", q.dtype), ("D", dim), ("BLOCKS", BLOCKS)],
                              grid=(dim, rows * heads, 1), threadgroup=(min(dim, 256), 1, 1), output_shapes=[q.shape], output_dtypes=[q.dtype])[0]


@dataclass
class Held:
    """One row's history as it stood at one moment, kept without a slot."""

    pages: np.ndarray                                   # its full pages (shared with whoever else points at them)
    n_buf: int
    buffers: list[tuple[np.ndarray, np.ndarray]]        # per layer: keys and values of the n_buf newest tokens (copies)


class PagePool:
    """Keys and values of every row, for every attention layer, in pages (see the top of this file).

    slots: how many rows can run at once. pages: the size of the pool (256 tokens each).
    A row is addressed by its slot number.
    """

    def __init__(self, *, layers: int, slots: int, pages: int, kv_heads: int, head_dim: int, dtype: mx.Dtype,
                 max_row_tokens: int, use_kernel: bool = True) -> None:
        self.layers, self.slots, self.pages = layers, slots, pages
        self.use_kernel = use_kernel
        self.keys = [mx.zeros((pages, kv_heads, PAGE, head_dim), dtype) for _ in range(layers)]
        self.values = [mx.zeros((pages, kv_heads, PAGE, head_dim), dtype) for _ in range(layers)]
        self.buf_keys = [mx.zeros((slots, kv_heads, PAGE, head_dim), dtype) for _ in range(layers)]
        self.buf_values = [mx.zeros((slots, kv_heads, PAGE, head_dim), dtype) for _ in range(layers)]
        mx.eval(self.keys, self.values, self.buf_keys, self.buf_values)
        self.table = np.zeros((slots, -(-max_row_tokens // PAGE) + 1), np.int32)
        self.n_pages = np.zeros(slots, np.int32)
        self.n_buf = np.zeros(slots, np.int32)
        self.sharers = np.zeros(pages, np.int32)          # how many rows point at each page
        self.free = list(range(pages - 1, -1, -1))
        self._table_mx: mx.array | None = None

    # -- bookkeeping ---------------------------------------------------------------------------

    @property
    def bytes_per_page(self) -> int:
        return 2 * self.layers * self.keys[0][0].nbytes

    def tokens_held(self, slot: int) -> int:
        return int(self.n_pages[slot]) * PAGE + int(self.n_buf[slot])

    def _table(self) -> mx.array:
        if self._table_mx is None:
            self._table_mx = mx.array(self.table)
        return self._table_mx

    def _take_page(self) -> int:
        if not self.free:
            raise MemoryError("the page pool is full")
        page = self.free.pop()
        self.sharers[page] = 1
        return page

    def _give_back(self, pages: np.ndarray) -> None:
        for page in pages:
            self.sharers[page] -= 1
            if self.sharers[page] == 0:
                self.free.append(int(page))

    def release(self, slot: int) -> None:
        """A row is finished or set aside: give its pages back (a shared page stays until its last user leaves)."""
        self._give_back(self.table[slot, : self.n_pages[slot]])
        self.n_pages[slot] = self.n_buf[slot] = 0

    # -- one decode step -----------------------------------------------------------------------

    def write(self, layer: int, k: mx.array, v: mx.array) -> None:
        """Put the newest token's key and value of every slot ([slots, kv_heads, dim]) at the end of its buffer."""
        at = mx.broadcast_to(mx.array(self.n_buf)[:, None, None, None], (self.slots, k.shape[1], 1, k.shape[2]))
        dtype = self.buf_keys[layer].dtype
        self.buf_keys[layer] = mx.put_along_axis(self.buf_keys[layer], at, k[:, :, None, :].astype(dtype), axis=2)
        self.buf_values[layer] = mx.put_along_axis(self.buf_values[layer], at, v[:, :, None, :].astype(dtype), axis=2)

    def attend(self, layer: int, q: mx.array, active: np.ndarray, scale: float) -> mx.array:
        """Attention for the token just written with `write`. q is [slots, heads, dim]; slots that are not active get zeros."""
        n_pages = np.where(active, self.n_pages, 0).astype(np.int32)
        n_buf = np.where(active, self.n_buf + 1, 0).astype(np.int32)
        if not self.use_kernel:
            return self.reference(layer, q, n_pages, n_buf, scale)
        return paged_attention(q.astype(self.keys[layer].dtype), self.keys[layer], self.values[layer], self.buf_keys[layer],
                               self.buf_values[layer], self._table(), mx.array(n_pages), mx.array(n_buf), scale)

    def advance(self, active: np.ndarray) -> None:
        """After a step: every active row holds one more token; full buffers move into pages."""
        self.n_buf[active] += 1
        for slot in np.flatnonzero(self.n_buf == PAGE):
            page = self._take_page()
            for layer in range(self.layers):
                self.keys[layer][page] = self.buf_keys[layer][int(slot)]
                self.values[layer][page] = self.buf_values[layer][int(slot)]
            self.table[slot, self.n_pages[slot]] = page
            self.n_pages[slot] += 1
            self.n_buf[slot] = 0
            self._table_mx = None

    # -- whole rows ----------------------------------------------------------------------------

    def load(self, slot: int, history: list[tuple[mx.array, mx.array]]) -> None:
        """Give an empty slot a history computed elsewhere: per layer, keys and values of shape [1, kv_heads, tokens, dim]."""
        tokens = history[0][0].shape[2]
        full, rest = divmod(tokens, PAGE)
        pages = [self._take_page() for _ in range(full)]
        for layer, (k, v) in enumerate(history):
            if full:
                at = mx.array(pages)
                paged = lambda x: x[0, :, : full * PAGE].reshape(x.shape[1], full, PAGE, x.shape[3]).transpose(1, 0, 2, 3)
                self.keys[layer][at] = paged(k)
                self.values[layer][at] = paged(v)
            if rest:
                self.buf_keys[layer][slot, :, :rest] = k[0, :, full * PAGE:]
                self.buf_values[layer][slot, :, :rest] = v[0, :, full * PAGE:]
        self.table[slot, :full] = pages
        self.n_pages[slot], self.n_buf[slot] = full, rest
        self._table_mx = None

    def hold(self, slot: int) -> Held:
        """Keep a row's history as it stands now, without a slot. The row itself carries on.

        The pages are shared (they count one more user until `adopt` or `drop`); the buffer is copied,
        in 32-bit, which holds every 16-bit value exactly.
        """
        pages, n_buf = self.table[slot, : self.n_pages[slot]].copy(), int(self.n_buf[slot])
        self.sharers[pages] += 1
        copy = lambda buffer: np.array(buffer[slot, :, :n_buf].astype(mx.float32))
        return Held(pages, n_buf, [(copy(k), copy(v)) for k, v in zip(self.buf_keys, self.buf_values)])

    def adopt(self, held: Held, slot: int) -> None:
        """Give an empty slot a held history. The hold is used up: its share of the pages passes to the slot."""
        n = len(held.pages)
        self.table[slot, :n] = held.pages
        self.n_pages[slot], self.n_buf[slot] = n, held.n_buf
        if held.n_buf:
            for layer, (k, v) in enumerate(held.buffers):
                dtype = self.buf_keys[layer].dtype
                self.buf_keys[layer][slot, :, : held.n_buf] = mx.array(k).astype(dtype)
                self.buf_values[layer][slot, :, : held.n_buf] = mx.array(v).astype(dtype)
        self._table_mx = None

    def drop(self, held: Held) -> None:
        """Give up a held history that no slot will take."""
        self._give_back(held.pages)

    def move(self, src: int, dst: int) -> None:
        """Move a row from one slot to an empty one. Its pages stay where they are; only the buffer is copied."""
        n = self.n_pages[src]
        self.table[dst, :n] = self.table[src, :n]
        self.n_pages[dst], self.n_buf[dst] = n, self.n_buf[src]
        self.n_pages[src] = self.n_buf[src] = 0
        for layer in range(self.layers):
            self.buf_keys[layer][dst] = self.buf_keys[layer][src]
            self.buf_values[layer][dst] = self.buf_values[layer][src]
        self._table_mx = None

    def shrink(self, slots: int) -> None:
        """Keep only the first `slots` slots. The slots given up must be empty."""
        self.buf_keys = [mx.contiguous(b[:slots]) for b in self.buf_keys]
        self.buf_values = [mx.contiguous(b[:slots]) for b in self.buf_values]
        self.table, self.n_pages, self.n_buf = self.table[:slots].copy(), self.n_pages[:slots].copy(), self.n_buf[:slots].copy()
        self.slots = slots
        self._table_mx = None

    def gather(self, layer: int, slot: int, n_pages: int | None = None, n_buf: int | None = None) -> tuple[mx.array, mx.array]:
        """One row's keys and values as plain arrays [1, kv_heads, tokens, dim] (for checks and for the plain-MLX path)."""
        n_pages = int(self.n_pages[slot]) if n_pages is None else n_pages
        n_buf = int(self.n_buf[slot]) if n_buf is None else n_buf

        def whole(pool: mx.array, buffer: mx.array) -> mx.array:
            parts = []
            if n_pages:
                paged = pool[mx.array(self.table[slot, :n_pages])]                      # [pages, kv_heads, PAGE, dim]
                parts.append(paged.transpose(1, 0, 2, 3).reshape(paged.shape[1], n_pages * PAGE, paged.shape[3]))
            parts.append(buffer[slot, :, :n_buf])
            return mx.concatenate(parts, axis=1)[None]

        return whole(self.keys[layer], self.buf_keys[layer]), whole(self.values[layer], self.buf_values[layer])

    def reference(self, layer: int, q: mx.array, n_pages: np.ndarray, n_buf: np.ndarray, scale: float) -> mx.array:
        """The same result as the kernel, from MLX's own attention, one row at a time."""
        out = []
        for slot in range(self.slots):
            if n_pages[slot] == 0 and n_buf[slot] == 0:
                out.append(mx.zeros_like(q[slot]))
                continue
            k, v = self.gather(layer, slot, int(n_pages[slot]), int(n_buf[slot]))
            out.append(mx.fast.scaled_dot_product_attention(q[slot][None, :, None, :], k, v, scale=scale, mask=None)[0, :, 0, :])
        return mx.stack(out)
