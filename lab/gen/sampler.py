"""Pick the next token for every row of a batch at once, each row with its own settings and its own random numbers.

The rule (the neural-harness study's, in its order)
  1. presence penalty: subtract the row's penalty from the raw score of every token the row
     has already written (once per token, however often it appeared; the prompt is not
     counted). The caller keeps this as one table per row: the penalty where a token was
     written, 0 elsewhere.
  2. divide by the temperature
  3. keep the `top_k` highest scores
  4. turn them into probabilities, sort from most to least likely, and keep a token while
     the probability of the tokens before it adds up to less than `top_p`
  5. draw one of the kept tokens in proportion to its probability
  A temperature of 0 means "always the highest score" (after the penalty).

Two things are done differently from the usual code, with the same result
  Top-k without sorting the vocabulary. Sorting 248,320 scores per row took 5.0 ms for 32
    rows on the M1 Max. Here the scores are cut into blocks of 256, and only the top_k
    blocks with the largest maximum are sorted: the top_k largest scores must lie in them
    (a block outside them has a maximum no larger than top_k other scores). 0.9 ms.
  One random number per row per step, from a stream made from the row's own seed. mlx-lm
    draws once for the whole batch, so what a row writes depends on its neighbours. Here it
    does not, which is what lets a row be restarted, or a run be resumed.

Use:     tokens = next_tokens(scores, penalties, u, temperature=t, top_k=k, top_p=p)
Changes: nothing. Read-only.
"""

from __future__ import annotations

import mlx.core as mx

BLOCK = 256
STREAM_BLOCK = 4096


def top_values(scores: mx.array, k: int) -> tuple[mx.array, mx.array]:
    """The k largest scores of each row, largest first, and their positions. scores is [rows, vocabulary]."""
    rows, vocab = scores.shape
    blocks = -(-vocab // BLOCK)
    if blocks <= k:
        order = mx.argsort(-scores, axis=-1)[:, :k]
        return mx.take_along_axis(scores, order, axis=-1), order
    x = mx.pad(scores, [(0, 0), (0, blocks * BLOCK - vocab)], constant_values=-float("inf")).reshape(rows, blocks, BLOCK)
    best = mx.argpartition(-x.max(axis=-1), kth=k - 1, axis=-1)[:, :k]      # the k blocks with the largest maximum
    candidates = mx.take_along_axis(x, best[:, :, None], axis=1).reshape(rows, k * BLOCK)
    order = mx.argsort(-candidates, axis=-1)[:, :k]
    values = mx.take_along_axis(candidates, order, axis=-1)
    positions = mx.take_along_axis(best, order // BLOCK, axis=-1) * BLOCK + order % BLOCK
    return values, positions


def kept_probabilities(scores: mx.array, penalties: mx.array, *, temperature: mx.array, top_k: mx.array,
                       top_p: mx.array) -> tuple[mx.array, mx.array]:
    """Steps 1 to 4: for each row, the candidate tokens (most likely first) and their probabilities (0 if dropped).

    scores [rows, vocabulary] in any float type; penalties [rows, vocabulary] in 32-bit, the row's
    penalty where it has written the token and 0 elsewhere; the three settings are [rows]. The
    probabilities of a row are not rescaled to sum to 1.
    """
    x = scores.astype(mx.float32) - penalties
    x = x / mx.where(temperature > 0, temperature, 1.0)[:, None]
    widest = int(top_k.max().item())
    values, tokens = top_values(x, widest)
    values = mx.where(mx.arange(widest)[None, :] < top_k[:, None], values, -float("inf"))
    probs = mx.softmax(values, axis=-1)
    before = mx.cumsum(probs, axis=-1) - probs
    keep = (before < top_p[:, None]) | (mx.arange(widest)[None, :] == 0)      # the most likely token always stays
    return tokens, mx.where(keep, probs, 0.0)


def next_tokens(scores: mx.array, penalties: mx.array, u: mx.array, *, temperature: mx.array, top_k: mx.array,
                top_p: mx.array) -> mx.array:
    """One token per row. u is one uniform random number in [0, 1) per row."""
    tokens, probs = kept_probabilities(scores, penalties, temperature=temperature, top_k=top_k, top_p=top_p)
    cumulative = mx.cumsum(probs, axis=-1)
    target = u[:, None] * cumulative[:, -1:]
    kept = (probs > 0).sum(axis=-1)
    choice = mx.minimum((cumulative <= target).sum(axis=-1), kept - 1)   # the first token whose running total passes u
    choice = mx.where(temperature > 0, choice, 0)                        # temperature 0: the most likely one
    return mx.take_along_axis(tokens, choice[:, None], axis=-1).squeeze(-1)


def row_stream(seed: int, length: int) -> mx.array:
    """The uniform random numbers of one row (seed below 2**32), one per token it may write.

    Made in blocks of a fixed size, each from its own key, so the numbers do not depend on `length`:
    a row with a smaller cap draws the same numbers as far as it goes.
    """
    blocks = [mx.random.uniform(shape=(STREAM_BLOCK,), key=mx.random.key(seed * 65536 + i)) for i in range(-(-length // STREAM_BLOCK))]
    return mx.concatenate(blocks)[:length]
