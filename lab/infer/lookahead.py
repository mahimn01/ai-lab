"""Write several tokens per pass for one prompt, by guessing ahead and checking the guesses together.

Why
  For one prompt, a model's speed on this Mac is capped by reading all its weights from
  memory once per token. The lab's GPU kernel (lab/kernels/qmm.py) makes extra rows in
  the same pass cheap: about a fifth of the first. So if we can guess the next few
  tokens, the model can check all of them in one pass: every correct guess is a token
  that did not need its own pass. The text is the same as normal greedy decoding (the
  model's own choice is always what gets written), up to rounding in the kernel.

Where the guesses come from ("prompt lookup")
  Text that copies from earlier text: a quoted passage, edited code, a name used
  again. If the last few tokens written also appear earlier in the prompt or the
  output, the tokens that followed them there are the guess. This costs nothing. When
  there is no match, the decoder takes a normal one-token step, so it cannot be slower
  than plain decoding by more than the checks that fail.

After a wrong guess the model's memory of the text (its cache) has to go back
  rewind   Qwen3, Llama, Gemma: mlx-lm can simply drop the last tokens from the cache.
           A wrong guess only wastes its own cheap rows.
  redo     Qwen3.5 family: part of its cache is a running summary that cannot be
           rewound. The decoder saves that part before each check; after a wrong guess
           it restores it and runs the accepted tokens again. A wrong guess costs one
           extra pass, so here the decoder starts with short guesses and stops guessing
           for a while after each miss.

Limits
  Greedy decoding only (always the most likely token).

Use
  from lab.infer.lookahead import generate
  qmm.install(model)                       # without the kernel the checks cost more than they save
  result = generate(model, tokenizer, prompt_tokens, max_tokens=256)
Read-only: nothing is written to disk.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import make_prompt_cache

MAX_GUESS = 7  # with the token being checked that makes 8 rows: one pass of the kernel
MAX_PAUSE = 32  # redo mode: longest stretch of tokens without guessing after repeated misses


@dataclass
class LookaheadResult:
    text: str
    token_ids: list[int]
    tokens_per_s: float  # new tokens per second of writing time (the prompt is read before the clock starts)
    passes: int  # model passes used (including redone ones); plain decoding needs one per token
    guessed: int  # tokens offered as guesses
    accepted: int  # guesses the model agreed with
    wall_s: float


class _Lookup:
    """Remembers where every run of `n` tokens last ended, so a repeat is found in one dictionary read."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.tokens: list[int] = []
        self.last_end: dict[tuple[int, ...], int] = {}

    def extend(self, new: list[int]) -> None:
        for token in new:
            if len(self.tokens) >= self.n:  # index the run that ended just before this token
                self.last_end[tuple(self.tokens[-self.n:])] = len(self.tokens)
            self.tokens.append(token)

    def guess(self, limit: int) -> list[int]:
        """What followed the last time the final `n` tokens appeared (empty if they never did)."""
        if len(self.tokens) < self.n or limit <= 0:
            return []
        end = self.last_end.get(tuple(self.tokens[-self.n:]))
        return self.tokens[end:end + limit] if end is not None else []


def _copy_state(state):
    """A cache state with its own lists (the arrays inside are never changed in place, so they are shared)."""
    if isinstance(state, (list, tuple)):
        return type(state)(_copy_state(v) for v in state)
    return state


def generate(model: nn.Module, tokenizer, prompt: list[int], *, max_tokens: int = 256, ngram: int = 3,
             max_guess: int = MAX_GUESS, lookahead: bool = True) -> LookaheadResult:
    """Greedy decoding with prompt-lookup guesses. lookahead=False gives plain one-token-per-pass decoding."""
    cache = make_prompt_cache(model)
    fixed = [c for c in cache if not c.is_trimmable()]  # the parts that cannot be rewound (empty for most models)
    stop = set(tokenizer.eos_token_ids)
    seen = _Lookup(ngram)
    seen.extend(prompt)

    # Read the prompt; the model's first choice is the first new token.
    nxt = mx.argmax(model(mx.array(prompt)[None], cache=cache)[0, -1]).item()
    out: list[int] = []
    passes = guessed = accepted = misses = pause = 0
    limit = 2 if fixed else max_guess  # redo mode starts carefully: a miss costs a whole pass there
    t0 = time.perf_counter()
    while len(out) < max_tokens and nxt not in stop:
        out.append(nxt)
        seen.extend([nxt])
        guess = seen.guess(min(limit, max_tokens - len(out))) if lookahead and not pause else []
        pause = max(0, pause - 1)
        saved = [_copy_state(c.state) for c in fixed] if guess else []
        # One pass over the new token and the guesses: choices[i] is the model's pick after input i.
        choices = mx.argmax(model(mx.array([nxt] + guess)[None], cache=cache)[0], axis=-1).tolist()
        passes += 1
        good = 0
        while good < len(guess) and guess[good] == choices[good] and guess[good] not in stop:
            good += 1
        if good < len(guess) and not fixed:  # rewind: forget the guesses the model did not agree with
            for c in cache:
                c.trim(len(guess) - good)
        elif good < len(guess):  # redo: undo the whole pass, then run the token and the accepted guesses again
            for c in cache:
                if c.is_trimmable():
                    c.trim(len(guess) + 1)
            for c, state in zip(fixed, saved):
                c.state = state
            mx.eval(model(mx.array([nxt] + guess[:good])[None], cache=cache))
            passes += 1
        out.extend(guess[:good])
        seen.extend(guess[:good])
        nxt = choices[good]
        guessed += len(guess)
        accepted += good
        if guess and good == len(guess):
            misses, limit = 0, (min(max_guess, 2 * limit) if fixed else max_guess)
        elif guess:
            misses += 1
            limit = 2 if fixed else max(2, good + 1)
            pause = min(MAX_PAUSE, 4 * misses) if fixed else 0
    wall = time.perf_counter() - t0
    return LookaheadResult(tokenizer.decode(out), out, len(out) / wall if wall > 0 else 0.0, passes, guessed, accepted, wall)
