"""Watching for loops while a row is being written, and the rule for stopping one.

The test
  The neural-harness study's own: take the last 512 tokens, decode them, compress the
  text with zlib; the window is "looping" when it shrinks below a fifth of its size.
  It is checked every 128 tokens. `LoopWatch` keeps, per row, how many windows in a row
  were looping, where the first loop began and the longest stretch so far.

Watching is not stopping
  Recording loops changes nothing. Stopping a row after N looping tokens does: on the
  one 32,768-token run on record, rows escaped loops thousands of tokens long and then
  answered correctly (docs/design-row-engine.md, finding 7). So the stopping rule is
  opt-in, a share of rows chosen by `audited` runs to the full budget anyway, and
  `stop_table` counts on full-length transcripts how many grades each N would change.

Use:     watch = LoopWatch(); watch.update(tokens, decode) after each new token
Changes: nothing. Read-only.
"""

from __future__ import annotations

import hashlib
import zlib
from dataclasses import dataclass
from typing import Callable, Iterable

WINDOW, STEP, RATIO = 512, 128, 0.2


def looping(text: str) -> bool:
    """Does this text compress below RATIO of its size?"""
    raw = text.encode()
    return bool(raw) and len(zlib.compress(raw, 9)) / len(raw) < RATIO


def ends_in_loop(text: str) -> bool:
    """The study's end-of-run measure: the last 1,000, 2,000 or 4,000 characters compress below 0.2."""
    return any(looping(text[-n:]) for n in (1000, 2000, 4000))


def audited(key: str, share: float) -> bool:
    """Is this row one of the `share` of rows that always run to the full budget? Decided by its key alone."""
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 2**32 < share


@dataclass
class LoopWatch:
    """Loop marks for one row. Call `update` with the row's generated tokens after each new token."""

    run: int = 0                    # looping windows in a row right now
    first_onset: int | None = None  # tokens written when the first looping window began
    longest: int = 0                # the longest stretch of looping tokens so far
    windows: int = 0                # windows checked
    looping_windows: int = 0

    def update(self, tokens: list[int], decode: Callable[[list[int]], str]) -> None:
        n = len(tokens)
        if n < WINDOW or n % STEP:
            return
        self.windows += 1
        if looping(decode(tokens[-WINDOW:])):
            self.run += 1
            self.looping_windows += 1
            if self.first_onset is None:
                self.first_onset = n - WINDOW
            self.longest = max(self.longest, self.looping_tokens)
        else:
            self.run = 0

    @property
    def looping_tokens(self) -> int:
        """How many tokens the current loop has lasted (0 when not looping)."""
        return WINDOW + (self.run - 1) * STEP if self.run else 0

    def summary(self) -> dict:
        return {"loop_first_onset": self.first_onset, "loop_longest": self.longest,
                "loop_windows": self.looping_windows, "windows_checked": self.windows}


def window_marks(tokens: list[int], decode: Callable[[list[int]], str]) -> list[bool]:
    """The loop test for every window of a finished row: windows end at 512, 640, 768, ... tokens."""
    return [looping(decode(tokens[n - WINDOW:n])) for n in range(WINDOW, len(tokens) + 1, STEP)]


def stop_point(tokens: list[int], decode: Callable[[list[int]], str], limit: int, marks: list[bool] | None = None) -> int | None:
    """The token count at which a row would be stopped by 'stop after `limit` looping tokens', or None."""
    run = 0
    for i, mark in enumerate(window_marks(tokens, decode) if marks is None else marks):
        run = run + 1 if mark else 0
        if run and WINDOW + (run - 1) * STEP >= limit:
            return WINDOW + i * STEP
    return None


def stop_table(rows: Iterable[tuple[list[int], bool]], decode: Callable[[list[int]], str],
               graded: Callable[[int, int], bool], limits: tuple[int, ...] = (512, 896, 1408, 2432, 4480, 8576)) -> list[dict]:
    """What each stopping limit would have done to full-length rows.

    rows:    (generated tokens, was the full row graded correct) per row
    graded:  graded(row index, token count) -> would the row cut at that count be correct?
    Returns, per limit: rows stopped, share of all tokens saved, correct grades lost, grades gained.
    """
    rows = list(rows)
    total = sum(len(t) for t, _ in rows)
    marks = [window_marks(tokens, decode) for tokens, _ in rows]
    table = []
    for limit in limits:
        out = {"limit": limit, "stopped": 0, "saved": 0, "lost": 0, "gained": 0}
        for i, (tokens, correct) in enumerate(rows):
            cut = stop_point(tokens, decode, limit, marks[i])
            if cut is None or cut >= len(tokens):
                continue
            early = graded(i, cut)
            out["stopped"] += 1
            out["saved"] += len(tokens) - cut
            out["lost"] += correct and not early
            out["gained"] += early and not correct
        out["saved_share"] = out.pop("saved") / max(total, 1)
        table.append(out)
    return table
