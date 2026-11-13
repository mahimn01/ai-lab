"""The ledger: an append-only record of every finished row, so a run can stop and restart anywhere.

Why
  The Phase 0 runner wrote its results once, at the end of a condition, so a crash lost
  the whole condition, and it kept only decoded text (turning that text back into tokens
  later was not exact). Here every row is written the moment it finishes, with its token
  ids, and a restarted run skips the rows already there.

Files in a ledger folder
  rows.jsonl    one JSON line per finished row: its key, what defined it, why it stopped,
                counts and timings, and where its tokens are
  tokens.bin    the token ids of all rows, 4 bytes each, back to back
  meta.json     what the run was (model, settings, software versions, the plan's hash)

Use:     ledger = Ledger("runs/phase1/edited-32k")
         if key not in ledger.done(): ...; ledger.append(record, tokens)
Changes: only the folder it is given.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def row_key(spec: dict) -> str:
    """A short, stable name for a row: a hash of everything that defines it."""
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]


def file_hash(path: str | Path) -> str:
    """sha256 of a file (used to tie a run to the plan written before it)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Ledger:
    def __init__(self, folder: str | Path) -> None:
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.rows_path = self.folder / "rows.jsonl"
        self.tokens_path = self.folder / "tokens.bin"

    def records(self) -> list[dict]:
        """Every complete row record. A line cut off by a crash is skipped."""
        if not self.rows_path.exists():
            return []
        out = []
        for line in self.rows_path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def done(self) -> set[str]:
        return {r["key"] for r in self.records()}

    def tokens(self, record: dict) -> list[int]:
        """The token ids of one row."""
        with open(self.tokens_path, "rb") as f:
            f.seek(4 * record["token_offset"])
            return np.frombuffer(f.read(4 * record["token_count"]), dtype="<i4").tolist()

    def append(self, record: dict, tokens: list[int]) -> None:
        """Write one finished row: its tokens first, then its line, each forced to disk."""
        with open(self.tokens_path, "ab") as f:
            size = f.seek(0, os.SEEK_END)
            if size % 4:  # a write cut off by a crash: drop the partial number
                f.truncate(size - size % 4)
                size -= size % 4
            f.write(np.asarray(tokens, dtype="<i4").tobytes())
            f.flush()
            os.fsync(f.fileno())
        with open(self.rows_path, "ab+") as f:
            end = f.seek(0, os.SEEK_END)
            if end:
                f.seek(end - 1)
                if f.read(1) != b"\n":  # a line cut off by a crash: drop it, so the next line starts clean
                    f.seek(0)
                    f.truncate(f.read().rfind(b"\n") + 1)
            f.write((json.dumps({**record, "token_offset": size // 4, "token_count": len(tokens)}) + "\n").encode())
            f.flush()
            os.fsync(f.fileno())

    def write_meta(self, meta: dict) -> None:
        (self.folder / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    def meta(self) -> dict:
        path = self.folder / "meta.json"
        return json.loads(path.read_text()) if path.exists() else {}
