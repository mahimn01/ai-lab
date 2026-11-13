"""lab.gen: many long generations at once, one row at a time (design: docs/design-row-engine.md).

engine.py   Row (what defines one generation) and Engine (runs rows on this Mac through MLX)
kv.py       the attention history of all rows in pages, and the kernel that reads it
sampler.py  the next token for every row, each with its own settings and random stream
loops.py    loop marks while a row is written, and the opt-in rule for stopping on a loop
ledger.py   the append-only record of finished rows that makes a run restartable

Everything a study needs besides the engine itself (rows, the ledger, the loop test, the
statistics in lab/stats) does not depend on the Mac. Another machine needs one thing: an
object with the engine's `run(rows, ledger, brancher=..., enough=...)` that returns the same
records. On NVIDIA GPUs that could be a thin client of a vLLM server; it is not built here.
"""
