"""Serve a model to other programs, with the lab's multi-row GPU kernel switched on.

What it does
  Starts mlx-lm's own OpenAI-compatible server (same options, same behaviour) and, every
  time the server loads a model, switches that model's quantized layers to the lab kernel
  (lab/kernels/qmm.py). Requests that are decoded 2 to 18 at a time then run 1.2x to
  2.6x faster on this Mac (SPEED-RESULTS.md section 9). One request at a time, or 19
  and more, behave exactly as with the plain server.

Run:     uv run python -m lab.serve --model mlx-community/Qwen3.5-9B-4bit --port 8642 --decode-concurrency 8
         (every option of `mlx_lm.server` works; see `uv run mlx_lm.server --help`)
Changes: nothing on disk. Listens on the port you give it.
"""

from __future__ import annotations

import mlx_lm.server as server

from lab.kernels import qmm

_load = server.load


def _load_with_kernel(*args, **kwargs):
    loaded = _load(*args, **kwargs)  # (model, tokenizer) or (model, tokenizer, config)
    switched = qmm.install(loaded[0])
    print(f"lab kernel: switched {switched} quantized layers of the model just loaded", flush=True)
    return loaded


def main() -> None:
    server.load = _load_with_kernel  # the server looks this name up each time it loads a model
    server.main()


if __name__ == "__main__":
    main()
