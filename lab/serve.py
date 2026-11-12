"""Serve a model to other programs, with the lab's GPU paths switched on.

What it does
  Starts mlx-lm's own OpenAI-compatible server (same options, same behaviour) and, every
  time the server loads a model, switches that model's quantized layers to the lab's
  paths (lab/kernels/qmm.py). Requests that are decoded 2 to 18 at a time then run 1.2x
  to 2.6x faster on this Mac (SPEED-RESULTS.md section 9) and long prompts are read 1.2x
  to 1.33x faster (section 11). One short request at a time behaves exactly as with the
  plain server.

  --float16 also converts each model from bfloat16 to float16 after checking that it is
  safe on sample prompts (lab/infer/float16.py, section 16): faster again for prompt
  reading and for 8 or more requests at once. If the check fails the model is served as
  it is and the reason is printed.

Run:     uv run python -m lab.serve --model mlx-community/Qwen3.5-9B-4bit --port 8642 --decode-concurrency 8 [--float16]
         (every option of `mlx_lm.server` works; see `uv run mlx_lm.server --help`)
Changes: nothing on disk. Listens on the port you give it.
"""

from __future__ import annotations

import sys

import mlx_lm.server as server

from lab.infer import float16
from lab.kernels import qmm

_load = server.load
_options = {"float16": False}


def _load_with_kernel(*args, **kwargs):
    loaded = _load(*args, **kwargs)  # (model, tokenizer) or (model, tokenizer, config)
    model, tokenizer = loaded[0], loaded[1]
    if _options["float16"]:
        try:
            report = float16.convert(model, tokenizer)
            print(f"float16: converted; the largest value seen on the sample prompts was {report['largest_value']:,.0f}, "
                  f"{report['headroom']:.0f}x below the limit", flush=True)
        except Exception as problem:  # convert leaves the model as it was on any error; serve it that way
            print(f"float16: not converted ({type(problem).__name__}: {problem})", flush=True)
    switched = qmm.install(model)
    print(f"lab kernel: switched {switched} quantized layers of the model just loaded", flush=True)
    return loaded


def main() -> None:
    if "--float16" in sys.argv:  # the lab's own option: take it out before mlx-lm reads the rest
        sys.argv.remove("--float16")
        _options["float16"] = True
    server.load = _load_with_kernel  # the server looks this name up each time it loads a model
    server.main()


if __name__ == "__main__":
    main()
