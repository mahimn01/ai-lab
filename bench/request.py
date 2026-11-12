"""Speed experiment 16 (part): one request, end to end, with the lab's paths and float16 off and on.

Background
  Writing new text for one prompt is not made faster by anything in this lab: each token
  waits on reading all the weights from memory. Reading the prompt is a different kind
  of work (many rows at once) and did get faster (SPEED-RESULTS.md sections 11 and 16).
  A real request does both, so how much it gains depends on how much there is to read.

What it measures
  For one model: the seconds from sending a prompt to the last new token, for prompts of
  512, 2,048 and 8,192 tokens of the lab's held-out text and 128 new tokens (greedy
  decoding through mlx-lm's own generator), in four set-ups:
    as shipped       bfloat16, MLX's kernels
    lab paths        bfloat16, qmm.install(model)
    float16          float16.convert(model), MLX's kernels
    float16 + lab    both

Run:     uv run python -m bench.request --model LABEL=PATH_OR_REPO
Changes: writes bench/results/request.json. Nothing else.
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
from mlx_lm import load, stream_generate

from bench.common import environment, free_gpu_memory, save
from bench.quant_frontier import held_out_text
from lab.infer import float16
from lab.infer.batch import CACHE_LIMIT
from lab.kernels import qmm

PROMPT_TOKENS = (512, 2048, 8192)
NEW_TOKENS = 128


def request(model, tokenizer, prompt: list[int]) -> dict:
    """Answer one prompt. Returns the seconds it took in all, and mlx-lm's own reading and writing speeds."""
    t0 = time.perf_counter()
    for response in stream_generate(model, tokenizer, prompt, max_tokens=NEW_TOKENS):
        pass
    return {"seconds": time.perf_counter() - t0, "read_tokens_per_s": response.prompt_tps,
            "write_tokens_per_s": response.generation_tps, "new_tokens": response.generation_tokens}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", action="append", required=True, metavar="LABEL=PATH")
    args = ap.parse_args()
    mx.set_cache_limit(CACHE_LIMIT)
    text = held_out_text()
    for spec in args.model:
        label, _, path = spec.partition("=")
        model, tokenizer = load(path)
        ids = tokenizer.encode(text[:400_000])
        entry = {"environment": environment(), "path": path, "new_tokens": NEW_TOKENS, "prompts": {}}
        request(model, tokenizer, ids[:64])  # warm-up
        for name in ("as shipped", "lab paths", "float16", "float16 + lab"):
            if name == "float16":
                entry["check"] = float16.convert(model, prompts=[ids[:512]] + float16.sample_prompts(tokenizer))
            if "lab" in name:
                qmm.install(model)
            for length in PROMPT_TOKENS:
                request(model, tokenizer, ids[:length])  # once unmeasured: the first use of a shape compiles GPU code
                entry["prompts"].setdefault(str(length), {})[name] = request(model, tokenizer, ids[:length])
            qmm.uninstall(model)
        save("request", label, entry)
        for length, runs in entry["prompts"].items():
            base = runs["as shipped"]["seconds"]
            print(f"{label}, {int(length):5,d}-token prompt + {NEW_TOKENS} new tokens: " + "   ".join(
                f"{name} {r['seconds']:5.2f} s ({base / r['seconds']:.2f}x)" for name, r in runs.items()), flush=True)
        del model
        free_gpu_memory()


if __name__ == "__main__":
    main()
