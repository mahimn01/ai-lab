"""Diagnostic: where does the time go in one batched decoding step?

Why
  The baseline showed an odd shape: for the 9B model one prompt took 18 ms per token,
  but a batch of 8 took 108 ms per step and a batch of 32 only 124 ms. So going from 1
  to 8 prompts barely helps, while 8 to 32 is almost free. This script separates the
  possible causes by timing three ways of writing the same tokens:
    single     mlx-lm's one-prompt generator (stream_generate)
    generator  mlx-lm's BatchGenerator (what batch_generate and the server use)
    raw        a bare loop: call the model on the batch, take the most likely token,
               repeat. No padding, no stop checks, no per-prompt bookkeeping.
  If "raw" is much faster than "generator" at the same batch size, the cost is in the
  generator's bookkeeping and can be avoided in the lab's own code. If "raw" is just as
  slow, the cost is inside the model's GPU kernels.

All prompts are the same length (64 random tokens), so nothing needs padding.

Run:     uv run python -m bench.batch_overhead --model LABEL=PATH_OR_REPO
Changes: writes bench/results/batch_overhead.json. Nothing else.
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
from mlx_lm import batch_generate, load, stream_generate
from mlx_lm.models.cache import make_prompt_cache

from bench.common import environment, free_gpu_memory, save
from lab.infer.batch import CACHE_LIMIT

PROMPT_TOKENS, STEPS = 64, 64
BATCHES = (1, 2, 4, 8, 16, 32, 64)


def raw_ms_per_step(model, prompts: list[list[int]]) -> float:
    cache = make_prompt_cache(model)
    y = mx.argmax(model(mx.array(prompts), cache=cache)[:, -1, :], axis=-1)
    mx.eval(y)
    for _ in range(4):  # warm-up for the one-token-per-prompt shape
        y = mx.argmax(model(y[:, None], cache=cache)[:, -1, :], axis=-1)
        mx.eval(y)
    t0 = time.perf_counter()
    for _ in range(STEPS):
        y = mx.argmax(model(y[:, None], cache=cache)[:, -1, :], axis=-1)
        mx.eval(y)
    return (time.perf_counter() - t0) / STEPS * 1000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", action="append", required=True, metavar="LABEL=PATH")
    ap.add_argument("--save-logits", metavar="FILE.npy",
                    help="also save the model's scores for 4 prompts, to compare two settings for equal results")
    args = ap.parse_args()
    mx.set_cache_limit(CACHE_LIMIT)
    for spec in args.model:
        label, _, path = spec.partition("=")
        model, tokenizer, config = load(path, return_config=True)
        tokenizer._eos_token_ids = {}  # never stop early
        vocab = config.get("vocab_size") or config["text_config"]["vocab_size"]
        mx.random.seed(0)
        prompts = mx.random.randint(0, vocab, (max(BATCHES), PROMPT_TOKENS)).tolist()

        if args.save_logits:
            import numpy as np

            np.save(args.save_logits, np.array(model(mx.array(prompts[:4])).astype(mx.float32)))
        for _ in range(2):  # second pass is the timed one
            for r in stream_generate(model, tokenizer, prompts[0], max_tokens=STEPS):
                pass
        entry = {"environment": environment(), "single_ms_per_token": 1000 / r.generation_tps, "batches": {}}
        print(f"{label}: single {entry['single_ms_per_token']:.1f} ms per token", flush=True)
        for b in BATCHES:
            row = {}
            try:
                for _ in range(2):
                    s = batch_generate(model, tokenizer, prompts[:b], max_tokens=STEPS, completion_batch_size=b,
                                       prefill_batch_size=b).stats
                row["generator_ms_per_step"] = s.decode_time / s.generation_steps * 1000
                row["generator_wall_ms_per_step"] = s.generation_time / s.generation_steps * 1000
                row["raw_ms_per_step"] = raw_ms_per_step(model, prompts[:b])
                row["raw_tokens_per_s"] = b / row["raw_ms_per_step"] * 1000
                row["generator_tokens_per_s"] = s.generation_tps
                print(f"  batch {b:3d}: generator {row['generator_wall_ms_per_step']:7.1f} ms/step "
                      f"({row['generator_tokens_per_s']:7.1f} tok/s)   raw {row['raw_ms_per_step']:7.1f} ms/step "
                      f"({row['raw_tokens_per_s']:7.1f} tok/s)", flush=True)
            except Exception as e:  # keep the other batch sizes if one fails (for example, out of memory)
                row["error"] = f"{type(e).__name__}: {e}"[:300]
                print(f"  batch {b:3d}: FAILED {row['error']}", flush=True)
            entry["batches"][str(b)] = row
            save("batch_overhead", label, entry)
        del model
        free_gpu_memory()


if __name__ == "__main__":
    main()
