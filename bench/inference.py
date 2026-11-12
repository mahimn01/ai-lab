"""Inference baseline: how fast does a model read a prompt and write tokens on this Mac?

What it does
  For each model, measures at several batch sizes (how many prompts run at once):
    load time        seconds to get the weights from disk into memory
    prompt speed     tokens/s while reading a 512-token prompt ("prefill")
    generation speed tokens/s while writing 256 new tokens, all prompts added together
    peak memory      the most memory MLX used, in GB
  The method is the one in mlx-lm's own `mlx_lm.benchmark`: random prompt tokens, the
  stop token disabled so every run writes the full length, one warm-up, then timed trials.

Run:     uv run python -m bench.inference --model qwen3.5-0.8b-4bit=mlx-community/Qwen3.5-0.8B-4bit
         (repeat --model LABEL=PATH_OR_REPO; add --batch-sizes 1,8,32 --trials 3)
Changes: writes bench/results/inference.json (one entry per model label). Downloads the
         model to the Hugging Face cache if it is not there yet. Nothing else.
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import batch_generate, load, stream_generate

from bench.common import environment, free_gpu_memory, mean_sd, save
from lab.infer.batch import CACHE_LIMIT

GB = 2**30


def bench_model(path: str, batch_sizes: list[int], prompt_tokens: int, gen_tokens: int, trials: int,
                prefill_batch: int = 8) -> dict:
    t0 = time.perf_counter()
    model, tokenizer, config = load(path, return_config=True)
    load_s = time.perf_counter() - t0
    weights_gb = sum(v.nbytes for _, v in tree_flatten(model.parameters())) / GB
    tokenizer._eos_token_ids = {}  # as mlx_lm.benchmark does: never stop early
    vocab = config.get("vocab_size") or config["text_config"]["vocab_size"]
    mx.random.seed(0)
    prompts = mx.random.randint(0, vocab, (max(batch_sizes), prompt_tokens)).tolist()

    def one(batch: int):
        if batch == 1:
            for response in stream_generate(model, tokenizer, prompts[0], max_tokens=gen_tokens):
                pass
            return response
        return batch_generate(model, tokenizer, prompts[:batch], max_tokens=gen_tokens,
                              completion_batch_size=batch, prefill_batch_size=min(prefill_batch, batch)).stats

    out = {"path": path, "load_s": load_s, "weights_gb": weights_gb, "prompt_tokens": prompt_tokens,
           "generation_tokens": gen_tokens, "batches": {}}
    print(f"  loaded in {load_s:.1f}s, weights {weights_gb:.2f} GB", flush=True)
    for batch in batch_sizes:
        one(batch)  # warm-up: compiles the GPU kernels for this shape
        mx.reset_peak_memory()
        runs = [one(batch) for _ in range(trials)]
        entry = {
            "prompt_tps": mean_sd([r.prompt_tps for r in runs]),
            "generation_tps": mean_sd([r.generation_tps for r in runs]),
            "peak_memory_gb": max(r.peak_memory for r in runs),
        }
        out["batches"][str(batch)] = entry
        print(f"  batch {batch:3d}: read {entry['prompt_tps']['mean']:7.0f} tok/s, "
              f"write {entry['generation_tps']['mean']:7.1f} tok/s (sd {entry['generation_tps']['sd']:.1f}), "
              f"peak {entry['peak_memory_gb']:.1f} GB", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", action="append", required=True, metavar="LABEL=PATH", help="repeatable")
    ap.add_argument("--batch-sizes", default="1,8,32")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--generation-tokens", type=int, default=256)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--prefill-batch", type=int, default=8,
                    help="how many prompts are read at once; lower it to cut peak memory on big models")
    args = ap.parse_args()
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    mx.set_cache_limit(CACHE_LIMIT)

    for spec in args.model:
        label, _, path = spec.partition("=")
        if not path:
            ap.error(f"--model {spec!r} must look like LABEL=PATH")
        print(f"{label}", flush=True)
        env = environment()
        entry = bench_model(path, batch_sizes, args.prompt_tokens, args.generation_tokens, args.trials,
                            args.prefill_batch)
        entry["prefill_batch"] = args.prefill_batch
        entry["environment"] = env
        print(f"  saved to {save('inference', label, entry)}", flush=True)
        # Free this model before the next one loads.
        free_gpu_memory()


if __name__ == "__main__":
    main()
