"""Speed experiments 1 and 2: batching, length sorting and shared-prefix reuse on an eval-shaped workload.

The workload
  64 real maths questions (the public MATH-500 test set), each answered with up to 256
  new tokens, greedy decoding. This is the shape of an eval run: many prompts, one model.

Experiment 1  the same 64 questions at batch sizes 1, 4, 8, 16, 32 and 64, and with
              prompts sorted by length or not. Question: which batch size finishes the
              job fastest, and does sorting help?
Experiment 2  the same questions behind one long shared instruction block (a system
              prompt plus eight worked examples, about 2,000 tokens). Question: how much
              time does reading that block once, instead of 64 times, save?

The score is end-to-end tokens per second: new tokens written divided by the whole
wall-clock time, including reading the prompts. Higher is better.

Run:     uv run python -m bench.speed_batching [--model mlx-community/Qwen3.5-4B-4bit]
Changes: writes bench/results/speed_batching.json. Nothing else.
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
from datasets import load_dataset
from mlx_lm import load

from bench.common import environment, save
from lab.infer.batch import chat_tokens, common_prefix_len, generate_batch

N_QUESTIONS = 64
MAX_TOKENS = 256
INSTRUCTIONS = ("You are a careful mathematics tutor. Solve the problem step by step, keep the working short, "
                "and finish with the final answer on its own line in the form 'Answer: <answer>'.")


def timed(model, tokenizer, prompts, **kwargs) -> dict:
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    r = generate_batch(model, tokenizer, prompts, max_tokens=MAX_TOKENS, **kwargs)
    wall = time.perf_counter() - t0
    return {"wall_s": wall, "new_tokens": r.generation_tokens, "end_to_end_tps": r.generation_tokens / wall,
            "generation_tps": r.generation_tps, "prompt_tps": r.prompt_tps, "peak_memory_gb": r.peak_memory_gb,
            "prefix_tokens_reused": r.prefix_tokens_reused, "prompts": len(prompts)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="mlx-community/Qwen3.5-4B-4bit")
    args = ap.parse_args()
    model, tokenizer = load(args.model)
    rows = load_dataset("HuggingFaceH4/MATH-500", split="test")
    questions = [r["problem"] for r in rows.select(range(N_QUESTIONS))]
    short = [chat_tokens(tokenizer, q, system=INSTRUCTIONS, enable_thinking=False) for q in questions]

    # A long shared block: the instructions plus eight worked examples from the end of the set.
    examples = "\n\n".join(f"Problem: {r['problem']}\nSolution: {r['solution']}\nAnswer: {r['answer']}"
                           for r in rows.select(range(400, 408)))
    long_system = f"{INSTRUCTIONS}\n\nWorked examples:\n\n{examples}"
    long = [chat_tokens(tokenizer, q, system=long_system, enable_thinking=False) for q in questions]

    results = {"model": args.model, "environment": environment(),
               "short_prompt_tokens_mean": sum(map(len, short)) / len(short),
               "long_prompt_tokens_mean": sum(map(len, long)) / len(long),
               "long_shared_prefix_tokens": common_prefix_len(long)}
    generate_batch(model, tokenizer, short[:8], max_tokens=32, reuse_prefix=False)  # warm-up

    def run(label: str, prompts, **kwargs) -> None:
        results[label] = timed(model, tokenizer, prompts, **kwargs)
        r = results[label]
        print(f"{label:44s} {r['end_to_end_tps']:7.1f} tok/s end to end, {r['wall_s']:6.1f} s, "
              f"peak {r['peak_memory_gb']:.1f} GB", flush=True)
        save("speed_batching", args.model, results)

    # Experiment 1. Batch 1 uses 16 questions to keep the run short; the score is a rate, so it compares.
    # Sorting is switched off by name in the plain arms, because it is the library's default.
    run("exp1 batch 1 (16 questions)", short[:16], batch_size=1, reuse_prefix=False, sort_by_length=False)
    for b in (4, 8, 16, 32, 64):
        run(f"exp1 batch {b}", short, batch_size=b, reuse_prefix=False, sort_by_length=False)
    run("exp1 batch 16 sorted by length", short, batch_size=16, reuse_prefix=False, sort_by_length=True)
    run("exp1 batch 32 sorted by length", short, batch_size=32, reuse_prefix=False, sort_by_length=True)

    # Experiment 2.
    run("exp2 long prompts, batch 32, no reuse", long, batch_size=32, reuse_prefix=False, sort_by_length=False)
    run("exp2 long prompts, batch 32, prefix reused", long, batch_size=32, reuse_prefix=True, sort_by_length=False)


if __name__ == "__main__":
    main()
