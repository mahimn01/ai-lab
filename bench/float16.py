"""Speed experiment 16: the same model run in float16 instead of the bfloat16 it was shipped in.

Background
  Apple GPUs have no bfloat16 type, so MLX's GPU code converts every bfloat16 number in
  software. float16 is native, keeps more digits, but tops out at 65,504.
  lab/infer/float16.py converts a model after checking that its values stay well below
  that on sample prompts.

What it measures
  For one model, on 8 passages x 512 tokens of the lab's held-out text:
    check      what the conversion check found: the largest value anywhere inside the
               model (on its built-in sample prompts plus these passages) and how far
               below float16's limit that is
    agreement  the model's loss on the text in both formats, and at what share of
               positions both pick the same next token
    speed      tokens per second in both formats: reading the text (scores after the
               last token) with MLX's kernels and with the lab's, and decoding 1, 8 and
               32 prompts at once (8 with both kernels)
  With --eval, also a whole eval-shaped job in both formats: 64 questions of the public
  MATH-500 set answered with up to 256 new tokens each, 32 at a time, through the lab's
  generate_batch (the workload of SPEED-RESULTS.md section 1), timed end to end.

Run:     uv run python -m bench.float16 --model LABEL=PATH_OR_REPO [--eval]
Changes: writes bench/results/float16.json. --eval downloads MATH-500 to the Hugging Face
         cache if it is not there. Nothing else.
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from bench.common import environment, free_gpu_memory, save
from bench.quant_frontier import held_out_text
from bench.speed_batching import INSTRUCTIONS, MAX_TOKENS
from lab.edit import hooks
from lab.infer import float16
from lab.infer.batch import CACHE_LIMIT, chat_tokens, generate_batch
from lab.kernels import qmm

PASSAGES, LENGTH = 8, 512
EVAL_QUESTIONS, EVAL_BATCH = 64, 32


def read_rate(model: nn.Module, tokens: mx.array, reps: int = 3) -> float:
    """Best tokens per second for reading the passages and scoring what follows each."""
    mx.eval(hooks.last_token_scores(model, tokens))
    best = 0.0
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(hooks.last_token_scores(model, tokens))
        best = max(best, tokens.size / (time.perf_counter() - t0))
    return best


def decode_rate(model: nn.Module, tokens: mx.array, rows: int, steps: int = 40) -> float:
    """Tokens per second for greedy decoding of `rows` prompts at once, after their first 64 tokens are read."""
    prompts = mx.concatenate([tokens] * (1 + rows // tokens.shape[0]))[:rows, :64]
    cache = make_prompt_cache(model)
    y = mx.argmax(model(prompts, cache=cache)[:, -1, :], axis=-1)
    mx.eval(y)

    def step() -> None:
        nonlocal y
        y = mx.argmax(model(y[:, None], cache=cache)[:, -1, :], axis=-1)
        mx.eval(y)

    for _ in range(5):
        step()
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    return rows * steps / (time.perf_counter() - t0)


def eval_job(model: nn.Module, tokenizer, questions: list[list[int]]) -> dict:
    """Answer the questions EVAL_BATCH at a time and time the whole job."""
    t0 = time.perf_counter()
    result = generate_batch(model, tokenizer, questions, max_tokens=MAX_TOKENS, batch_size=EVAL_BATCH)
    wall = time.perf_counter() - t0
    return {"wall_s": wall, "tokens": result.generation_tokens, "tokens_per_s": result.generation_tokens / wall}


def measure(model: nn.Module, tokens: mx.array, tokenizer=None, questions: list[list[int]] | None = None) -> tuple[dict, mx.array]:
    """Loss and speeds for the model in its current number format, and the next token it picks at every position."""
    logits = model(tokens).astype(mx.float32)
    loss = -mx.take_along_axis(nn.log_softmax(logits[:, :-1, :], axis=-1), tokens[:, 1:, None], axis=-1).mean()
    picks = mx.argmax(logits, axis=-1)
    mx.eval(loss, picks)
    del logits
    out = {"loss": loss.item(), "read_mlx": read_rate(model, tokens), "decode_1": decode_rate(model, tokens, 1),
           "decode_8_mlx": decode_rate(model, tokens, 8), "decode_32": decode_rate(model, tokens, 32)}
    if questions:
        out["eval_mlx"] = eval_job(model, tokenizer, questions)
    qmm.install(model)
    out |= {"read_lab": read_rate(model, tokens), "decode_8_lab": decode_rate(model, tokens, 8)}
    if questions:
        out["eval_lab"] = eval_job(model, tokenizer, questions)
    qmm.uninstall(model)
    return out, picks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", action="append", required=True, metavar="LABEL=PATH")
    ap.add_argument("--eval", action="store_true", help="also time a 64-question eval end to end in both formats")
    args = ap.parse_args()
    mx.set_cache_limit(CACHE_LIMIT)
    text = held_out_text()
    problems = []
    if args.eval:
        from datasets import load_dataset

        rows = load_dataset("HuggingFaceH4/MATH-500", split="test")
        problems = [rows[i]["problem"] for i in range(EVAL_QUESTIONS)]
    for spec in args.model:
        label, _, path = spec.partition("=")
        model, tokenizer = load(path)
        ids = tokenizer.encode(text[:400_000])[:PASSAGES * LENGTH]
        tokens = mx.array(np.array(ids).reshape(PASSAGES, LENGTH))
        questions = [chat_tokens(tokenizer, problem, system=INSTRUCTIONS, enable_thinking=False) for problem in problems]
        if questions:
            generate_batch(model, tokenizer, questions[:8], max_tokens=32, reuse_prefix=False)  # warm-up
        before, picks_before = measure(model, tokens, tokenizer, questions)
        # The check runs on the built-in sample prompts and on the passages used for timing.
        check = float16.convert(model, prompts=float16.sample_prompts(tokenizer) + np.array(ids).reshape(PASSAGES, LENGTH).tolist())
        after, picks_after = measure(model, tokens, tokenizer, questions)
        same = (picks_before == picks_after).astype(mx.float32).mean().item()
        entry = {"environment": environment(), "path": path, "passages": PASSAGES, "tokens_each": LENGTH, "check": check,
                 "bfloat16": before, "float16": after, "same_top_token_on_text": same}
        save("float16", label, entry)
        print(f"{label}: largest value inside the model {check['largest_value']:,.0f} ({check['headroom']:.0f}x below float16's limit); "
              f"loss {before['loss']:.4f} -> {after['loss']:.4f}; same top token {same:.1%}", flush=True)
        for key, name in (("read_mlx", "read, MLX's kernels"), ("read_lab", "read, lab kernels"), ("decode_1", "decode 1 prompt"),
                          ("decode_8_mlx", "decode 8, MLX's kernels"), ("decode_8_lab", "decode 8, lab kernel"),
                          ("decode_32", "decode 32")):
            print(f"  {name:24s} {before[key]:7.0f} -> {after[key]:7.0f} tok/s  ({after[key] / before[key]:.2f}x)", flush=True)
        for key, name in (("eval_mlx", "whole eval, MLX's kernels"), ("eval_lab", "whole eval, lab kernels")) if questions else ():
            print(f"  {name:24s} {before[key]['wall_s']:7.1f} -> {after[key]['wall_s']:7.1f} s  "
                  f"({before[key]['wall_s'] / after[key]['wall_s']:.2f}x)", flush=True)
        del model
        free_gpu_memory()


if __name__ == "__main__":
    main()
