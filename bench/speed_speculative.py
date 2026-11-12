"""Speed experiment 3: speculative decoding (a small model drafts, the big model checks).

The idea
  A small "draft" model guesses the next few tokens cheaply. The big model then checks
  all the guesses in one pass and keeps the ones it agrees with. The output is exactly
  what the big model would have written alone; only the speed changes. It helps when
  the draft is much faster than the big model and guesses right often.

What it does
  For one big model and one draft model, writes 256 tokens for four maths questions,
  greedy decoding, with no draft and with 2, 3 and 4 drafted tokens per step. Reports
  single-stream tokens/s and the share of tokens that came from the draft.

Run:     uv run python -m bench.speed_speculative \\
             [--model mlx-community/Qwen3.5-9B-4bit] [--draft mlx-community/Qwen3.5-0.8B-4bit]
Changes: writes bench/results/speed_speculative.json. Nothing else.
"""

from __future__ import annotations

import argparse
import statistics

from datasets import load_dataset
from mlx_lm import load, stream_generate

from bench.common import environment, save
from lab.infer.batch import chat_tokens

MAX_TOKENS = 256


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="mlx-community/Qwen3.5-9B-4bit")
    ap.add_argument("--draft", default="mlx-community/Qwen3.5-0.8B-4bit")
    args = ap.parse_args()
    model, tokenizer = load(args.model)
    draft, _ = load(args.draft)
    rows = load_dataset("HuggingFaceH4/MATH-500", split="test")
    prompts = [chat_tokens(tokenizer, r["problem"], enable_thinking=False) for r in rows.select(range(4))]

    label = f"{args.model} + draft {args.draft}"
    results = {"environment": environment(), "arms": {}}
    for n_draft in (0, 2, 3, 4):
        kwargs = {"draft_model": draft, "num_draft_tokens": n_draft} if n_draft else {}
        name = f"draft {n_draft} tokens" if n_draft else "no draft"
        try:
            for r in stream_generate(model, tokenizer, prompts[0], max_tokens=32, **kwargs):  # warm-up
                pass
            speeds, from_draft, total = [], 0, 0
            for p in prompts:
                for r in stream_generate(model, tokenizer, p, max_tokens=MAX_TOKENS, **kwargs):
                    from_draft += bool(r.from_draft)
                    total += 1
                speeds.append(r.generation_tps)
            arm = {"generation_tps_mean": statistics.mean(speeds), "generation_tps_sd": statistics.stdev(speeds),
                   "share_of_tokens_from_draft": from_draft / total}
            print(f"{name:16s} {arm['generation_tps_mean']:6.1f} tok/s (sd {arm['generation_tps_sd']:.1f}), "
                  f"{arm['share_of_tokens_from_draft']:.0%} of tokens from the draft", flush=True)
        except Exception as e:  # record an unsupported model pair instead of losing the other arms
            arm = {"error": f"{type(e).__name__}: {e}"[:400]}
            print(f"{name:16s} FAILED {arm['error']}", flush=True)
        results["arms"][name] = arm
        save("speed_speculative", label, results)


if __name__ == "__main__":
    main()
