"""Speed experiment 10: does guess-and-check decoding make ONE prompt faster once extra rows are cheap?

The idea (lab/infer/lookahead.py)
  When the text being written repeats earlier text, guess the next tokens by copying
  what followed last time, and let the model check up to 7 guesses in one pass. Each
  accepted guess is a token without its own pass. With MLX's kernels a pass over 8
  rows costs about as much as 8 passes on this Mac, so there is nothing to win. With
  the lab's multi-row kernel the extra rows are cheap.

What it measures, on three kinds of task
  edit     rewrite a real source file with one small change (most of the output is a copy)
  quote    answer a question about a passage by quoting from it
  solve    a maths problem (little to copy: the hard case)
for three arrangements: plain decoding, lookahead with MLX's kernels, lookahead with the
lab kernel. Reports tokens per second, passes used, the share of guesses accepted, and
whether the text is identical to plain decoding.

Run:     uv run python -m bench.speed_lookahead [--model mlx-community/Qwen3-4B-Instruct-2507-4bit]
         (for the Qwen3.5 family add --no-thinking, so the answers start at once)
Changes: writes bench/results/speed_lookahead.json. Nothing else.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from mlx_lm import load

from bench.common import environment, save
from bench.quant_frontier import held_out_text
from lab.infer.batch import chat_tokens
from lab.infer.lookahead import generate
from lab.kernels import qmm

ROOT = Path(__file__).resolve().parent.parent
MAX_TOKENS = 400


def tasks() -> dict[str, list[str]]:
    """Real inputs only: a lab source file, held-out web text, and public maths problems."""
    source = (ROOT / "lab" / "preflight.py").read_text()
    functions = source.split("\ndef ")
    edits = [f"Here is a Python function.\n\n```python\ndef {body.strip()}\n```\n\nRewrite it with every variable "
             "name kept, adding one short comment line at the top of the body that says what it returns. "
             "Output only the code." for body in functions[1:5]]
    text = held_out_text()[:40_000]
    passages = [text[i:i + 1500] for i in range(2000, 34_000, 8000)]
    quotes = [f"Passage:\n\n{p}\n\nList the three most important sentences of the passage, copied word for word, "
              "each on its own line." for p in passages]
    rows = load_dataset("HuggingFaceH4/MATH-500", split="test")
    solve = [r["problem"] + "\n\nSolve step by step." for r in rows.select(range(4))]
    return {"edit": edits, "quote": quotes, "solve": solve}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="mlx-community/Qwen3-4B-Instruct-2507-4bit")
    ap.add_argument("--no-thinking", action="store_true", help="pass enable_thinking=False to the chat template")
    args = ap.parse_args()
    template = {"enable_thinking": False} if args.no_thinking else {}
    model, tokenizer = load(args.model)
    results = {"environment": environment(), "max_tokens": MAX_TOKENS, "tasks": {}}
    for task, prompts in tasks().items():
        prompt_ids = [chat_tokens(tokenizer, p, **template) for p in prompts]
        generate(model, tokenizer, prompt_ids[0], max_tokens=16)  # warm-up
        arms: dict[str, dict] = {}
        plain_texts: list[str] = []
        for arm, lookahead, kernel in (("plain", False, False), ("lookahead, MLX kernels", True, False),
                                       ("lookahead, lab kernel", True, True)):
            if kernel:
                qmm.install(model)
                generate(model, tokenizer, prompt_ids[0], max_tokens=32)  # compiles the kernels outside the timing
            runs = [generate(model, tokenizer, ids, max_tokens=MAX_TOKENS, lookahead=lookahead) for ids in prompt_ids]
            if kernel:
                qmm.uninstall(model)
            if arm == "plain":
                plain_texts = [r.text for r in runs]
            tokens, wall = sum(len(r.token_ids) for r in runs), sum(r.wall_s for r in runs)
            guessed, accepted = sum(r.guessed for r in runs), sum(r.accepted for r in runs)
            arms[arm] = {"tokens_per_s": tokens / wall, "tokens": tokens, "passes": sum(r.passes for r in runs),
                         "guessed": guessed, "accepted": accepted,
                         "identical_to_plain": sum(r.text == t for r, t in zip(runs, plain_texts)), "prompts": len(runs)}
            a = arms[arm]
            share = f"{accepted / guessed:.0%} of {guessed} guesses accepted" if guessed else "no guesses"
            print(f"{task:6s} {arm:24s} {a['tokens_per_s']:6.1f} tok/s   {a['tokens']} tokens in {a['passes']} passes   "
                  f"{share}   identical text {a['identical_to_plain']}/{a['prompts']}", flush=True)
        results["tasks"][task] = arms
        save("speed_lookahead", args.model, results)


if __name__ == "__main__":
    main()
