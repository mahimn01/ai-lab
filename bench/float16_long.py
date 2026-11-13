"""Speed experiment 19 (check): is float16 safe for the neural-harness study's own models on long texts?

Background
  Section 16 found that the Qwen models run faster in float16 than in the bfloat16 they ship in,
  and that float16 breaks if any value inside the model passes 65,504. Its check used short
  prompts and the quantized models. The study uses the dense Qwen3.5-0.8B (as released, and with
  the published edit) and writes thousands of tokens, so the room has to be measured again there.

What it measures
  For each arm of the study (base, edited): the largest value written anywhere inside the model
  (every layer's output, every weight table's input and output) while it reads saved transcripts
  of the study, prompt plus the first 4,000 generated tokens each, and how far below float16's
  limit that is. Then the same check that `float16.convert` runs: converted, do the two formats
  still pick the same next token?

Run:     uv run python -m bench.float16_long --harness ~/Documents/Dev/neural-harness
Changes: writes bench/results/float16_long.json. Reads the study's folder, never writes to it.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

from bench.common import environment, free_gpu_memory, save
from lab.infer import float16

TRANSCRIPTS, TOKENS = 4, 4000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--harness", type=Path, default=Path.home() / "Documents/Dev/neural-harness")
    args = ap.parse_args()
    root = args.harness.expanduser()
    sys.path.insert(0, str(root))
    from harness.model import load_arm

    for arm, run in (("base", "test__base__s0"), ("edited", "test__astra__s0")):
        saved = [json.loads(line) for line in gzip.open(root / f"runs/sd1/gen/{run}.text.jsonl.gz", "rt")]
        texts = [(r["prompt"] + r["tokens"])[: len(r["prompt"]) + TOKENS] for r in saved if len(r["tokens"]) >= TOKENS][:TRANSCRIPTS]
        model, tokenizer = load_arm(arm)
        largest, _ = float16.largest_value(model, texts)
        entry = {"environment": environment(), "arm": arm, "transcripts": len(texts), "tokens_each": len(texts[0]),
                 "largest_value": largest, "headroom": float16.LIMIT / largest}
        print(f"{arm}: largest value inside the model on {len(texts)} transcripts of {len(texts[0]):,} tokens: {largest:,.0f} "
              f"({float16.LIMIT / largest:.0f}x below float16's limit)", flush=True)
        try:
            entry["convert_check"] = float16.convert(model, prompts=[t[:2048] for t in texts] + float16.sample_prompts(tokenizer))
            print(f"  converted: same top token at {entry['convert_check']['same_top_token']:.1%} of positions", flush=True)
        except ValueError as refused:
            entry["convert_refused"] = str(refused)
            print(f"  {refused}", flush=True)
        save("float16_long", arm, entry)
        del model
        free_gpu_memory()


if __name__ == "__main__":
    main()
