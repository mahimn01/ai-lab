"""Speed experiment 4: how much quality does each level of quantization cost, and what does it buy?

Plain words
  Quantization stores each weight in fewer bits (16 -> 8, 6, 5, 4 or 3), so the model
  is smaller and usually faster, at some cost in quality. "Mixed" recipes give the
  most sensitive layers more bits than the rest.

What it does
  Starting from one full-precision model, makes each quantized version with mlx-lm's
  converter and measures for each:
    size on disk
    perplexity   how surprised the model is by held-out text (lower = better quality).
                 Measured on 64 x 512 tokens of the lab's FineWeb-Edu validation text,
                 the same text for every version.
    speed        tokens/s for one prompt and for 32 prompts at once
  The converted copies are deleted after measuring unless --keep is given.

Run:     uv run python -m bench.quant_frontier [--source PATH_TO_FULL_PRECISION_MODEL]
Changes: writes bench/results/quant_frontier.json; writes then deletes model copies
         under runs/quant/ (several GB while it runs).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import convert, load
from mlx_lm.perplexity import eval_ppl
from tokenizers import Tokenizer

from bench.common import environment, free_gpu_memory, save
from lab.infer.batch import CACHE_LIMIT
from bench.inference import bench_model

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "fineweb_edu"
DEFAULT_SOURCE = Path.home() / "Documents/Dev/neural-harness/models/qwen3.5-4b"
# label -> extra arguments for mlx_lm.convert (None = the full-precision source itself)
VARIANTS: dict[str, dict | None] = {
    "full precision (bf16)": None,
    "8-bit": {"q_bits": 8},
    "6-bit": {"q_bits": 6},
    "5-bit": {"q_bits": 5},
    "mixed 4/6-bit": {"q_bits": 4, "quant_predicate": "mixed_4_6"},
    "4-bit": {"q_bits": 4},
    "mixed 3/6-bit": {"q_bits": 3, "quant_predicate": "mixed_3_6"},
    "3-bit": {"q_bits": 3},
}
N_SEQUENCES, SEQ_LEN = 64, 512


def held_out_text() -> str:
    """The first ~200k tokens of the lab's validation set, decoded back to text."""
    lab_tokenizer = Tokenizer.from_file(str(DATA_DIR / "tokenizer.json"))
    ids = np.fromfile(DATA_DIR / "val.bin", dtype=np.uint16, count=200_000)
    return lab_tokenizer.decode(ids.tolist())


def folder_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*.safetensors")) / 2**30


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", default=str(DEFAULT_SOURCE))
    ap.add_argument("--keep", action="store_true", help="keep the converted models under runs/quant/")
    args = ap.parse_args()
    source = Path(args.source)
    text = held_out_text()
    mx.set_cache_limit(CACHE_LIMIT)

    for label, convert_args in VARIANTS.items():
        slug = label.replace(" ", "-").replace("/", "-").replace("(", "").replace(")", "")
        path = source if convert_args is None else ROOT / "runs" / "quant" / f"{source.name}-{slug}"
        try:
            if convert_args is not None and not path.exists():
                convert(str(source), mlx_path=str(path), quantize=True, **convert_args)
            model, tokenizer = load(str(path))
            ids = tokenizer.encode(text, add_special_tokens=False)[: N_SEQUENCES * SEQ_LEN]
            ppl, ppl_se = eval_ppl(model, mx.array(ids).reshape(N_SEQUENCES, SEQ_LEN))
            del model
            free_gpu_memory()
            speed = bench_model(str(path), [1, 32], prompt_tokens=512, gen_tokens=128, trials=2)
            entry = {"size_gb": folder_gb(path), "perplexity": ppl, "perplexity_se": ppl_se,
                     "single_tps": speed["batches"]["1"]["generation_tps"]["mean"],
                     "batch32_tps": speed["batches"]["32"]["generation_tps"]["mean"],
                     "peak_memory_gb_batch32": speed["batches"]["32"]["peak_memory_gb"]}
            print(f"{label:24s} {entry['size_gb']:5.2f} GB  perplexity {ppl:7.3f} (+/- {ppl_se:.3f})  "
                  f"{entry['single_tps']:6.1f} tok/s single  {entry['batch32_tps']:7.1f} tok/s batch 32", flush=True)
        except Exception as e:  # keep the other variants if one recipe is not supported for this model
            entry = {"error": f"{type(e).__name__}: {e}"[:400]}
            print(f"{label:24s} FAILED {entry['error']}", flush=True)
        entry["environment"] = environment()
        save("quant_frontier", f"{source.name} | {label}", entry)
        free_gpu_memory()
        if convert_args is not None and not args.keep and path.exists():
            shutil.rmtree(path)  # a copy this script made a moment ago


if __name__ == "__main__":
    main()
