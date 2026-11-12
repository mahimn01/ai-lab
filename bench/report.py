"""Build BASELINE.md from the measured benchmark files.

What it does
  Reads bench/results/{inference,batch_overhead,training,lora}.json and writes
  BASELINE.md: one table per benchmark, each naming the command that produced it. No
  number in BASELINE.md is typed by hand, so re-running a benchmark and then this
  script always gives a consistent file.

Run:     uv run python -m bench.report
Changes: overwrites BASELINE.md. Nothing else.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results"


def load(name: str) -> dict:
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def inference_section(data: dict) -> str:
    rows = []
    for label, e in data.items():
        b = e["batches"]

        def cell(size: str, key: str, fmt: str) -> str:
            return format(b[size][key]["mean"] if isinstance(b[size][key], dict) else b[size][key], fmt) if size in b else "not run"

        biggest = max(b, key=int)
        rows.append([label, f"{e['weights_gb']:.1f}", f"{e['load_s']:.1f}", cell("1", "prompt_tps", ",.0f"),
                     cell("1", "generation_tps", ".1f"), cell("8", "generation_tps", ".1f"),
                     cell("32", "generation_tps", ".1f"), f"{b[biggest]['peak_memory_gb']:.1f} (batch {biggest})"])
    return table(["Model", "Weights GB", "Load s", "Read tok/s (1 prompt)", "Write tok/s, 1 prompt",
                  "Write tok/s, 8 at once", "Write tok/s, 32 at once", "Peak memory GB"], rows)


def overhead_section(data: dict) -> str:
    parts = []
    for label, e in data.items():
        if any(tag in label for tag in ("mxfp4", "arch-", "rerun")):  # speed experiments: see SPEED-RESULTS.md
            continue
        rows = [[size, f"{r['generator_wall_ms_per_step']:.1f}", f"{r['generator_tokens_per_s']:.0f}",
                 f"{r['raw_ms_per_step']:.1f}"] if "error" not in r else [size, "failed", "", ""]
                for size, r in e["batches"].items()]
        parts.append(f"**{label}** (one prompt alone: {e['single_ms_per_token']:.1f} ms per token)\n\n"
                     + table(["Prompts at once", "ms per step", "Tokens/s (all prompts)", "ms per step, bare loop"], rows))
    return "\n\n".join(parts)


def training_section(data: dict) -> str:
    rows = []
    for label, e in data.items():
        size, arm = label.split(" | ")
        if "error" in e:
            rows.append([size, arm, "failed", "", e["error"][:60]])
            continue
        memory = e.get("peak_memory_gb", e.get("driver_allocated_gb"))
        note = "" if e["loss_is_finite"] else "loss became NaN: not usable"
        rows.append([size, arm, f"{e['tokens_per_s']:,.0f}", f"{memory:.1f}", note])
    return table(["Model size", "Setup", "Tokens/s", "Memory GB", "Note"], rows)


def lora_section(data: dict) -> str:
    rows = [[label, f"{e['tokens_per_second']:,.0f}", f"{e['peak_memory_gb']:.1f}",
             f"{e['train_loss_first']:.2f} -> {e['train_loss_last']:.2f}"] if "error" not in e
            else [label, "failed", "", e["error"][:60]] for label, e in data.items()]
    return table(["Model", "Tokens/s", "Peak memory GB", "Training loss, first -> last report"], rows)


def main() -> None:
    inference, overhead, training, lora = (load(n) for n in ("inference", "batch_overhead", "training", "lora"))
    env = next(iter(inference.values()))["environment"] if inference else {}
    text = f"""# Baseline: what this Mac does today

Measured on {env.get('chip', '?')}, {env.get('memory_gb', '?')} GB, macOS {env.get('macos', '?')}, MLX {env.get('mlx', '?')},
mlx-lm {env.get('mlx_lm', '?')}. Built by `uv run python -m bench.report` from the JSON files in
`bench/results/`; every number here comes from those files.

"tok/s" means tokens per second. A token is a piece of a word, about four characters.

## 1. Running models (inference)

Command: `uv run python -m bench.inference --model LABEL=PATH ...` (512-token prompt, 256 new tokens, 3 trials).
"Write tok/s, N at once" adds up the tokens of all N prompts.

{inference_section(inference)}

## 2. The cost of one batched step

Command: `uv run python -m bench.batch_overhead --model LABEL=PATH`.
Time for one step that writes one token for every prompt in the batch. "Bare loop" calls
the model directly with no generator code around it; it matches the generator, so the
time is spent inside the model's GPU code, not in bookkeeping.

{overhead_section(overhead)}

## 3. Training a small GPT from scratch

Command: `uv run --group torch python -m bench.training` (batch 32 x 512 tokens, AdamW, 20 s per setup).
Sizes: s = 14M parameters, m = 29M, gpt2 = 91M (see `lab/train/model.py`).

{training_section(training)}

## 4. LoRA fine-tuning

Command: `uv run python -m bench.lora --model LABEL=PATH` (40 steps, batch 4, last 16 layers, up to 512 tokens).

{lora_section(lora)}
"""
    (ROOT / "BASELINE.md").write_text(text)
    print(f"wrote {ROOT / 'BASELINE.md'}")


if __name__ == "__main__":
    main()
