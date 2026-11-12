# Baseline: what this Mac does today

Measured on Apple M1 Max, 32 GB, macOS 27.0.1, MLX 0.32.4,
mlx-lm 0.32.2. Built by `uv run python -m bench.report` from the JSON files in
`bench/results/`; every number here comes from those files.

"tok/s" means tokens per second. A token is a piece of a word, about four characters.

## 1. Running models (inference)

Command: `uv run python -m bench.inference --model LABEL=PATH ...` (512-token prompt, 256 new tokens, 3 trials).
"Write tok/s, N at once" adds up the tokens of all N prompts.

| Model | Weights GB | Load s | Read tok/s (1 prompt) | Write tok/s, 1 prompt | Write tok/s, 8 at once | Write tok/s, 32 at once | Peak memory GB |
|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b-bf16 | 1.4 | 1.0 | 3,476 | 120.5 | 377.5 | 1076.3 | 4.3 (batch 32) |
| qwen3.5-0.8b-4bit | 0.4 | 1.3 | 3,543 | 234.6 | 568.6 | 1350.9 | 3.1 (batch 32) |
| qwen3.5-4b-bf16 | 7.8 | 3.3 | 756 | 31.4 | 96.1 | 299.1 | 13.2 (batch 32) |
| qwen3.5-4b-4bit | 2.2 | 1.7 | 706 | 96.1 | 130.2 | 413.1 | 7.4 (batch 32) |
| qwen3.5-9b-4bit | 4.7 | 2.0 | 337 | 56.7 | 74.1 | 257.7 | 9.9 (batch 32) |
| qwen3.5-9b-8bit | 8.9 | 3.5 | 379 | 35.9 | 74.2 | 258.6 | 14.4 (batch 32) |
| qwen3.8-27b-4bit | 14.1 | 5.6 | 108 | 19.7 | 21.9 | not run | 21.6 (batch 8) |

## 2. The cost of one batched step

Command: `uv run python -m bench.batch_overhead --model LABEL=PATH`.
Time for one step that writes one token for every prompt in the batch. "Bare loop" calls
the model directly with no generator code around it; it matches the generator, so the
time is spent inside the model's GPU code, not in bookkeeping.

**qwen3.5-0.8b-4bit** (one prompt alone: 3.4 ms per token)

| Prompts at once | ms per step | Tokens/s (all prompts) | ms per step, bare loop |
|---|---|---|---|
| 1 | 3.6 | 274 | 4.6 |
| 2 | 4.9 | 410 | 5.8 |
| 4 | 7.3 | 548 | 8.5 |
| 8 | 12.6 | 635 | 13.7 |
| 16 | 17.0 | 943 | 17.9 |
| 32 | 19.2 | 1664 | 20.2 |
| 64 | 32.7 | 1958 | 32.9 |

**qwen3.5-9b-4bit** (one prompt alone: 16.0 ms per token)

| Prompts at once | ms per step | Tokens/s (all prompts) | ms per step, bare loop |
|---|---|---|---|
| 1 | 16.6 | 60 | 17.8 |
| 2 | 24.4 | 82 | 25.7 |
| 4 | 44.6 | 90 | 45.6 |
| 8 | 104.6 | 77 | 105.4 |
| 16 | 115.0 | 139 | 115.4 |
| 32 | 114.0 | 281 | 114.0 |
| 64 | 217.3 | 295 | 215.4 |

**qwen3.5-4b-4bit** (one prompt alone: 10.0 ms per token)

| Prompts at once | ms per step | Tokens/s (all prompts) | ms per step, bare loop |
|---|---|---|---|
| 1 | 10.5 | 95 | 12.1 |
| 2 | 15.2 | 132 | 16.6 |
| 4 | 26.5 | 151 | 28.2 |
| 8 | 59.3 | 135 | 61.1 |
| 16 | 66.9 | 239 | 67.9 |
| 32 | 70.4 | 454 | 71.2 |
| 64 | 133.3 | 480 | 131.2 |

## 3. Training a small GPT from scratch

Command: `uv run --group torch python -m bench.training` (batch 32 x 512 tokens, AdamW, 20 s per setup).
Sizes: s = 14M parameters, m = 29M, gpt2 = 91M (see `lab/train/model.py`).

| Model size | Setup | Tokens/s | Memory GB | Note |
|---|---|---|---|---|
| s | mlx float32 compiled | 53,644 | 7.9 |  |
| s | mlx float32 not compiled | 48,746 | 6.7 |  |
| s | mlx bfloat16 compiled | 62,323 | 4.0 |  |
| s | mlx float16 compiled | 71,982 | 4.0 | loss became NaN: not usable |
| s | torch-mps float32 | 39,296 | 7.1 |  |
| s | torch-mps autocast float16 | 38,666 | 6.6 |  |
| s | torch-mps autocast bfloat16 | 29,574 | 6.6 |  |
| m | mlx float32 compiled | 28,418 | 10.9 |  |
| m | mlx float32 not compiled | 24,880 | 9.2 |  |
| m | mlx bfloat16 compiled | 32,256 | 5.6 |  |
| m | mlx float16 compiled | 37,244 | 5.6 | loss became NaN: not usable |
| m | torch-mps float32 | 21,100 | 10.7 |  |
| m | torch-mps autocast float16 | 21,086 | 8.7 |  |
| m | torch-mps autocast bfloat16 | 15,640 | 8.7 |  |
| gpt2 | mlx float32 compiled | 125 | 20.4 |  |
| gpt2 | mlx float32 not compiled | 201 | 18.7 |  |
| gpt2 | mlx bfloat16 compiled | 11,436 | 10.3 |  |
| gpt2 | mlx float16 compiled | 13,222 | 10.3 | loss became NaN: not usable |
| gpt2 | torch-mps float32 | 8,191 | 21.1 |  |
| gpt2 | torch-mps autocast float16 | 8,176 | 18.1 |  |
| gpt2 | torch-mps autocast bfloat16 | 5,794 | 18.1 |  |

## 4. LoRA fine-tuning

Command: `uv run python -m bench.lora --model LABEL=PATH` (40 steps, batch 4, last 16 layers, up to 512 tokens).

| Model | Tokens/s | Peak memory GB | Training loss, first -> last report |
|---|---|---|---|
| qwen3-0.6b-4bit | seq 512, batch 4, grad checkpoint off | 1,460 | 2.0 | 2.55 -> 1.74 |
| qwen3.5-0.8b-4bit | seq 128, batch 4, grad checkpoint off | 258 | 14.8 | 2.00 -> 1.47 |
| qwen3.5-0.8b-4bit | seq 512, batch 4, grad checkpoint off | 205 | 18.3 | 2.00 -> 1.45 |
| qwen3.5-0.8b-4bit | seq 512, batch 4, grad checkpoint on | 219 | 3.9 | 2.00 -> 1.45 |
| qwen3-4b-4bit | seq 512, batch 4, grad checkpoint off | 313 | 5.1 | 2.11 -> 1.43 |
| qwen3.5-4b-4bit | seq 512, batch 4, grad checkpoint auto | 88 | 9.4 | 1.82 -> 1.29 |
| qwen3.5-9b-4bit | seq 512, batch 4, grad checkpoint auto | 56 | 12.0 | 1.83 -> 1.28 |
