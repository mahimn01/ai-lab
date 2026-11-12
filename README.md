# ai-lab

Tools for ML research on one Apple Silicon Mac, with the measurements behind them. It runs on MLX.

The repo has a trainer for small GPTs, LoRA fine-tuning, weight editing, activation hooks and batched inference, plus a few GPU code paths that replace slow spots in MLX and mlx-lm. Every number below was measured on a Mac Studio (M1 Max, 32 GB) with MLX 0.32.4 and mlx-lm 0.32.2.

## What was slow

Three findings explain most of the gains.

The Qwen models converted for MLX store their 16-bit numbers as bfloat16. Apple GPUs have no such type, so MLX's GPU code converts each number in software. float16 is native.

mlx-lm trains Qwen3.5's recurrent layers with a Python loop over tokens. Doing the same arithmetic 64 tokens at a time is 25x faster per layer, and on identical batches the two trainers end within 0.001 of each other in validation loss.

MLX's kernel for quantized weights decodes them again for every 32 rows. Decoding once reads prompts 1.2 to 1.33x faster.

## What got faster

tok/s is tokens per second. The Qwen models are the 4-bit MLX conversions, and a row that names no model used Qwen3.5-4B.

| Work | Before | After |
|---|---|---|
| LoRA on Qwen3.5-0.8B, 4B and 9B, short examples | 212, 86 and 64 tok/s | 1,172, 338 and 166 tok/s |
| LoRA on Qwen3.5-4B, four 512-token examples | out of memory | 345 tok/s in 9.7 GB |
| LoRA on Qwen3-4B, short examples | 289 tok/s | 422 tok/s |
| Train a 14M GPT from scratch | 48.9k tok/s in 7.7 GB | 60.7k tok/s in 4.5 GB |
| Read prompts | 724 tok/s | 1,022 in float16, 1,117 with the lab's paths as well |
| Decode 32 prompts at once | 450 tok/s | 560 tok/s in float16 |
| Decode 8 prompts at once | 131 tok/s | 271 tok/s with the lab's kernel |
| A 64-question eval, 32 at a time | 51.8 s | 40.1 s |
| Collect an early layer's activations | a full pass | about 6x faster |
| Sweep an edit three quarters of the way up | a full pass per variant | 5.3x faster |
| One chat writing new text | 81 to 92 tok/s on the 4B models | unchanged |

## Before you trust a number

Most numbers are single runs on one machine, and where a result was repeated `SPEED-RESULTS.md` says so.

float16 is for inference only. `float16.convert` runs sample prompts through the model first and refuses if any value inside it comes within 4x of float16's limit of 65,504. The Qwen3.5 models had 162 to 809x of room but Qwen3-4B had only 9x, so run the check on prompts like your own.

One chat did not get faster. About half of each token's time goes to reading the weights from memory, and the rest goes to roughly 1,100 small GPU operations that nothing here removes.

## Start

You need an Apple Silicon Mac and [uv](https://docs.astral.sh/uv/).

```
git clone https://github.com/mahimn01/ai-lab
cd ai-lab
uv sync
uv run python -m lab.preflight
```

`lab.preflight` prints free memory, swap, disk space and heat, and says whether the Mac is ready for a heavy job. `uv run pytest -m "not slow"` runs the quick tests in about 5 seconds.

## Where to read more

- `SPEED-RESULTS.md` has every experiment with its command, its numbers and a keep or reject verdict. The rejected ideas are in there too.
- `GUIDE.md` lists the commands, the Python entry points and what fits in 32 GB.
- `BASELINE.md` has this Mac's speeds before any of the changes.

MIT license. Mahimn Patel ([mahimn.dev](https://mahimn.dev), [x.com/mahimnpatel](https://x.com/mahimnpatel)).
