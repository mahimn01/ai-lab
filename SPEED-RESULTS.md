# Speed results: what was tested, and what to keep

Every idea below was run on this Mac (Apple M1 Max, 32 GB, MLX 0.32.4, mlx-lm 0.32.2) on
2026-11-11. Each section gives the command, the numbers (from `bench/results/*.json`)
and a verdict: **keep**, **reject**, or **not settled**.

Words used here:

- **tok/s**: tokens per second. A token is a piece of a word, about four characters.
- **batch**: several prompts answered at the same time on the GPU.
- **end to end**: counting all the time, both reading the prompts and writing the answers.
- **quantization, N-bit**: storing each weight of a model in N bits instead of 16, to
  make it smaller.
- **perplexity**: how surprised a model is by real text. Lower is better.
- **val_bpb**: validation bits per byte, the trainer's score. Lower is better.

## Summary

| # | Idea | Result | Verdict |
|---|---|---|---|
| 1 | Answer 32 prompts at once | 3.2x faster end to end than one at a time (4B model) | keep |
| 1 | Sort prompts by length | 9% faster on top of that (one run) | keep, now the default |
| 1 | Batches of 8 or 64 | 8 is no faster than 4; 64 is no faster than 32 | reject |
| 2 | Read a shared prompt opening once | 5.0x faster end to end on long shared prompts | keep, automatic |
| 3 | Speculative decoding (small model drafts) | 12-19% slower on Qwen3; not supported on Qwen3.5 | reject |
| 4 | 8-bit or 6-bit instead of 4-bit for batch work | same speed at batch 32, quality nearly untouched | keep |
| 4 | 3-bit | 34% worse perplexity for no speed gain | reject |
| 5 | Muon optimizer for training | 0.32 val_bpb better than the best AdamW in the same time | keep, the default |
| 5 | Muon rate 0.04 instead of 0.02 | 0.05 val_bpb better, 7 times the noise | keep, now the default |
| 5 | 16-bit training numbers (bfloat16) | 13% more steps, half the memory, score no worse | keep as an option |
| 5 | float16 training | loss became NaN every time | reject |
| 5 | Compiled training step | 9% faster | keep, the default |
| 6 | A different GPU code path (mxfp4 format, architecture override) | no gain, or slower | reject |
| 7 | Small vocabulary for from-scratch training | 2.6x faster, 40% less memory | keep, the default |
| 8 | Gradient checkpointing for Qwen3.5 fine-tuning | 18.3 GB down to 3.9 GB, same speed | keep, automatic |
| 9 | Raising the GPU memory cap | not tested | not settled |

**Was a new, faster technique found?** No new GPU-level technique. The gains are at the
workload level: batch at exactly 32, share the prompt opening, choose bits by the job,
and train with Muon. One finding is new to me and may be worth reporting to the MLX
project: on M1 chips a decoding step for 8 prompts costs as much as one for 32
(section 1). A fix would have to be made inside MLX's GPU code, and it is untested.

## 1. Batch size and sorting

Command: `uv run python -m bench.speed_batching` (model `mlx-community/Qwen3.5-4B-4bit`,
64 MATH-500 questions, up to 256 new tokens each, greedy decoding, one run per row).

| Prompts at once | tok/s end to end | Wall time | Peak memory |
|---|---|---|---|
| 1 (16 questions) | 88.1 | 45.5 s | 3.3 GB |
| 4 | 131.3 | 119.8 s | 4.4 GB |
| 8 | 124.1 | 127.8 s | 5.4 GB |
| 16 | 183.8 | 86.6 s | 6.3 GB |
| 32 | 280.6 | 56.4 s | 8.3 GB |
| 64 | 269.1 | 58.6 s | 12.3 GB |
| 16, sorted by length | 189.6 | 83.3 s | 6.2 GB |
| 32, sorted by length | 305.6 | 51.7 s | 7.9 GB |

**Why the odd shape.** The time for one step (one new token for every prompt) does not
grow smoothly. Command: `uv run python -m bench.batch_overhead --model LABEL=PATH`.

| Prompts at once | 0.8B 4-bit, ms per step | 4B 4-bit | 9B 4-bit |
|---|---|---|---|
| 1 | 3.6 | 10.5 | 16.6 |
| 2 | 4.9 | 15.2 | 24.4 |
| 4 | 7.3 | 26.5 | 44.6 |
| 8 | 12.6 | 59.3 | 104.6 |
| 16 | 17.0 | 66.9 | 115.0 |
| 32 | 19.2 | 70.4 | 114.0 |
| 64 | 32.7 | 133.3 | 217.3 |

For the 9B model a step for 8 prompts costs 105 ms and a step for 32 costs 114 ms. A
bare loop with no generator code around it gives the same times, so the cost is inside
MLX's GPU code. Its source (v0.32.4, `mlx/backend/metal/quantized.cpp` and
`kernels/quantized.h`) explains it: for a quantized model on this chip generation, a
small batch is handled one prompt at a time, and above a threshold of 6 to 14 prompts
(it depends on the layer size) the work is done in fixed blocks of 32 rows. A batch of
8 is padded with zeros to 32 and pays for all 32.

**What to do with this.**

- Use 32 at once for throughput work. 64 is two blocks: same tok/s, 4 GB more memory.
- Never use 8: it is slower per token than 4.
- If a job has 8 to 31 prompts, fill the batch to 32. Extra samples of the same
  question (for majority voting or measuring variance) are nearly free.
- Sorting by length is now the default in `lab/infer/batch.py`. The gain came from
  reading the prompts faster (562 against 426 tok/s), because less padding is needed.
  It was measured once, so treat the 9% as approximate.

**Possible upstream contribution.** A smaller block (8 or 16 rows) for batches of 8 to
31 could make those sizes much cheaper. This is a guess from the numbers, not a tested
change, and it needs new GPU kernel code in MLX. The MLX project does not accept
AI-written issue or PR text, so any report has to be written by you.

Verdict: **keep** batch 32 and sorting; **reject** 8 and 64.

## 2. Reading a shared prompt opening once

Command: `uv run python -m bench.speed_batching` (second half). The same 64 questions,
each behind the same 2,747-token block of instructions and worked examples, batch 32.

| Arrangement | tok/s end to end | Wall time | Peak memory |
|---|---|---|---|
| Every prompt read in full | 47.0 | 331.5 s | 15.9 GB |
| Shared opening read once | 236.9 | 65.9 s | 14.6 GB |

5.0x faster. `generate_batch` does this by itself whenever all prompts start with the
same 64 or more tokens. A test checks that the answers are identical to one-at-a-time
generation.

Verdict: **keep**. Put the part every prompt shares (system prompt, examples, document)
at the start, and the part that differs at the end.

## 3. Speculative decoding

Command: `uv run python -m bench.speed_speculative --model BIG --draft SMALL`
(four maths questions, 256 new tokens each, one prompt at a time, greedy decoding).

A small "draft" model guesses the next few tokens and the big model checks them in one
pass. The text is the same as the big model alone; only the speed changes.

| Big model + draft | Drafted tokens per step | tok/s (spread) | Tokens accepted from the draft |
|---|---|---|---|
| Qwen3-4B 4-bit + Qwen3-0.6B 4-bit | none | 108.0 (0.7) | |
| | 2 | 95.4 (5.3) | 62% |
| | 3 | 92.3 (7.9) | 69% |
| | 4 | 87.8 (11.7) | 74% |
| Qwen3.5-9B 4-bit + Qwen3.5-0.8B 4-bit | none | 61.2 (0.1) | |
| | 2, 3, 4 | failed | |

The draft guessed right 62 to 74% of the time and it was still slower, because on this
Mac the small model is not cheap enough next to the big one. For the Qwen3.5 family
mlx-lm refuses outright ("requires a trimmable prompt cache"). It also only works one
prompt at a time, where batching already gives more.

Verdict: **reject** on this Mac for these model pairs.

## 4. Quantization: quality against speed

Command: `uv run python -m bench.quant_frontier` (Qwen3.5-4B, converted from the
full-precision copy with mlx-lm; perplexity on 64 x 512 tokens of FineWeb-Edu validation
text, the same text for every row).

| Version | Size | Perplexity (standard error) | Change | tok/s, 1 prompt | tok/s, 32 at once |
|---|---|---|---|---|---|
| Full precision (16-bit) | 8.68 GB | 11.600 (0.172) | | 35.3 | 303.2 |
| 8-bit | 4.16 GB | 11.599 (0.172) | 0.0% | 61.8 | 408.6 |
| 6-bit | 3.18 GB | 11.625 (0.173) | +0.2% | 74.3 | 398.4 |
| 5-bit | 2.69 GB | 11.779 (0.175) | +1.5% | 83.9 | 399.4 |
| Mixed 4/6-bit | 2.29 GB | 12.209 (0.184) | +5.3% | 94.6 | 403.8 |
| 4-bit | 2.20 GB | 12.277 (0.184) | +5.8% | 97.6 | 405.1 |
| Mixed 3/6-bit | 1.85 GB | 15.025 (0.231) | +29.5% | 97.5 | 401.5 |
| 3-bit | 1.72 GB | 15.524 (0.240) | +33.8% | 101.1 | 398.1 |

Two things stand out:

- **At 32 prompts at once, every quantized version runs at the same speed** (398 to 409
  tok/s). The baseline shows the same for the 9B model: 258 tok/s at both 4-bit and
  8-bit. So for batch work, fewer bits buy nothing but memory.
- **For one prompt, fewer bits are faster**: 4-bit is 58% faster than 8-bit.

Verdict: **keep** 8-bit (or 6-bit) for batched evals where memory allows, and 4-bit for
single-prompt chat or when memory is tight. **Reject** 3-bit: a third worse on
perplexity for no speed. The mixed recipes are not worth their complexity here.

Not tested: mlx-lm's learned methods (DWQ, dynamic, AWQ, GPTQ). They need a calibration
run per model and were left out for time. They aim to make 4-bit behave more like 6-bit,
which would matter most for single-prompt use.

## 5. Training choices at a fixed time budget

Command: `uv run python -m bench.train_sweep --budget-s 150` (14M-parameter GPT, batch
32 x 512 tokens, 150 seconds of training each; then `--only confirm` for the last three
rows). The score is val_bpb, lower is better.

| Setup | val_bpb | Training tok/s | Steps |
|---|---|---|---|
| Muon rate 0.02, seed 0 | 1.7304 | 48,925 | 448 |
| Muon rate 0.02, seed 1 | 1.7433 | 48,577 | 445 |
| Muon rate 0.02, seed 2 | 1.7336 | 48,472 | 444 |
| AdamW only, rate 1e-3 | 2.1517 | 52,510 | 481 |
| AdamW only, rate 3e-3 | 2.0515 | 52,297 | 479 |
| AdamW only, rate 6e-3 | 2.2283 | 52,439 | 481 |
| Muon rate 0.01 | 1.7496 | 48,483 | 444 |
| Muon rate 0.04 | 1.6869 | 48,517 | 445 |
| Muon rate 0.08 | 1.8677 | 49,044 | 450 |
| Muon rate 0.02, bfloat16 | 1.7099 | 54,972 | 504 |
| Muon rate 0.02, float16 | NaN at step 77 | | |
| Muon rate 0.02, not compiled | 1.7560 | 44,308 | 406 |
| Muon rate 0.04, bfloat16, seed 0 | 1.6779 | 54,780 | 502 |
| Muon rate 0.04, bfloat16, seed 1 | 1.6870 | 54,977 | 504 |

**Noise.** The same setup with three seeds gave 1.7358 on average with a spread
(standard deviation) of 0.0067. A difference smaller than about 0.02 is not a result.

- **Muon against AdamW.** Muon is an optimizer (the rule that turns each error signal
  into a weight change) used on the big weight tables, with AdamW on the rest. It
  scored 0.32 better than the best of three AdamW rates, although it takes 7% fewer
  steps in the same time. Verdict: **keep** (already the default).
- **Muon rate.** 0.04 beat 0.02 by 0.049, seven times the noise, and all three runs at
  0.04 (1.6869, 1.6779, 1.6870) beat all three at 0.02. 0.08 was much worse. Verdict:
  **keep**; 0.04 is now the trainer's default.
- **bfloat16** (16-bit numbers). 13% more steps and half the memory (4.0 against 7.9
  GB in the baseline). At rate 0.02 it scored 0.026 better (one run); at rate 0.04 the
  two bfloat16 runs average 1.6825 against 1.6869 for 32-bit, which is inside the
  noise. So it is at least as good and cheaper. Verdict: **keep as an option**
  (`--dtype bfloat16`); it is required for the 91M size, where 32-bit runs out of
  memory. The default stays 32-bit until a score gain is shown at the new rate.
- **float16.** The loss became NaN in every run, here and in the baseline. **Reject.**
- **Compiled step.** 9% faster, so more steps and a better score (by 0.020, three
  times the noise). **Keep** (the default).

Limits: one model size, 150 seconds, and most rows are single runs. These are "best on
this Mac in this time", not general claims.

## 6. Other GPU code paths (both rejected)

Command: `uv run python -m bench.batch_overhead --model LABEL=PATH`
(for the second test, with `MLX_METAL_GPU_ARCH=applegpu_g15s` set and `--save-logits`).

**mxfp4 format instead of the usual 4-bit** (4B model). MLX runs this format through a
different GPU routine. ms per step:

| Prompts at once | usual 4-bit | mxfp4 |
|---|---|---|
| 1 | 10.5 | 11.6 |
| 2 | 15.2 | 14.7 |
| 4 | 26.5 | 23.6 |
| 8 | 59.3 | 59.8 |
| 32 | 70.4 | 73.3 |

11% faster at exactly 4 prompts, 10% slower for one, about the same elsewhere. Its
quality was not measured. Verdict: **reject**.

**Telling MLX the chip is a newer generation** (9B model), which switches on a routine
MLX disables for M1. The model's scores were bit-for-bit identical, and the speed was
worse: 36.4 against 24.4 ms at 2 prompts, 65.2 against 44.7 at 4, 135.6 against 105.7
at 8, and unchanged from 16 up. MLX's own choice for this chip is the right one.
Verdict: **reject**.

## 7. Vocabulary size for from-scratch training

The first trainer used GPT-2's 50,257-token vocabulary. With a 14M-parameter model body
it reached 12.7 GB, pushed the Mac into swap and ran at about 15,000 tok/s at best. The
lab now trains its own 8,192-token vocabulary (`lab/train/data.py`): 40,000 tok/s and
7.65 GB for the same model body, with Muon. These two numbers come from the smoke runs
made while building the trainer, not from a results file, so treat them as approximate.
Verdict: **keep**.

## 8. Gradient checkpointing when fine-tuning Qwen3.5-type models

Command: `uv run python -m bench.lora --model LABEL=PATH [--grad-checkpoint on|off]`
(LoRA, batch 4, up to 512 tokens, last 16 layers, 20 steps).

| Model | Checkpointing | tok/s | Peak memory |
|---|---|---|---|
| Qwen3-0.6B 4-bit | off | 1,460 | 2.0 GB |
| Qwen3.5-0.8B 4-bit | off | 205 | 18.3 GB |
| Qwen3.5-0.8B 4-bit | on | 219 | 3.9 GB |
| Qwen3-4B 4-bit | off | 313 | 5.1 GB |
| Qwen3.5-4B 4-bit | off | out of memory | |
| Qwen3.5-4B 4-bit | on (automatic) | 88 | 9.4 GB |
| Qwen3.5-9B 4-bit | on (automatic) | 56 | 12.0 GB |

Gradient checkpointing recomputes intermediate results instead of storing them. For
the Qwen3.5 family it cut memory by 79% at no cost in speed, and the training losses
were identical. `lab/finetune/lora.py` turns it on by itself for these models. Why
Qwen3.5 needs so much memory without it was not investigated. Even with it, Qwen3.5
models fine-tune 3.5 to 7 times slower per token than Qwen3 models of similar size.

Verdict: **keep**.

## 9. Not tested

- **Raising the GPU memory cap** (`iogpu.wired_limit_mb`). It needs `sudo`. Nothing
  measured here needed it: every model up to 27B at 4-bit loaded and ran under the
  default cap. The one job that failed (27B with 32 prompts at once) ran the whole Mac
  out of memory, which a higher cap would not fix.
- **Ollama against MLX** on the same model. Not run.
- **Learned quantization** (DWQ, dynamic, AWQ, GPTQ). Not run.
- **27B model, 32 prompts at once.** Tried; the watchdog stopped it when free memory
  fell to 4%. No number. Use 8 at once or fewer for this model (21.6 GB peak at 8).
