# Speed results: what was tested, and what to keep

Every idea below was run on this Mac (Apple M1 Max, 32 GB, MLX 0.32.4, mlx-lm 0.32.2) on
2026-11-11 and 2026-11-12. Each section gives the command, the numbers (from `bench/results/*.json`)
and a verdict: **keep**, **reject**, or **not settled**.

Words used here:

- **tok/s**: tokens per second. A token is a piece of a word, about four characters.
- **batch**: several prompts answered at the same time on the GPU.
- **row**: one prompt's share of a batch. 8 prompts decoded together are 8 rows.
- **kernel**: a small program that runs on the GPU. MLX ships its own; section 9 adds one.
- **end to end**: counting all the time, both reading the prompts and writing the answers.
- **quantization, N-bit**: storing each weight of a model in N bits instead of 16, to
  make it smaller.
- **perplexity**: how surprised a model is by real text. Lower is better.
- **val_bpb**: validation bits per byte, the trainer's score. Lower is better.
- **forward pass, backward pass**: running the model on some text, and then working
  backwards through it to find how each weight should change. Training does both.
- **LoRA**: fine-tuning that leaves a model's weights alone and trains small add-on
  tables in its last layers.
- **dense**: weights stored as plain 16-bit or 32-bit numbers, not quantized.
- **the chip's arithmetic limit**: the most multiply-adds this GPU can do, 5.3 trillion
  per second (section 11).

## Summary

| # | Idea | Result | Verdict |
|---|---|---|---|
| 1 | Answer 32 prompts at once | 3.2x faster end to end than one at a time (4B model) | keep |
| 1 | Sort prompts by length | 9% faster on top of that (one run) | keep, now the default |
| 1 | Batches of 8 or 64 | 8 is no faster than 4; 64 is no faster than 32 | reject |
| 2 | Read a shared prompt opening once | 5.0x faster end to end on long shared prompts | keep, automatic |
| 3 | Speculative decoding (small model drafts) | 11-19% slower on Qwen3 with MLX's kernels; 5-9% faster with the lab kernel | reject without the kernel |
| 4 | 8-bit or 6-bit instead of 4-bit for batch work | same speed at batch 32, quality nearly untouched | keep |
| 4 | 3-bit | 34% worse perplexity for no speed gain | reject |
| 5 | Muon optimizer for training | 0.32 val_bpb better than the best AdamW in the same time | keep, the default |
| 5 | Muon rate 0.04 instead of 0.02 | 0.05 val_bpb better, 7 times the noise | keep, now the default |
| 5 | 16-bit training numbers (bfloat16) | 13% more steps, half the memory, score no worse | keep as an option |
| 5 | float16 training | loss became NaN every time | reject (cause found and fixed in 12) |
| 5 | Compiled training step | 9% faster | keep, the default |
| 6 | A different GPU code path (mxfp4 format, architecture override) | no gain, or slower | reject |
| 7 | Small vocabulary for from-scratch training | 2.6x faster, 40% less memory | keep, the default |
| 8 | Gradient checkpointing for Qwen3.5 fine-tuning | 18.3 GB down to 3.9 GB, same speed | keep (13 removes most of the need) |
| 9 | A new GPU kernel for 2-18 prompts at once | 1.3-1.7x faster at 4 prompts, 2.1-2.6x at 8; the 27B model's best speed doubles | keep, one line to switch on |
| 10 | Guess-and-check decoding for one prompt | 1.75-2.5x faster when the answer copies earlier text; no change on fresh text | keep, needs the kernel |
| 11 | Decode quantized weights once when 256 or more rows arrive (reading prompts, scoring) | prompts read 1.2-1.33x faster, same scores, 1-4 GB more memory while it runs | keep, on with the lab kernel |
| 12 | Training with 32-bit weights and 16-bit arithmetic | 24% more training per second, 41% less memory, a better score in the same time | keep, now the default |
| 13 | Fine-tuning through the lab: its GPU paths, Qwen3.5's recurrent layers done 64 tokens at a time, sorted batches | Qwen3-4B 1.3-1.5x; Qwen3.5 2.6-5.5x on short examples and 6-10x on full-length ones; Qwen3.5-4B at full length no longer runs out of memory | keep, the default |
| 14 | Fine-tuning with the frozen bottom layers computed once | 1.2-2.6x per step after a one-off cache (most when few layers are trained); pays back after about 290-470 steps | keep, for sweeps and repeated passes |
| 15 | Stop after a layer, resume from a layer, score only the last token | collecting an early layer 6x faster than a full pass; sweeps from three quarters up 5.3x faster | keep |
| 16 | float16 instead of bfloat16 for inference | prompts read 1.4x faster with MLX's kernels and 1.5x with the lab's on top; 32 prompts at once 1.18-1.31x; a 64-question eval 1.29x faster end to end; one prompt unchanged; the model's values must stay under 65,504 | keep, one line, checks first |
| 17 | Eight smaller ideas (other kernel layouts, resident decoded weights, compiled steps, two trainings at once, ...) | 0-10% each | reject |
| 18 | Raising the GPU memory cap | not tested | not settled |

**Was a new, faster technique found?** Yes, several, each with clear edges.

- **A GPU kernel for small batches** (section 9). On this chip MLX pays for 32 prompts
  whenever it decodes 8, and re-reads all the weights for each of 2-6 prompts. The lab's
  kernel removes both: a step now costs the first prompt plus about a fifth of that for
  each extra one, up to 8. Section 10 uses those cheap extra rows to make a single
  prompt faster when its answer repeats earlier text.
- **Qwen3.5's recurrent layers worked out 64 tokens at a time** (section 13). The same
  arithmetic rearranged into ordinary matrix multiplies, so that training no longer
  needs a loop over the tokens: 25 times faster per layer than mlx-lm's training path,
  and fine-tuning these models is 2.6 to 10 times faster end to end.
- **Two findings about number formats** (sections 11, 12 and 16), from measuring how
  close each kind of GPU work runs to the chip's arithmetic limit. MLX's kernel for
  quantized weights decodes them again for every 32 rows, and MLX converts bfloat16
  numbers in software. Decoding once and running in float16 give 1.2 to 1.5x on prompt
  reading, 1.18 to 1.31x at 32 prompts at once, and 24% in from-scratch training.

What none of this changes: one prompt writing fresh text. That is limited by how fast
the Mac reads memory (about 375 GB/s, measured) and by the cost of issuing a thousand
small GPU operations per token (section 17). Everything here was measured only on this
Mac and with these versions of MLX and mlx-lm.

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

**What came of it.** Section 9 builds the missing kernel inside the lab, so the rule
"never 8" only holds for MLX's own kernels. Two things there may interest the MLX
project: the small-batch gap on M1 and M2 generation GPUs, and the extra rounding in
its 32-row kernel. The MLX project does not accept AI-written issue or PR text, so any
report has to be written by you.

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

| Big model + draft | Drafted tokens per step | tok/s, MLX's kernels (spread) | tok/s, lab kernel on the big model (spread) | Tokens accepted from the draft |
|---|---|---|---|---|
| Qwen3-4B 4-bit + Qwen3-0.6B 4-bit | none | 105.3 (1.1) | 105.2 (0.6) | |
| | 2 | 93.5 (5.2) | 110.6 (7.5) | 62% |
| | 3 | 89.8 (7.4) | 115.0 (9.4) | 69% |
| | 4 | 85.8 (10.9) | 113.2 (15.1) | 74% |
| Qwen3.5-9B 4-bit + Qwen3.5-0.8B 4-bit | none | 61.2 (0.1) | | |
| | 2, 3, 4 | failed | | |

With MLX's kernels the draft guessed right 62 to 74% of the time and it was still slower.
The reason, found later (section 9): checking n guesses is a pass over n+1 rows, and
MLX's kernels on this chip run that as n+1 sweeps of the weights. With the lab kernel
on the big model (`--rows-kernel`) the same setup is 5 to 9% faster than no draft. The
gain stays small because the draft model has its own cost for every token. For the
Qwen3.5 family mlx-lm refuses outright ("requires a trimmable prompt cache").

Verdict: **reject** with MLX's kernels. With the lab kernel it is a small gain; guesses
that cost nothing (section 10) do much better.

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
  (Section 12 later found the cause, a sum that overflows, and a setup where float16
  arithmetic is the fastest and best-scoring option.)
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

Section 13 later found the cause of both the memory and the slowness (mlx-lm trains
these models' recurrent layers with a loop over the tokens) and replaced that loop. The
numbers above are what MLX and mlx-lm give by themselves.

## 9. A new GPU kernel for 2 to 18 prompts at once

Commands: `uv run python -m bench.kernel_rows --model LABEL=PATH` (speed, and agreement on
real text) and `uv run python -m bench.kernel_exact` (accuracy against the exact answer).
Code: `lab/kernels/qmm.py`. Switch it on with `qmm.install(model)`, or serve through
`uv run python -m lab.serve`.

**The problem.** Every weight table of a model is used as y = x @ W.T, where x has one
row per prompt being decoded. MLX's source shows what it does on M1-generation GPUs:
for 2 to 6 rows it runs its one-row kernel once per row, so every row decodes all the
weights again; from 7 to 14 rows (by layer size) it switches to a kernel that always
computes a block of 32 rows. MLX does have a kernel meant for a few rows, but switches
it off on this chip generation because it is slower here (confirmed in section 6).

**The kernel.** One pass over the weights serves all the rows. Each GPU thread keeps
its slice of every row's inputs in registers (the chip's fastest storage), decodes each
weight once and uses it for every row. Up to 8 rows fit in one pass; 9 to 18 rows run
as 2 or 3 passes. It handles 3, 4, 5, 6 and 8-bit weights. One row, and 19 rows or
more, are left to MLX's kernels.

**Result in a real decoding loop** (greedy decoding from 64-token prompts, 48 timed
steps; tok/s with MLX's kernels → with the lab kernel):

| Prompts at once | Qwen3.5-0.8B 4-bit | Qwen3.5-4B 4-bit | Qwen3-4B 4-bit | Qwen3.5-9B 4-bit | Qwen3.5-9B 8-bit | Qwen3.8-27B 4-bit |
|---|---|---|---|---|---|---|
| 1 | 180 → 180 | 78 → 77 | 89 → 89 | 52 → 53 | 33 → 33 | 18 → 18 |
| 2 | 293 → 283 (0.96x) | 114 → 121 (1.06x) | 129 → 142 (1.10x) | 75 → 88 (1.16x) | 59 → 62 (1.05x) | 25 → 31 (1.22x) |
| 3 | 365 → 387 (1.06x) | 125 → 154 (1.23x) | 143 → 173 (1.21x) | 80 → 111 (1.38x) | 64 → 90 (1.41x) | 27 → 38 (1.41x) |
| 4 | 415 → 470 (1.13x) | 134 → 179 (1.34x) | 150 → 208 (1.39x) | 84 → 130 (1.54x) | 69 → 117 (1.71x) | 28 → 44 (1.60x) |
| 6 | 487 → 574 (1.18x) | 144 → 232 (1.61x) | 160 → 259 (1.62x) | 87 → 152 (1.76x) | 71 → 141 (1.99x) | 28 → 52 (1.85x) |
| 8 | 527 → 720 (1.37x) | 125 → 259 (2.07x) | 138 → 289 (2.09x) | 72 → 172 (2.38x) | 70 → 160 (2.27x) | 22 → 57 (2.64x) |
| 12 | 667 → 861 (1.29x) | 171 → 270 (1.58x) | 186 → 311 (1.68x) | 101 → 174 (1.71x) | 101 → 164 (1.62x) | not run |
| 16 | 821 → 960 (1.17x) | 224 → 303 (1.35x) | 244 → 343 (1.41x) | 134 → 191 (1.43x) | 132 → 183 (1.38x) | not run |
| 18 | 898 → 996 (1.11x) | 245 → 294 (1.20x) | 271 → 331 (1.22x) | 149 → 182 (1.22x) | 147 → 173 (1.17x) | not run |
| 24 | 1143 → 1129 | 315 → 314 | 355 → 355 | 194 → 193 | not run | not run |
| 32 | 1447 → 1440 | 429 → 439 | 493 → 492 | 270 → 270 | not run | not run |

How to read it:

- **4 at once: 1.3 to 1.7x. 8 at once: 2.1 to 2.6x. 16 at once: about 1.4x.** One
  prompt and 24 or more are unchanged, as designed.
- **The 27B model's best speed doubles**: 57 tok/s at 8 prompts, against a best of 28
  with MLX's kernels. This model cannot run 32 at once in 32 GB, so small batches are
  all it has.
- **The 0.8B model gains least**, and loses 4% at 2 prompts: most of its step is not
  spent in the weight tables.
- A locally made 6-bit copy of Qwen3.5-4B gave 101 → 175 tok/s at 4 prompts and 118 →
  240 at 8.
- Through the server with a real eval (Inspect, 20 MATH-500 questions, 8 requests at a
  time, Qwen3.5-9B 4-bit): 73 s with `lab.serve` against 138 s with `mlx_lm.server`,
  accuracy 0.65 both times.

The same thing layer by layer (total ms to run all of a model's weight tables once,
MLX → lab). A decoding step also does other work that the kernel does not touch (about
5 ms per step for the 9B model), so the gain in the decoding loop above is smaller than
the gain here. The decoding loop is the number that counts.

| Rows | Qwen3.5-0.8B 4-bit | Qwen3.5-4B 4-bit | Qwen3-4B 4-bit | Qwen3.5-9B 4-bit |
|---|---|---|---|---|
| 1 | 2.1 | 6.9 | 7.0 | 14.3 |
| 2 | 2.2 → 2.1 | 11.0 → 8.6 | 10.7 → 9.2 | 21.8 → 17.3 |
| 4 | 3.7 → 2.9 | 21.8 → 12.4 | 20.8 → 12.5 | 42.0 → 25.3 |
| 8 | 6.9 → 5.1 | 48.1 → 19.2 | 46.7 → 20.1 | 100.4 → 38.6 |
| 12 | 9.0 → 5.9 | 50.9 → 28.6 | 51.6 → 29.5 | 105.6 → 59.0 |
| 16 | 8.8 → 7.0 | 50.8 → 35.8 | 52.1 → 36.3 | 105.0 → 71.0 |
| 18 | 8.9 → 8.1 | 51.3 → 42.4 | 51.8 → 42.5 | 106.0 → 86.6 |
| 32 | 8.6 | 49.2 | 49.3 | 99.5 |

**Accuracy.** Three checks.

1. Against the exact answer (weights fully decoded, 32-bit maths) on the 9B model's
   real layers. The outputs are 16-bit numbers, so the best any kernel can do is the
   exact answer rounded once: an error of 0.165% of the typical output. The lab kernel
   and MLX's one-row kernel both sit exactly there. MLX's 32-row block kernel is at
   0.217%, a third worse, because it rounds every decoded weight to 16 bits before
   multiplying. So for 7 to 18 rows the lab kernel is the more exact of the two.
2. On real text, 8 passages read together, one token per step, identical inputs for
   both kernels at every step:

   | Model | Same next token, lab vs MLX | Same next token, MLX (8 together) vs MLX (one at a time) | Loss with MLX | Loss with lab |
   |---|---|---|---|---|
   | Qwen3.5-0.8B 4-bit | 98.0% | 98.2% | 3.2162 | 3.2160 |
   | Qwen3.5-4B 4-bit | 98.3% | 99.0% | 2.7094 | 2.7077 |
   | Qwen3-4B 4-bit | 98.2% | 98.3% | 2.9393 | 2.9413 |
   | Qwen3.5-9B 4-bit | 98.6% | 98.0% | 2.5738 | 2.5746 |
   | Qwen3.5-9B 8-bit | 99.0% | 98.2% | 2.5261 | 2.5251 |
   | Qwen3.8-27B 4-bit | 98.3% | 98.6% | 2.4987 | 2.5000 |

   The lab kernel differs from MLX exactly as much as MLX's two own kernels differ from
   each other, and the model's loss (lower is better) is the same to three decimals.
   These models keep 8 significant bits between layers, so any change in the order of a
   sum moves a near-tie now and then.
3. On random-token prompts, where the model is unsure at every step, the two runs
   wrote identical tokens in 50 to 100% of positions. One changed token changes all that
   follow, so this number says little; it is in the results file for completeness.

**What limits it.** Command: `uv run python -m bench.kernel_limits` (the kernel built up one
step at a time on the 9B model's weight tables; total ms for all layers).

| Step added | 1 row | 4 rows | 8 rows |
|---|---|---|---|
| Start threads, loop, store | 2.2 | 2.0 | 2.4 |
| + read the weights from memory | 12.8 | 13.0 | 13.1 |
| + unpack and decode them | 14.8 | 14.9 | 14.9 |
| + read the rows of x into registers | 16.4 | 23.1 | 35.4 |
| + multiply-adds (the full kernel) | 15.7 | 25.6 | 38.7 |

- **One prompt cannot be helped by a kernel.** Just reading the 3.97 GB of weights
  takes 10.6 ms of the 15.7 ms: 375 GB/s, close to the chip's limit of 400 GB/s.
- **Extra rows cost about 3.5 ms each on the 9B model.** Most of that is reading the
  row's inputs again from the chip's cache for every group of 4 outputs; the
  multiply-adds add little on top.
- **Registers.** A GPU thread has 128 registers, and the more a kernel uses, the fewer
  threads the GPU runs at once. 8 rows per pass is the most that fits. Tried during
  development without gain: 8 outputs per thread group, 16-bit registers, different
  addressing, wider loads, fast-math.
- **From about 19 rows MLX's 32-row kernel is as fast**, so it takes over there.

**Limits of the result.** Measured only on the M1 Max and MLX 0.32.4; MLX picks other
kernels on M3 and later chips, so run the benchmark before trusting it elsewhere.
Inference only (no gradient, so layers fall back to MLX's kernels in training mode).
The first use of each batch size compiles a kernel, which takes a fraction of a second.
It relies on MLX's custom-kernel feature and its packed weight layout, so re-run
`uv run pytest tests/test_kernels.py` after an MLX upgrade.

Verdict: **keep**. Switch it on for any work with 2 to 18 prompts at once.

## 10. Guess-and-check decoding for one prompt

Commands: `uv run python -m bench.speed_lookahead [--model ...] [--no-thinking]` and, for a
draft model, `uv run python -m bench.speed_speculative --rows-kernel ...` (section 3).
Code: `lab/infer/lookahead.py`.

**The idea.** One prompt is limited by reading all the weights once per token. Section
9 made extra rows in the same pass cheap. So guess the next few tokens, let the model
check up to 7 guesses in one pass, and keep the ones it agrees with: each is a token
that needed no pass of its own. The guesses here are free: when the last 3 tokens
written also appear earlier in the prompt or the output, the tokens that followed them
there are the guess ("prompt lookup"). The model's own choice is always what gets
written, so the text is that of normal greedy decoding.

**Result** (tok/s for one prompt, four prompts per task, up to 400 new tokens each):

| Model | Task | Plain | Lookahead, MLX's kernels | Lookahead, lab kernel | Guesses accepted | Same text as plain |
|---|---|---|---|---|---|---|
| Qwen3-4B 4-bit | rewrite a function | 88.6 | 103.3 | 158.6 (1.79x) | 79% | 4 of 4 |
| | quote from a passage | 85.5 | 118.6 | 212.8 (2.49x) | 92% | 4 of 4 |
| | solve a maths problem | 87.2 | 76.9 | 89.6 (1.03x) | 32% | 0 of 4 |
| Qwen3.5-9B 4-bit | rewrite a function | 52.5 | 56.3 | 91.9 (1.75x) | 82% | 3 of 4 |
| | quote from a passage | 51.7 | 54.5 | 90.6 (1.75x) | 81% | 4 of 4 |
| | solve a maths problem | 51.8 | 47.9 | 52.0 (1.00x) | 49% | 3 of 4 |

- **When the answer copies earlier text it is 1.75 to 2.5x faster**; on fresh text
  (maths) it neither helps nor hurts.
- **Without the lab kernel it is not worth using**: checking 7 guesses then costs
  almost 8 passes.
- **The Qwen3.5 family needed extra work.** Part of its memory of the text cannot be
  rewound after a wrong guess (which is why mlx-lm refuses speculative decoding for
  it). The decoder saves that part before each check, restores it after a miss and runs
  the accepted tokens again. A miss costs a whole pass there, so it starts with short
  guesses and stops guessing for a while after each miss.
- **"Same text" is not always 4 of 4** on long free-form answers. A pass over several
  rows rounds differently from a one-row pass (section 9, accuracy), a near-tie flips,
  and everything after it differs. It happens with MLX's own kernels too (1 of 4 on
  maths). Both texts are valid greedy outputs of the same model.

Limits: greedy decoding only; one prompt at a time; guesses come only from repeated
text, so the gain depends on the task.

Verdict: **keep** for editing, extraction and other copy-heavy work, with the lab
kernel switched on.

## 11. Reading prompts: decode the weights once, then use the dense multiply

Commands: `uv run python -m bench.prefill --model LABEL=PATH` (speed, memory, agreement) and
`uv run python -m bench.gemm_limits` (how close each kernel runs to the chip's limit).
Code: the many-rows path in `lab/kernels/qmm.py`. `qmm.install(model)` switches it on
together with the small-batch kernel of section 9.

**The problem.** Reading a prompt, scoring a text and the frozen part of fine-tuning all
run every weight table on hundreds or thousands of rows at once, one row per token. That
work is limited by arithmetic, not by memory. The chip's arithmetic limit is known from
public measurements of this GPU: 32 cores x 128 multiply-adds per clock cycle x 1.296
billion cycles per second = 5.3 trillion multiply-adds per second. Against that limit:

| Kernel | Share of the limit it reaches |
|---|---|
| MLX's multiply for unquantized ("dense") weights, 16-bit float16 | 84-87% |
| The same, 32-bit | 71-81% |
| The same, bfloat16 (the other 16-bit format) | 71-74% |
| MLX's multiply for quantized weights, 512 rows or more | 49-52% |
| Decode the quantized weights to a temporary dense table, then the dense multiply, 512 rows | 52-68% |
| The same, 2,048 rows | 59-70% |

MLX's quantized kernel decodes each block of weights again for every 32 rows, which holds
it at half the limit however many rows there are. Decoding a whole table once costs a
fixed quarter of a millisecond (0.24 ms for a table of 34 million weights); after that
the dense kernel does the work, and the more rows there are the less the decoding
matters.

**The path.** From 256 rows up, a quantized layer decodes its weights and calls the dense
multiply. Below 256 rows MLX's quantized kernel is as fast or faster, so nothing changes
there. The decoded table is temporary and is freed as soon as the layer has run. It has a
gradient, so fine-tuning uses it too (section 13).

**Result** (tokens read per second, MLX's kernels → lab):

| Prompts x tokens each | Qwen3.5-0.8B 4-bit | Qwen3.5-4B 4-bit | Qwen3-4B 4-bit | Qwen3.5-9B 4-bit |
|---|---|---|---|---|
| 1 x 512 | 3,085 → 3,467 (1.12x) | 608 → 742 (1.22x) | 649 → 814 (1.25x) | 328 → 400 (1.22x) |
| 1 x 2,048 | 3,092 → 3,744 (1.21x) | 609 → 774 (1.27x) | 633 → 826 (1.30x) | 335 → 423 (1.26x) |
| 8 x 512 | 3,238 → 4,196 (1.30x) | 622 → 827 (1.33x) | 671 → 895 (1.33x) | 337 → 433 (1.28x) |
| 32 x 256 | 3,157 → 4,057 (1.29x) | 623 → 826 (1.33x) | 675 → 899 (1.33x) | 335 → 428 (1.28x) |

- **1.2 to 1.33x faster** from 4B up; the 0.8B model gains a little less.
- **Same answers.** In 12 of the 16 cases the scores for the last token are bit-for-bit
  identical to MLX's. In the other four (the 1 x 512 row) they differ by at most 0.19 on
  scores that spread over 1.1 to 1.8, and the top token is the same. MLX's block kernel
  also rounds every decoded weight to 16 bits, so both compute the same products; only
  the order of the sums can differ.
- **It costs memory while it runs.** The temporary tables add 0.9 to 3.8 GB to the peak.
  For 32 x 256 tokens the 4B model goes from 10.6 to 13.7 GB and the 9B from 13.1 to
  16.9 GB.

Verdict: **keep**. It is on whenever `qmm.install(model)` is, and the fine-tuning scripts
switch it on themselves.

## 12. Training in 16-bit arithmetic with 32-bit weights

Command: `uv run python -m bench.train_sweep --budget-s 150 --only precision` (the model
and time budget of section 5, Muon rate 0.04). Code: `lab/train/train.py` (`--dtype mixed16`,
now the default) and `lab/train/model.py`.

**Why float16 failed before.** Section 5 rejected float16 because the loss became NaN (not
a number) in every run. The cause: the trainer added up the losses of all 16,384 tokens
of a batch as one 16-bit number, and a 16-bit float cannot hold a number above 65,504.
The arithmetic inside the model was never the problem.

**The setup ("mixed precision").** The weights and the optimizer's memory stay 32-bit.
Each layer converts its weight to float16 as it uses it, so the data flowing through the
model is 16-bit, and the loss is worked out in 32-bit. On the way back the error signal
is scaled up (by 8 per token) and scaled down again before the weights change, so that
small values do not round to zero in 16 bits. It is faster for two reasons that section
11 measured: MLX's float16 multiply runs nearer the chip's limit than its 32-bit one, and
everything else moves half as many bytes.

| Setup | val_bpb | Training tok/s | Steps in 150 s | Peak memory |
|---|---|---|---|---|
| 32-bit throughout, seed 0 | 1.6799 | 49,010 | 449 | 7.7 GB |
| 32-bit throughout, seed 1 | 1.6946 | 48,719 | 447 | 7.7 GB |
| **Mixed: 32-bit weights, float16 arithmetic**, seed 0 | 1.6527 | 59,911 | 549 | 4.5 GB |
| **Mixed**, seed 1 | 1.6436 | 60,815 | 557 | 4.5 GB |
| bfloat16 throughout (section 5's option) | 1.6760 | 54,010 | 495 | 4.5 GB |
| Mixed with bfloat16 arithmetic | 1.6594 | 53,745 | 493 | 4.5 GB |

- **24% more training per second and 41% less memory** than 32-bit.
- **A better model in the same time**: 1.648 on average against 1.687, a gap of 0.039,
  more than five times the noise of section 5 (0.007). Both mixed runs beat both 32-bit
  runs.
- It also beats bfloat16 throughout, on speed (float16 multiplies are faster) and on
  score (0.028, one run).

Limits: one model size (14 million parameters), 150 seconds, two seeds.

Verdict: **keep**; `mixed16` is now the trainer's default. `--dtype float32` gives the old
behaviour.

## 13. Fine-tuning: the lab's GPU paths, Qwen3.5's recurrent layers, and better batches

Commands: `uv run python -m bench.lora --model LABEL=PATH [--mlx-kernels] [--file-order] [--long-text]`
(whole fine-tunes; one model per run, so every run starts in a fresh process) and
`uv run python -m bench.gated_delta` (one recurrent layer by itself).
Code: `lab/finetune/lora.py`, `lab/finetune/gated_delta.py`.

`lab.finetune.lora` wraps mlx-lm's trainer. It now adds three things, all on by default.

**1. The frozen layers use section 11's path.** A fine-tuning batch is hundreds or
thousands of rows, so every frozen quantized layer decodes its weights once and uses the
dense multiply, in the forward and in the backward pass.

**2. Qwen3.5's recurrent layers are worked out 64 tokens at a time.** Three of every
four layers of a Qwen3.5 model carry a running summary of the text forward, one token at
a time. For generating text mlx-lm has a fast GPU kernel for that. For training it falls
back to a plain Python loop over the tokens, because its kernel cannot run backwards.
That loop is why section 8 found these models 3.5 to 7 times slower to fine-tune and so
hungry for memory. MLX itself has a built-in version with a backward pass, but on this
chip generation that backward pass also goes one token at a time and stores every step.

The lab replaces the loop with the same update rearranged so that 64 tokens are done at
once with ordinary matrix multiplies, which MLX can run backwards by itself. It is the
same arithmetic, not an approximation: outputs and gradients match the loop's to
rounding (`tests/test_gated_delta.py`). One recurrent layer of the 4B model's shape, a
batch of 4, forward plus backward:

| Tokens | mlx-lm's loop | MLX's built-in | Lab, 64 at a time |
|---|---|---|---|
| 100 | 86 ms, 2.4 GB | 66 ms, 1.7 GB | 6.3 ms, 0.4 GB |
| 256 | 244 ms, 4.8 GB | 152 ms, 4.2 GB | 12.4 ms, 0.8 GB |
| 512 | 637 ms, 10.2 GB | 299 ms, 8.4 GB | 25.1 ms, 1.3 GB |
| 1,024 | not run (about 20 GB) | 611 ms, 8.5 GB | 50.8 ms, 1.9 GB |

At 512 tokens that is 25 times faster than the loop and 12 times faster than MLX's
built-in, in an eighth of the memory, and the cost now grows in step with the length.

**3. Batches of examples of similar length.** mlx-lm means to group examples by length
so that little of each batch is padding, but it measures the length of the raw record
(a dictionary with one or two entries, the same for every row), so its batches come in
file order. On the WikiSQL set it computes 1.30 positions for every real token; with a
real sort it is 1.18. `--file-order` gives mlx-lm's behaviour back.

**Whole fine-tunes** (LoRA on the last 16 layers, batches of 4; tok/s counts real tokens,
not padding).

Short examples (WikiSQL, 58 to 215 tokens each, 20 steps):

| Model | mlx-lm as it comes | Lab GPU paths, mlx-lm's batch order | Lab defaults (GPU paths + sorted batches) | Speed-up |
|---|---|---|---|---|
| Qwen3-4B 4-bit | 289 tok/s, 5.1 GB | 369 tok/s, 8.6 GB | 422 tok/s, 8.1 GB | 1.5x |
| Qwen3.5-0.8B 4-bit | 212 tok/s, 3.9 GB | not run | 1,172 tok/s, 4.0 GB | 5.5x |
| Qwen3.5-4B 4-bit | 86 tok/s, 9.4 GB | 292 tok/s, 12.2 GB | 338 tok/s, 10.9 GB | 3.9x |
| Qwen3.5-9B 4-bit | 64 tok/s, 12.0 GB | not run | 166 tok/s, 14.1 GB | 2.6x |

- **Qwen3-4B: 1.5x.** The GPU paths give 1.28x and sorted batches another 1.14x.
- **Qwen3.5: 2.6 to 5.5x**, almost all of it from the recurrent layers. The 9B model
  gains least because the lab keeps checkpointing on for it (see below).

Full-length examples (passages cut at 512 tokens; 10 steps, 16 for the one-example row):

| Model, examples per step x tokens | mlx-lm as it comes | Lab defaults | Speed-up |
|---|---|---|---|
| Qwen3-4B 4-bit, 4 x 512 | 396 tok/s, 10.1 GB | 516 tok/s, 13.8 GB | 1.3x |
| Qwen3.5-0.8B 4-bit, 4 x 512 | 144 tok/s, 16.3 GB | 1,505 tok/s, 10.6 GB | 10.4x |
| Qwen3.5-4B 4-bit, 4 x 512 | out of memory | 345 tok/s, 9.7 GB |  |
| Qwen3.5-4B 4-bit, 1 x 512 | 67 tok/s, 10.3 GB | 389 tok/s, 11.0 GB | 5.8x |
| Qwen3.5-9B 4-bit, 4 x 512 | not run (the 4B already runs out of memory) | 200 tok/s, 16.3 GB |  |

- **With mlx-lm alone, Qwen3.5-4B cannot be fine-tuned on four full-length examples at a
  time on this Mac**: it runs out of GPU memory even with checkpointing. With one
  example per step it runs at 67 tok/s. The lab runs four per step at 345 tok/s in
  under 10 GB.
- The 0.8B model needed 16.3 GB with mlx-lm and runs ten times faster in 10.6 GB (or 8.6
  times faster in 4.3 GB with checkpointing).
- Sorting changes nothing here (every example is cut at 512 tokens), so these gains are
  the GPU paths alone.

**Memory, and when to trade speed for it.** Gradient checkpointing (section 8) recomputes
intermediate results instead of keeping them. With the loop gone it is no longer always
needed, and on short examples it costs speed. On long ones it is both lighter and faster,
because the Mac stops running out of memory:

| Model, batch | Without checkpointing | With checkpointing | The lab's own choice |
|---|---|---|---|
| Qwen3.5-4B, 4 short examples | 338 tok/s, 10.9 GB | 281 tok/s, 7.9 GB | without |
| Qwen3.5-4B, 4 x 256 tokens | 405 tok/s, 14.8 GB | 341 tok/s, 8.5 GB | with |
| Qwen3.5-4B, 4 x 512 tokens | 280 tok/s, 22.6 GB | 347 tok/s, 9.7 GB | with |
| Qwen3.5-0.8B, 4 x 512 tokens | 1,505 tok/s, 10.6 GB | 1,245 tok/s, 4.3 GB | without |
| Qwen3.5-9B, 4 short examples | 201 tok/s, 17.6 GB | 166 tok/s, 14.1 GB | with |

`lab.finetune.lora` decides by itself from an estimate of the peak (3 GB + 2.5 x the
weights + what the trained recurrent layers would keep) and switches checkpointing on
when that passes 45% of the Mac's memory. `--grad-checkpoint on|off` overrides it.

Two things found on the way that are MLX's or mlx-lm's to fix (any report has to be
written by you):

- mlx-lm switches checkpointing on by changing the layer's class, so it stays on for
  every later model in the same Python process and stacks up if switched on twice. The
  lab now undoes it after each run. Tonight's baseline numbers were re-measured with one
  process per run and did not change.
- MLX turns a split into more than 10 parts into separate slices, and the backward pass
  of each slice rebuilds the whole array. The cost of anything split that way grows with
  the square of the number of parts. The lab's update splits in two levels to avoid it.

Limits: batch 4, LoRA rank 8, 10 to 20 steps per run, one run each. The dense path adds
about 3 GB to the peak. The speed-ups for Qwen3.5 are over mlx-lm 0.32.2; a later mlx-lm
may fix its training path.

Verdict: **keep**; all three are on by default.

## 14. Fine-tuning with the frozen bottom layers computed once

Command: `uv run python -m bench.lora_cached --model LABEL=PATH` (WikiSQL, batch 4, 30
steps, LoRA in the last 16, 8 or 4 layers). Code: `lab/finetune/cached.py`.

**The idea.** LoRA usually trains add-on tables in the last few layers only. The layers
below them never change, so for a given example they always produce the same result.
This script runs those bottom layers once for every batch of examples, keeps the results
(in memory and on disk), and then trains with only the top layers in the loop.

It is the same training, not an approximation: before it starts, the script checks that
its split forward pass gives the same loss as the normal one, and stops if not. In the
table, both runs were scored on the whole validation set before and after training. The
cached runs end 0.04 to 0.08 lower in all six cases. The likely reason is bookkeeping,
not better training: mlx-lm scores the validation set before its last training step, so
its number is one step behind, and the loss is still falling fast at step 30.

| Model | Layers trained | Normal tok/s | Cached tok/s | Speed-up | Time to build the cache | Pays back after | GPU peak, normal → cached | Validation loss before → after, normal | The same, cached |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-4B 4-bit | last 16 | 418 | 534 | 1.28x | 76 s | 402 steps | 8.1 → 6.6 GB | 2.821 → 1.240 | 2.821 → 1.179 |
| Qwen3-4B 4-bit | last 8 | 495 | 911 | 1.84x | 105 s | 314 steps | 5.8 → 5.5 GB | 2.821 → 1.394 | 2.821 → 1.346 |
| Qwen3-4B 4-bit | last 4 | 544 | 1,433 | 2.63x | 120 s | 290 steps | 4.7 → 4.9 GB | 2.821 → 1.514 | 2.821 → 1.479 |
| Qwen3.5-4B 4-bit | last 16 | 333 | 386 | 1.16x | 71 s | 470 steps | 10.9 → 10.6 GB | 2.587 → 1.237 | 2.587 → 1.159 |
| Qwen3.5-4B 4-bit | last 8 | 399 | 636 | 1.60x | 106 s | 308 steps | 7.6 → 8.1 GB | 2.587 → 1.368 | 2.587 → 1.312 |
| Qwen3.5-4B 4-bit | last 4 | 439 | 909 | 2.07x | 123 s | 286 steps | 5.9 → 6.9 GB | 2.587 → 1.459 | 2.587 → 1.402 |

- **The fewer layers are trained, the bigger the gain.** The output layer over the
  151,936-word vocabulary is a fixed cost of every step, which is why training a quarter
  of the layers is not four times faster.
- **It pays when examples are seen more than once.** Building the cache takes 71 to 123
  seconds here, as much as 290 to 470 training steps. After that every pass is faster,
  and a second run on the same data and split point (another learning rate, rank or
  seed) loads the cache from disk in about a second.
- The cache file is named after the model's settings, its frozen weights, the split
  point and the exact tokens, so editing the model or the data never reuses a stale one.

Limits: the peak-memory column is GPU memory while training; the cache itself sits in
ordinary memory on top of it (0.6 GB for these 1,100 short examples, and it grows with
the tokens in the dataset; the script refuses above 6 GB). Prompt masking is not
supported. Tested on the Qwen3 and Qwen3.5 families only.

Verdict: **keep** for repeated passes and sweeps; use the normal script for one pass.

## 15. Looking inside a model without running layers that are not needed

Command: `uv run python -m bench.interp --model LABEL=PATH` (8 prompts x 256 tokens).
Code: `lab/edit/hooks.py` (`run_to`, `run_from`, `last_token_scores`, `save_activations`).

Interpretability work repeats two jobs. **Collecting**: record what one layer writes for
a whole dataset, to train probes or sparse autoencoders on. **Sweeping**: change
something at or above a layer and read the model's scores again, for many variants on
the same prompts. A plain forward pass runs every layer and scores every position both
times.

- `run_to(model, tokens, layer)` stops after the layer. (MLX only computes what is asked
  for, so recording a layer inside a full pass and asking for nothing else costs the
  same; `run_to` makes that the default and does not build the rest of the pass.)
- `run_from(model, states, layer)` starts after the layer from stored results, so each
  variant in a sweep pays only for the layers above.
- `last_token_scores` and `run_from(..., last_only=True)` score only the position after
  the last token. The output layer is a tenth of a 4B Qwen model's weights and is
  otherwise run for every position.

**Qwen3-4B 4-bit** (36 layers; tok/s, MLX's kernels → lab's)

| Job | tok/s | Against the full pass with MLX's kernels |
|---|---|---|
| Full pass, every position scored | 677 → 901 | 1.33x |
| Full pass, last token scored | 745 → 992 | 1.46x |
| Collect layer 9's output | 2,982 → 3,964 | 5.9x |
| Collect layer 18's output | 1,490 → 1,982 | 2.9x |
| Collect layer 27's output | 995 → 1,319 | 1.9x |
| Sweep from layer 9 (last token scored) | 993 → 1,323 | 2.0x |
| Sweep from layer 18 (last token scored) | 1,486 → 1,974 | 2.9x |
| Sweep from layer 27 (last token scored) | 2,961 → 3,963 | 5.9x |

Largest difference between the resumed scores and the full pass: 0.031.

**Qwen3.5-4B 4-bit** (32 layers; tok/s, MLX's kernels → lab's)

| Job | tok/s | Against the full pass with MLX's kernels |
|---|---|---|
| Full pass, every position scored | 625 → 827 | 1.32x |
| Full pass, last token scored | 726 → 961 | 1.54x |
| Collect layer 8's output | 2,909 → 3,839 | 6.1x |
| Collect layer 16's output | 1,450 → 1,923 | 3.1x |
| Collect layer 24's output | 968 → 1,279 | 2.0x |
| Sweep from layer 8 (last token scored) | 967 → 1,281 | 2.1x |
| Sweep from layer 16 (last token scored) | 1,446 → 1,920 | 3.1x |
| Sweep from layer 24 (last token scored) | 2,881 → 3,844 | 6.2x |

Largest difference between the resumed scores and the full pass: 0.062.

- **Collecting an early layer costs a sixth of a full pass**, and a layer half way up
  costs a third. The lab's kernels account for 1.33x of that; the rest is not running
  the layers above.
- **A sweep from three quarters of the way up runs 5.3 times faster than full passes**
  that score the last token, and 5.9 to 6.2 times faster than plain full passes.
- **Scoring only the last token is worth 10 to 16% by itself** (745 against 677 tok/s
  on Qwen3-4B with MLX's kernels).
- The resumed scores differ from the full pass by rounding only (at most 0.06, on scores
  whose spread is above 1), because the output layer then runs on 8 rows instead of
  2,048 and a different kernel handles it.

Limits: random-token prompts (speed does not depend on the text); `run_from` relies on
the model starting from given layer inputs unchanged, which holds for the lab's GPT and
the Qwen families and should be checked once on any other.

Verdict: **keep**.

## 16. Running a model in float16 instead of bfloat16

Command: `uv run python -m bench.float16 --model LABEL=PATH`. Code: `lab/infer/float16.py`
(`float16.convert(model, tokenizer)`).

**The finding.** The Qwen models in MLX format store their 16-bit numbers as bfloat16
(8 significant bits, a huge range). Apple GPUs have no bfloat16 type, so MLX's GPU code
converts every such number in software each time it is read. float16 (11 significant
bits, largest value 65,504) is a native type. Section 11 had already shown MLX
multiplying float16 faster than bfloat16; here the whole model is switched over.

**The risk, and the check.** If any value inside the model went above 65,504 the output
would turn into nonsense without an error. `float16.convert` therefore first runs sample
prompts through the model as it is and notes the largest value written anywhere (every
layer's output, every weight table's input and output). It refuses if that comes within
4x of the limit. Then it converts, runs the same prompts again, and undoes the
conversion if any score is not a number or the two versions pick a different next token
too often.

**Result** (8 passages x 512 tokens of the lab's held-out text; tok/s, bfloat16 → float16):

| | Qwen3.5-0.8B 4-bit | Qwen3.5-4B 4-bit | Qwen3.5-9B 4-bit | Qwen3-4B 4-bit |
|---|---|---|---|---|
| Largest value inside the model | 81 | 200 | 404 | 7,008 |
| Room below float16's limit | 809x | 328x | 162x | 9x |
| Loss on the text, bfloat16 → float16 | 3.1286 → 3.1282 | 2.5344 → 2.5339 | 2.3886 → 2.3875 | 2.7336 → 2.7327 |
| Same top next token | 97.5% | 98.5% | 97.9% | 98.5% |
| Read prompts, MLX's kernels, tok/s | 4,598 → 6,475 (1.41x) | 724 → 1,023 (1.41x) | 384 → 543 (1.41x) | 739 → 1,036 (1.40x) |
| Read prompts, lab's, tok/s | 5,925 → 7,167 (1.21x) | 960 → 1,116 (1.16x) | 490 → 557 (1.14x) | 985 → 1,114 (1.13x) |
| Decode 1 prompt, tok/s | 205 → 212 (1.04x) | 82 → 83 (1.01x) | 55 → 56 (1.02x) | 91 → 93 (1.02x) |
| Decode 8, MLX's kernels, tok/s | 565 → 655 (1.16x) | 131 → 165 (1.26x) | 75 → 97 (1.28x) | 142 → 180 (1.27x) |
| Decode 8, lab kernel, tok/s | 803 → 849 (1.06x) | 271 → 289 (1.07x) | 170 → 183 (1.08x) | 297 → 320 (1.08x) |
| Decode 32, tok/s | 1,573 → 1,855 (1.18x) | 451 → 561 (1.25x) | 279 → 364 (1.31x) | 505 → 661 (1.31x) |

- **Prompts are read 1.40 to 1.41x faster with MLX's own kernels**, and 1.13 to 1.21x
  faster with the lab's (section 11). Both together are 1.45 to 1.56x over MLX in
  bfloat16.
- **32 prompts at once: 1.18 to 1.31x.** This is the setting that gives the most tokens
  per second, and nothing else in this document had moved it.
- **8 at once: 1.16 to 1.28x with MLX's kernels, 1.06 to 1.08x with the lab kernel**
  (which was already fast there). **One prompt: 1.01 to 1.04x**, as expected: that case
  waits on memory, not arithmetic.
- **A whole eval, end to end** (`--eval`: 64 MATH-500 questions, up to 256 new tokens
  each, 32 at a time, Qwen3.5-4B): 51.7 s as shipped, 40.1 s in float16 with the lab's
  paths on, 1.29x faster. float16 alone gives 1.27x; the lab's paths alone 1.05x.
- **A small eval through the server showed nothing.** 20 questions sent 32 at a time took
  46 to 50 seconds through mlx-lm's server, the lab's, and the lab's with `--float16`
  alike (a quick check, not saved). With so few questions the batch empties early and
  most of the time goes to the last few long answers, decoded one or two at a time.
- **The model is the same, to rounding.** The loss on the text moves by at most 0.0011,
  and is a little lower in float16 for all four models, since float16 keeps more
  digits. Both versions pick the same next token at 97.5% to 98.5% of positions, the
  level at which MLX's own kernels differ from each other (section 9).
- **How much room there is depends on the model.** The three Qwen3.5 models stay 160 to
  800 times below the limit. Qwen3-4B writes one very large value (about 7,000) at most
  of its layers and has only 9 times of room. The check's bar is 4 times; raise
  `min_headroom` to be stricter, and run it on prompts like your own.

Limits:

- **The check covers the prompts it is given.** A very different input could still
  overflow, and the only sign would be garbled output. Pass your own prompts with
  `convert(model, prompts=[...])`; without them it uses a built-in sample of chat,
  prose, code, other scripts and numbers.
- **Not for training.** The error signals that training passes backwards are far smaller
  than the values going forwards and would round to zero in float16. Fine-tune in
  bfloat16 (the lab's from-scratch trainer handles this differently, section 12).
- Measured on 4-bit models. The conversion is in memory only and takes a few seconds.

Verdict: **keep**, switched on by one line or by `--float16` on the lab's server.

## 17. Checked and rejected on 2026-11-12, and where the limits are now

These were quick measurements made while working, not saved to a results file, so treat
their figures as approximate.

- **Other block layouts for MLX's quantized kernel.** Rebuilt from MLX's own source with
  seven other layouts, including the one its dense kernel uses. No gain: the cost is the
  decoding of the weights inside every block. Section 11 decodes once instead.
- **Keeping the decoded weights in memory** between uses, instead of decoding them again
  at every use. About 10% faster on the trained layers of a fine-tune, for several GB
  held the whole time (a 4B model is 2.2 GB at 4-bit and 7.8 GB decoded). Not worth it.
- **A larger MLX buffer cache.** No effect on speed.
- **MLX's built-in kernel for the Qwen3.5 recurrent layers when generating** (mlx-lm uses
  one of its own). The same speed for one token; twice as fast when reading a prompt,
  but that layer is about 3% of prompt reading.
- **Compiling the per-token step of the recurrent layers** (`mx.compile`). 2.5 to 7%
  faster decoding. Too little for the risk of a stale compiled program.
- **More operations per GPU submission** (`MLX_MAX_OPS_PER_BUFFER=500`). 3 to 7% for one
  prompt. Not adopted.
- **float16 for the many-rows multiply only**, with each row rescaled so that nothing can
  overflow and the rest of the model left in bfloat16. 0 to 10% faster on real 4B layers
  and closer to the exact answer, but no longer bit for bit equal to MLX. Converting the
  whole model (section 16) gains more.
- **Two training runs at the same time** (to speed up sweeps). One run alone trains
  60,530 tok/s; two at once train 64,836 between them (1.07x), each at half speed and
  with twice the memory in use. Not worth it.

**Where the limits are now.** What any further speed-up would have to beat:

- **One prompt writing fresh text: unchanged.** A Qwen3.5-4B step takes about 12 ms.
  Reading its 2.2 GB of weights at this Mac's memory speed (375 GB/s, section 9)
  accounts for about 6 ms. The rest is the cost of issuing the step's 1,117 separate
  small GPU operations (249 quantized multiplies, 153 normalisations, 160 reshapes and
  so on); the program's main thread is busy for 4 of the 12 ms. For the 0.8B model it is
  3 of 5 ms. Cutting that means replacing groups of MLX operations with one hand-written
  kernel per layer type. Estimated gain 1.2x for the 4B model and up to 1.5x for the
  0.8B. Not attempted. Section 10 (guess-and-check) remains the only gain for one
  prompt, and only on text that repeats.
- **19 to 255 rows** (20 to 250 prompts decoded at once, or very short prompts read
  alone): no better kernel was found. Decoding first only wins from 256 rows (section
  11) and the lab's small-batch kernel only up to 18 (section 9). What does help there
  is float16 (section 16), which makes MLX's own kernel faster.
- **Big multiplies** (training, fine-tuning, long prompts): MLX's dense kernel already
  runs at 71-87% of the chip's arithmetic limit. The gains in sections 11 to 16 come
  from using the faster number format or from not doing work at all, not from a faster
  multiply.
- **Training from scratch.** One step of the 14M model takes 261 ms: 69 ms for the
  forward pass, 168 ms for the backward pass and 24 ms for the optimizer. Attention
  accounts for 45 ms of that and the loss for 5 ms. The step as a whole runs at 61% of
  the chip's arithmetic limit and its multiplies at 86-87%, so at most 1.4x is left and
  no single part stands out.

## 18. Not tested

- **Raising the GPU memory cap** (`iogpu.wired_limit_mb`). It needs `sudo`. Nothing
  measured here needed it: every model up to 27B at 4-bit loaded and ran under the
  default cap. The one job that failed (27B with 32 prompts at once) ran the whole Mac
  out of memory, which a higher cap would not fix.
- **Ollama against MLX** on the same model. Not run.
- **Learned quantization** (DWQ, dynamic, AWQ, GPTQ). Not run.
- **27B model, 32 prompts at once.** Tried; the watchdog stopped it when free memory
  fell to 4%. No number. Use 8 at once or fewer for this model (21.6 GB peak at 8).
