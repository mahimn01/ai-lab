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
| 5 | float16 training | loss became NaN every time | reject |
| 5 | Compiled training step | 9% faster | keep, the default |
| 6 | A different GPU code path (mxfp4 format, architecture override) | no gain, or slower | reject |
| 7 | Small vocabulary for from-scratch training | 2.6x faster, 40% less memory | keep, the default |
| 8 | Gradient checkpointing for Qwen3.5 fine-tuning | 18.3 GB down to 3.9 GB, same speed | keep, automatic |
| 9 | A new GPU kernel for 2-18 prompts at once | 1.3-1.7x faster at 4 prompts, 2.1-2.6x at 8; the 27B model's best speed doubles | keep, one line to switch on |
| 10 | Guess-and-check decoding for one prompt | 1.75-2.5x faster when the answer copies earlier text; no change on fresh text | keep, needs the kernel |
| 11 | Raising the GPU memory cap | not tested | not settled |

**Was a new, faster technique found?** Yes, one, with clear edges. Section 1 found that
on this chip MLX pays for 32 prompts whenever it decodes 8, and re-reads all the weights
for each of 2-6 prompts. Section 9 is a new GPU kernel that removes both: a step now
costs the first prompt plus about a fifth of that for each extra one, up to 8. Section
10 uses those cheap extra rows to make a single prompt faster when its answer repeats
earlier text. What it does not change: one prompt writing fresh text is limited by how
fast the Mac reads memory (about 375 GB/s, measured), and 32 prompts at once were
already efficient. It was measured only on this Mac and this MLX version.

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
MLX → lab). Each layer runs on its own here, which lets the GPU overlap one layer with
the next, so these are best-case figures; the decoding loop above is what counts.

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

## 11. Not tested

- **Raising the GPU memory cap** (`iogpu.wired_limit_mb`). It needs `sudo`. Nothing
  measured here needed it: every model up to 27B at 4-bit loaded and ran under the
  default cap. The one job that failed (27B with 32 prompts at once) ran the whole Mac
  out of memory, which a higher cap would not fix.
- **Ollama against MLX** on the same model. Not run.
- **Learned quantization** (DWQ, dynamic, AWQ, GPTQ). Not run.
- **27B model, 32 prompts at once.** Tried; the watchdog stopped it when free memory
  fell to 4%. No number. Use 8 at once or fewer for this model (21.6 GB peak at 8).
