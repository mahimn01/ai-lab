# Speed results: what was tested, and what to keep

Every idea below was run on this Mac (Apple M1 Max, 32 GB, MLX 0.32.4, mlx-lm 0.32.2) on
2026-11-11 and 2026-11-12. Each section gives the command, the numbers (from `bench/results/*.json`)
and a verdict: **keep**, **reject**, or **not settled**. Section 19 was added on the evening
of 2026-11-12 and finished the next morning.

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
| 16 | float16 instead of bfloat16 for inference | prompts read 1.4x faster with MLX's kernels and 1.4-1.6x with the lab's on top; 32 prompts at once 1.16-1.31x; a 64-question eval 1.29x faster end to end; one request with a 2,048-token prompt 1.29 to 1.32x faster; writing speed for one prompt unchanged; the model's values must stay under 65,504 | keep, one line, checks first |
| 17 | Eight smaller ideas (other kernel layouts, resident decoded weights, compiled steps, two trainings at once, ...) | 0-10% each | reject |
| 18 | Raising the GPU memory cap | not tested | not settled |
| 19 | A row engine for many long generations (history in pages, no padding, sampling on the GPU, 48 rows at once instead of 16) | a 100-problem condition at a 32,768-token cap in 59 and 56 minutes (two draws) instead of 104: 1.75x and 1.85x; 12 minutes instead of 18 at an 8,192 cap | keep |
| 19 | One long run cut short to get every smaller cap | free, and exact for the rows of that run (a separate 8,192-cap run with the same seed wrote the same first 8,192 tokens for 98 of 100 rows) | keep |
| 19 | Forced exits as branches of a run instead of runs of their own | four more conditions in 31 minutes when added to a finished run; as separate runs an estimated 2.1 to 4.4 hours | keep |
| 19 | Branches that wait for a slot, so a run's rows and its forced exits are one run | the edited model's five conditions in 68 minutes instead of 91, the base model's in 117 instead of 133 (17 and 13 minutes of that by the planner's one clock: the earlier runs were made on a busier evening); 599 of 645 branches start from their row's own history instead of being read back in | keep |
| 19 | A planner that replays a run's row lengths through the engine's rules, without a model | the same steps, rows set aside, branch counts and tokens read back as all nine timing runs; minutes within 1.5% of the three long runs made after midnight and of the two 8,192-cap runs, while the first evening's long runs took 3 to 7% longer than it says | keep, for sizing runs |
| 19 | Deciding a comparison early with a test that may be looked at after every problem | "base loops more than edited" decided at problem 18 and 26 of 100 on saved runs | keep |
| 19 | Stopping rows once they loop | on the edited model it saves up to 40% of tokens but loses correct answers unless the rule is very strict (6.5 to 11% saved, none lost); on the base model it saves 49 to 74% and loses none | off by default, with an audit |
| 19 | Guessing ahead from repeats, a shorter cap as a stand-in, skipping near-zero attention values, float16 for this model | 1.74 tokens per pass; misranks edits; no gain; 4 to 6% | reject |

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
  reading, 1.16 to 1.31x at 32 prompts at once, and 24% in from-scratch training.

- **A row engine for long reasoning runs** (section 19). mlx-lm pads every row of a
  batch to the longest one and sorts the whole vocabulary at every step. With the history
  in pages and the top 20 found without the sort, a 100-problem run at a 32,768-token
  cap takes 56 to 59 minutes instead of 104. It fell short of the 3x it was designed
  for, and section 19 says why.

What none of this changes: the speed at which one prompt's answer is written when the
text is new. That is limited by how fast the Mac reads memory (about 375 GB/s, measured)
and by the cost of issuing a thousand small GPU operations per token (section 17). A
whole request still gets faster when it has a lot to read: 1.29 to 1.32x with a
2,048-token prompt and 128 new tokens (section 16). Everything here was measured only on
this Mac and with these versions of MLX and mlx-lm.

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
there. The decoded table is temporary and is freed as soon as the layer has run (a table
whose decoded copy would be bigger than 2 GB is left to MLX's kernel). It has a
gradient, so fine-tuning uses it too (section 13).

**Result** (tokens per second for reading the prompts and scoring every position, which
is what scoring a text needs; MLX's kernels → lab):

| Prompts x tokens each | Qwen3.5-0.8B 4-bit | Qwen3.5-4B 4-bit | Qwen3-4B 4-bit | Qwen3.5-9B 4-bit |
|---|---|---|---|---|
| 1 x 512 | 3,085 → 3,467 (1.12x) | 608 → 742 (1.22x) | 649 → 814 (1.25x) | 328 → 400 (1.22x) |
| 1 x 2,048 | 3,092 → 3,744 (1.21x) | 609 → 774 (1.27x) | 633 → 826 (1.30x) | 335 → 423 (1.26x) |
| 8 x 512 | 3,238 → 4,196 (1.30x) | 622 → 827 (1.33x) | 671 → 895 (1.33x) | 337 → 433 (1.28x) |
| 32 x 256 | 3,157 → 4,057 (1.29x) | 623 → 826 (1.33x) | 675 → 899 (1.33x) | 335 → 428 (1.28x) |

- **1.2 to 1.33x faster** from 4B up; the 0.8B model gains a little less. Reading a
  prompt before generating skips the output layer; section 16 times that case and the
  gain is the same (1.28 to 1.33x for 8 x 512 tokens).
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
| 32-bit throughout, seed 0 | 1.6799 | 49,009 | 449 | 7.7 GB |
| 32-bit throughout, seed 1 | 1.6946 | 48,719 | 447 | 7.7 GB |
| **Mixed: 32-bit weights, float16 arithmetic**, seed 0 | 1.6349 | 60,503 | 554 | 4.5 GB |
| **Mixed**, seed 1 | 1.6353 | 60,992 | 559 | 4.5 GB |
| bfloat16 throughout (section 5's option) | 1.6760 | 54,010 | 495 | 4.5 GB |
| Mixed with bfloat16 arithmetic | 1.6543 | 54,298 | 498 | 4.6 GB |

- **24% more training per second and 41% less memory** than 32-bit.
- **A better model in the same time**: 1.635 on average against 1.687, a gap of 0.052,
  8 times the noise of section 5 (0.007). Both mixed runs beat both 32-bit runs.
- It also beats bfloat16 throughout, on speed (float16 multiplies are faster) and on
  score (0.041, one run).
- **Nothing overflowed.** A gradient value too large for 16 bits is left out of that
  step, and the trainer now counts the steps where that happens: 0, 0, 0 in these three
  mixed runs. It also checks the loss at every step, and scores the mixed formats in
  32-bit as the 32-bit runs are scored.
  Beyond the batch size measured here the scale stops growing with the batch, because a
  weight's gradient is a sum over the batch.

Limits: one model size (14 million parameters), 150 seconds, two seeds. A 45-second check
of the 91M size with the default format ran at 10,300 tok/s in 10.9 GB with no overflowed
step (a quick check, not saved; in 32-bit that size needs 20 GB and collapses to 125
tok/s, `BASELINE.md` section 3).

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

**Is it the same training?** With `--file-order --val-batches -1` both trainers see the
same batches in the same order from the same starting point, and both score the whole
validation set, so the losses can be compared number for number:

| Model | Trainer | Validation loss before | After 60 steps | Training loss, first and last 10 steps | tok/s |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | mlx-lm as it comes | 3.029 | 1.319 | 1.998, 1.360 | 243 |
| Qwen3.5-0.8B | lab GPU paths | 3.030 | 1.320 | 1.998, 1.363 | 1,061 |
| Qwen3.5-4B | mlx-lm as it comes | 2.587 | 1.149 | 1.815, 1.153 | 94 |
| Qwen3.5-4B | lab GPU paths | 2.587 | 1.150 | 1.815, 1.154 | 305 |

The two end within 0.001 of each other. The recurrent update and the many-rows path
change how the arithmetic is done, not what is computed.

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
  vocabulary (151,936 words for Qwen3, 248,320 for Qwen3.5) is a fixed cost of every
  step, which is why training a quarter of the layers is not four times faster.
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
  the last token. The output layer is a tenth (Qwen3-4B) to a sixth (Qwen3.5-4B) of the
  model's weights and is otherwise run for every position.

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

Commands: `uv run python -m bench.float16 --model LABEL=PATH [--eval] [--small]` and
`uv run python -m bench.request --model LABEL=PATH`. Code: `lab/infer/float16.py`
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

**Result** (8 passages x 512 tokens of the lab's held-out text, 2 passages for the 27B
model; tok/s, bfloat16 → float16):

| | Qwen3.5-0.8B | Qwen3.5-4B | Qwen3.5-9B | Qwen3-4B | Qwen3.8-27B |
|---|---|---|---|---|---|
| Largest value inside the model | 81 | 200 | 404 | 7,008 | 1,056 |
| Room below float16's limit | 809x | 328x | 162x | 9x | 62x |
| Loss on the text, bfloat16 → float16 | 3.1286 → 3.1282 | 2.5344 → 2.5339 | 2.3886 → 2.3875 | 2.7336 → 2.7327 | 2.5393 → 2.5411 |
| Same top next token | 97.5% | 98.5% | 97.9% | 98.5% | 98.4% |
| Read prompts, MLX's kernels, tok/s | 4,594 → 6,467 (1.41x) | 724 → 1,022 (1.41x) | 384 → 543 (1.41x) | 739 → 1,037 (1.40x) | 109 → 154 (1.42x) |
| Read prompts, lab's, tok/s | 5,919 → 7,161 (1.21x) | 959 → 1,117 (1.16x) | 491 → 556 (1.13x) | 984 → 1,115 (1.13x) | 127 → 154 (1.21x) |
| Decode 1 prompt, tok/s | 203 → 215 (1.06x) | 81.4 → 84.3 (1.04x) | 54.7 → 55.0 (1.01x) | 91.9 → 92.9 (1.01x) | 18.7 → 18.9 (1.01x) |
| Decode 8, MLX's kernels, tok/s | 567 → 644 (1.14x) | 131 → 167 (1.28x) | 75.1 → 96.4 (1.28x) | 142 → 180 (1.27x) | 22.2 → 29.6 (1.33x) |
| Decode 8, lab kernel, tok/s | 783 → 843 (1.08x) | 271 → 289 (1.07x) | 177 → 184 (1.04x) | 296 → 318 (1.07x) | 58.2 → 59.4 (1.02x) |
| Decode 32, tok/s | 1,576 → 1,825 (1.16x) | 450 → 560 (1.25x) | 278 → 364 (1.31x) | 505 → 662 (1.31x) | does not fit |

- **Prompts are read 1.40 to 1.42x faster with MLX's own kernels.** With the lab's paths
  switched on as well, reading is 1.42 to 1.56x faster than MLX in bfloat16.
- **32 prompts at once: 1.16 to 1.31x.** This is the setting that gives the most tokens
  per second, and nothing else in this document had moved it.
- **8 at once: 1.14 to 1.33x with MLX's kernels, 1.02 to 1.08x with the lab kernel**
  (which was already fast there). **One prompt: 1.01 to 1.06x**, as expected: that case
  waits on memory, not arithmetic.
- **A whole eval, end to end** (`--eval`: 64 MATH-500 questions, up to 256 new tokens
  each, 32 at a time, Qwen3.5-4B): 51.8 s as shipped, 40.1 s in float16 with the lab's
  paths on, 1.29x faster. float16 alone gives 1.27x; the lab's paths alone 1.05x.
- **A small eval through the server showed nothing.** 20 questions sent 32 at a time took
  46 to 50 seconds through mlx-lm's server, the lab's, and the lab's with `--float16`
  alike (a quick check, not saved). With so few questions the batch empties early and
  most of the time goes to the last few long answers, decoded one or two at a time.
- **The model is the same, to rounding.** The loss on the text moves by at most 0.0018
  (lower in float16 for 4 of the 5 models). Both versions pick the same next token at
  97.5% to 98.5% of positions, the level at which MLX's own kernels differ from each
  other (section 9).
- **How much room there is depends on the model.** The three Qwen3.5 models stay 162 to
  809 times below the limit and the 27B model 62 times. Qwen3-4B writes one very
  large value (about 7,000) at most of its layers and has only 9 times of room. The
  check's bar is 4 times; raise `min_headroom` to be stricter, and run it on prompts
  like your own.
- **In float16 the lab's many-rows path (section 11) matters less**, because MLX's own
  kernel is then nearly as fast: 3 to 13% slower on the 9B model. On the 27B model,
  which fills most of this Mac's memory, the many-rows path was the slower of the two
  in a first run (140 against 154 tok/s; the cause was not established). `qmm.install`
  therefore leaves that path off for a float16 model whose weights take more than a
  third of the Mac's memory, and the 27B column above was measured that way.

**One request, end to end** (`uv run python -m bench.request --model LABEL=PATH`: one
prompt of the given length, 128 new tokens, greedy decoding through mlx-lm's own
generator; seconds from sending the prompt to the last token):

| Model, prompt | As shipped | Lab paths | float16 | float16 + lab paths |
|---|---|---|---|---|
| Qwen3.5-4B, 512 tokens | 2.06 s | 1.95 s (1.06x) | 1.83 s (1.13x) | 1.80 s (1.14x) |
| Qwen3.5-4B, 2,048 tokens | 4.23 s | 3.68 s (1.15x) | 3.35 s (1.26x) | 3.21 s (1.32x) |
| Qwen3.5-4B, 8,192 tokens | 13.29 s | 10.97 s (1.21x) | 9.80 s (1.36x) | 9.22 s (1.44x) |
| Qwen3.5-9B, 512 tokens | 3.49 s | 3.24 s (1.08x) | 3.14 s (1.11x) | 3.02 s (1.16x) |
| Qwen3.5-9B, 2,048 tokens | 7.43 s | 6.41 s (1.16x) | 5.85 s (1.27x) | 5.77 s (1.29x) |
| Qwen3.5-9B, 8,192 tokens | 23.70 s | 19.55 s (1.21x) | 17.46 s (1.36x) | 17.04 s (1.39x) |
| Qwen3-4B, 512 tokens | 1.97 s | 1.84 s (1.07x) | 1.74 s (1.13x) | 1.72 s (1.15x) |
| Qwen3-4B, 2,048 tokens | 4.30 s | 3.65 s (1.18x) | 3.44 s (1.25x) | 3.32 s (1.29x) |
| Qwen3-4B, 8,192 tokens | 16.25 s | 13.62 s (1.19x) | 12.25 s (1.33x) | 11.79 s (1.38x) |

- **A single request is 1.14 to 1.16x faster with a 512-token prompt, 1.29 to 1.32x with
  2,048 tokens and 1.38 to 1.44x with 8,192.** Writing the answer is no faster (section
  17); all of the gain is in reading the prompt, so it grows with the prompt and shrinks
  with the length of the answer.

Limits:

- **The check covers the prompts it is given.** A very different input could still
  overflow, and the only sign would be garbled output. Pass your own prompts with
  `convert(model, prompts=[...])`; without them it uses a built-in sample of chat,
  prose, code, other scripts and numbers.
- **Not for training.** The error signals that training passes backwards are far smaller
  than the values going forwards and would round to zero in float16. Fine-tune in
  bfloat16 (the lab's from-scratch trainer handles this differently, section 12).
- Measured on 4-bit models. The conversion is in memory only and takes a few seconds.
  While it checks, it holds a second copy of the parameters it converts (small for a
  quantized model; as much again as the weights for one that is not quantized).

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

- **One prompt writing fresh text: the writing is unchanged** (a request with a long
  prompt is still faster overall, because the reading is: section 16). A Qwen3.5-4B step
  takes about 12 ms. Reading its 2.2 GB of weights at this Mac's memory speed (375 GB/s,
  section 9) accounts for about 6 ms. The rest is the cost of issuing the step's 1,117 separate
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

## 19. Long-reasoning experiments: a row engine

Commands: `uv run python -m bench.phase1 --arm edited --tag edited-32k` (one condition;
the two draws below were made with `--pool-gb 12`, and the default is now 10),
`uv run python -m bench.engine_checks`, `uv run python -m bench.long_step --rows 16`,
`uv run python -m bench.transcripts`, `uv run python -m bench.float16_long`,
`uv run python -m bench.plan_sim`. Code:
`lab/gen/`, `lab/stats/sequential.py`. Design, written before the code and corrected after
an independent review: `docs/design-row-engine.md`.

Words used in this section:

- **Row**: one generation, from prompt to stop. A **condition** is a set of rows with
  the same settings (model, cap, sampling).
- **Cap**: the most tokens a row may write.
- **Step**: one pass of the model that writes one token for every row in the batch.
- **History**: what the attention layers store for every earlier token of a row, about
  12 KB per token for this model. A 32,768-token row holds 0.4 GB.

**The workload.** The `neural-harness` study asks whether a small model (Qwen3.5-0.8B,
dense weights) can be repaired from looping by a weight edit. One condition is 100
MATH-500 problems with thinking on and up to 32,768 new tokens each, about 2.1 million
tokens. The study's own runner (mlx-lm's batch generator, 16 rows at once) needed 104
minutes for it. Its recommended next phase has about ten such conditions on 500 problems
with 4 samples each.

### Where a step goes

`bench.long_step`: mlx-lm's own model pass for one token per row, no sampling, with every
row's history at the given length.

| Rows | History 64 tokens | 4,096 | 16,384 | 32,768 |
|---|---|---|---|---|
| 16 | 21.9 ms | 25.4 ms | 35.3 ms | 48.5 ms |
| 32 | 25.6 ms | 32.6 ms | 52.3 ms | not run (over the test's 7 GB) |
| 48 | 29.7 ms | 41.2 ms | not run | not run |
| 64 | 31.0 ms | 44.9 ms | not run | not run |
| 96 | 48.5 ms | 71.8 ms | not run | not run |

- **Most of a step is a fixed cost.** 16 rows at a short history take 22 ms, of which
  the arithmetic needs about 3. The rest is roughly a thousand small GPU operations at
  about 20 millionths of a second each. Compiling three of the recurrent layers
  (`mx.compile`) made those layers 2 to 6% faster, and they are an eighth of a step.
  No single operation stands out.
- **The rest is reading the history**, and MLX's attention kernel already reads it at
  170 to 270 GB/s (the chip gives about 375).
- **mlx-lm pads every row's history to the longest row in the batch.** With rows
  starting and ending at different times the longest is nearly always close to the cap,
  so every row pays for a full history at every step. The study's 337 tokens a second
  is the table's 16-row, 32,768-token cell (16 rows in 48.5 ms is 330 a second).
- **Picking the next token the usual way costs a tenth to a third of a step.** Finding the top 20
  of 248,320 scores with a full sort takes 2.7, 5.2 and 9.9 ms for 16, 32 and 64 rows.
  Sorting only the 20 blocks of 256 with the largest maximum gives the same values in
  0.6, 0.9 and 1.2 ms.
- **float16 gains 4 to 6% here** (21.0 against 21.9 ms at a short history, 33.7
  against 35.3 at 16,384), far less than on the 4B models of section 16: this model's
  steps are mostly fixed cost. It is safe on the study's models: over 4 saved
  transcripts of 4,241 tokens the largest value inside either model was 82, 794 times
  below float16's limit (`bench.float16_long`). The runs below stay in bfloat16.

### Three shortcuts checked on the study's saved transcripts, and turned down

`bench.transcripts` needs no GPU: 3,732 transcripts the study had saved (900 as text,
2,832 with token ids).

1. **Guessing ahead from the text's own repeats** (section 10's idea, with the loop as
   its own guess). 52% of the 32k run's tokens can be guessed, but 5.6 at a time, and
   77% of guesses break early: 1.74 tokens per model pass. The loops in this study are
   soft: the wording drifts. Not enough to pay for a second lane.
2. **Stopping a row once it loops** (the study's test: a 512-token window whose text
   compresses below a fifth). On the saved 32k run:

   | Stop after this many looping tokens | 512 | 896 | 1,408 | 2,432 | 4,480 | 8,576 |
   |---|---|---|---|---|---|---|
   | Tokens saved | 27.6% | 19.1% | 15.9% | 13.1% | 9.4% | 6.5% |
   | Correct answers lost (of 54) | 4 | 2 | 2 | 1 | 1 | 0 |

   The edited model escapes loops thousands of tokens long and then answers. So the
   rule exists in the engine but is off by default, and a run that uses it also runs a
   share of its rows to the full budget to measure what it changed.
3. **A shorter cap as a stand-in.** Loops form late: at a 2,048 cap the base model shows
   2.5 looping rows of 64, at 8,192 it shows 40. The study's 38 saved edit variants rank
   differently at 2,048 (rank agreement 0.39) and 4,096 (0.71) than at 8,192.

What the saved results do support is **deciding early**. A test that may be looked at
after every problem (`lab/stats/sequential.py`) would have decided "the base model loops
more than the edited one" at problem 18 of 100 on one saved pair of runs and at problem
26 on another. The accuracy comparisons would not have been decided in 100 problems, in
line with their ordinary p-values (0.07, 0.125, 0.55).

### The engine

`lab/gen` treats the row as the unit of work. Each row has its own cap, sampling
settings, random stream, optional weight edit and rules.

- **History in pages of 256 tokens**, with a GPU kernel that walks each row's own pages.
  Nothing is padded, memory follows what the rows really hold, and a finished row's
  pages go back to the pool. On 48 rows of equal length the kernel reads at 254 GB/s
  against 273 for MLX's own. On 48 rows of mixed length it needs 5.1 ms for one layer
  where MLX needs 7.7 ms for the padded block (`bench.long_step`).
- **Sampling on the GPU** with the study's rule and the block method above.
- **A random stream per row.** mlx-lm draws once for the whole batch, so a row's text
  depends on which rows share the batch. Here it does not, as long as the batch keeps
  its width ("Branches that wait for a slot" below has the measurement).
- **Branches.** A row can branch at a token count. The branch shares its trunk's
  pages, at once if a slot is free and otherwise after waiting for one. Failing that it
  is read in later from the trunk's saved tokens.
- **Several weight edits in one batch**, each row carrying its own (kept unmerged).
- **A ledger**: every finished row is on disk at once, with its token ids, and a
  restarted run skips what is there.
- The batch narrows when rows run out, and a row is set aside and replayed later if the
  pool runs short.

### Is it the same computation?

`bench.engine_checks`, on the study's real model:

| Check | Result |
|---|---|
| Engine's scores against mlx-lm reading the same text in one pass, 2,400 positions | largest difference 0.664, same top token at 99.33% |
| The yardstick: mlx-lm token by token against mlx-lm in one pass | largest difference 0.679, same top token at 99.42% |
| Two rows forced through 32,768 tokens, each compared at 7 positions up to the last | largest difference 0.250, same top token at 14 of 14 |
| With no randomness, tokens identical to mlx-lm's own generator | 259, 400, 331 and 333 of 400; the first difference is always between candidates 0.125 or less apart |
| The same sampled row alone in 8 slots, among 7 others, and among them in the last slot instead of the first | identical for 400 of 400 tokens each time |
| One text in a batch of 48 that keeps its width, and in one that narrows to 24 after 200 steps | equal to the last digit for the 200 steps; after them up to 0.438 apart, same top token at 99.00%. The same numbers whether the narrowing moves the row to another slot or leaves it in its own |
| A sampled branch started at once, after waiting for a slot, and read in from saved tokens | waiting: identical for 400 of 400 tokens; read in: identical for the first 23 |
| With no randomness, a branch against the next 200 tokens its own trunk goes on to write | 200 of 200 started at once, 200 of 200 after a wait that ended while the trunk was still running |
| With no randomness, the paged kernel against MLX's own attention row by row | 259, 400, 43 and 281 of 400; the first difference is always between candidates 0.125 apart |
| 1,500 sampled tokens worked out again outside the engine, the slow way | 1,500 identical |
| 8,192 tokens that the study's runner drew: outside what the engine would keep? | 2 (1 with mlx-lm's own scores and the rule written the slow way) |
| Kept tokens' probabilities against the study's own sampler code, 60 real score vectors | equal to at least 6 digits. The two lists of 20 tokens are identical for 54 vectors; in the other 6 they differ only in the order of equal-scored tokens that the top-p cut had already dropped (a quick check, not saved) |
| A row carrying the edit unmerged against the study's adapter model | largest difference 0.531 where the edit moves scores by up to 5.1; same top token at 98.67% |
| Step time with a 1 GB and an 8 GB pool | 22.2 and 22.2 ms: the pool is written in place, never copied. (Timed at the end of an earlier version of the script, in a process then holding 5 GB of kept scores, the 8 GB pool was 2.4 ms slower. Not saved) |

This model computes in bfloat16, where a score near 20 can only move in steps of 0.125.
Two correct ways of computing a score therefore differ, and the first two lines show the
engine differs from mlx-lm by as much as mlx-lm differs from itself.

### Result: the Phase 0 condition

Edited model, thinking on, 32,768 cap, the study's 100 problems and sampling.

| | The study's runner | Row engine, first draw | Row engine, second draw |
|---|---|---|---|
| Wall-clock | 103.5 min (104.1 on record from 2026-11-03) | **59.2 min** | **55.8 min** |
| Tokens per second, whole run | 339 | 620 | 654 |
| Tokens per second while every slot was busy | 359 (the first 89 minutes) | 762 (the first 34 minutes) | 809 (the first 32 minutes) |
| Rows at once | 16 | 48 | 48 |
| Steps | about 150,000 (worked out from its rows' lengths) | 65,536 | 65,536 |
| Peak memory | not recorded | 15.2 GB | 15.2 GB |

- **1.75x and 1.85x faster for the whole condition**, with 48 rows at once where the
  study's runner holds 16. The same-day rerun of the study's runner wrote the same text
  as on 2026-11-03 for all 100 rows in 0.6% less time, so the "before" is solid. While
  every slot was busy the engine wrote 762 and 809 tokens a second against 359.
- **The first draw looks disturbed.** The two draws use
  different random seeds and wrote almost the same number of tokens (2.20 and 2.19
  million) in the same number of steps, so the 3.4 minutes between them do not come
  from the work. From its 14th minute the first draw's log is uneven: between 30,500
  and 43,500 tokens a minute, up and down, where the second draw falls smoothly from
  46,500 to 38,700. What was running then was not established (other programs were in
  use that evening; a look shortly before the second draw found the Mac idle). Replayed
  by the planner below with the second draw's costs, the first draw's rows take 55.1
  minutes.
- **This is less than the 3x the design predicted** (35 to 45 minutes). Three reasons,
  each measured:
  - A whole engine step at a short history takes 23.6 ms for 16 rows, 33.0 for 48 and
    53.7 for 96: about 18 ms a step plus 0.3 to 0.4 ms per row. The per-row part (the
    output layer over the vocabulary, each recurrent layer's state, the arithmetic) is
    real work, so more rows do not come free.
  - The history is read at about 250 GB/s by MLX's kernel and by this one. Dropping the
    padding cut what is read, and nothing tried made the reading itself faster.
  - 100 rows are too few to keep 48 slots busy: 10 rows only started when the first
    long rows ended, at step 32,768, and 2 of them ran to the cap. So the run took two
    rounds of 32,768 steps and its last 25 minutes ran with slots emptying.
- **Where a token's time goes with the batch full**: between minutes 9 and 32 of the
  second draw all 48 slots were busy, a row held 15,700 tokens of history on average
  and a step took 67 ms. 33 ms of that do not depend on the history: the model's
  arithmetic for 48 rows and the fixed cost of a step. The other 34 ms are spent
  reading 48 histories, 45 billionths of a second per token of history. So reading is
  half of a step, and it already runs at the memory speed both kernels reach: on this
  Mac a faithful long row has little left to give. The larger savings below come from
  writing fewer tokens. (Over the uneven minutes of the first draw the same sum gives
  54 billionths. The history cost was first taken from that draw, as 51.)
- **Memory limits the batch.** 12 GB of history pool plus the model and the per-row
  state was 15.2 GB. A later run with the same pool, which also read long texts back
  in, stalled when the Mac ran out of memory: swap nearly full, 5 seconds a step until
  it was stopped. What else was using memory at that moment was not established. The
  runs with forced exits after that use a 10 GB pool and did not stall.

**Is it the same condition?** The two runners draw different random numbers, so their
texts differ and only rates can be compared.

| | The study's runner | Row engine, first draw | Row engine, second draw |
|---|---|---|---|
| Correct | 54 | 47 | 55 |
| Rows at the cap | 32 | 35 | 31 |
| Rows ending in a loop | 25 | 35 | 30 |
| Median tokens | 20,958 | 22,371 | 22,961 |

The first draw is 7 points below the study's run (95% interval from 14 lower to equal)
and the second 1 point above it (from 8 lower to 10 higher). The published scores of
this model (4 samples per problem) put the spread between two 100-problem draws at
about 4.3 points (worked out from the study's `models/edited/eval` files; a quick
check, not saved), so the first draw was 1.6 spreads low: unusual, not impossible. Four
things say it was the draw and not the engine: the second draw; the direct checks
above; the first 4,096 tokens of all rows are as repetitive in both runners (18.3% of
tokens guessable in each; a quick check, not saved); and another seed at 8,192
tokens, below. The second draw was made because the first could not settle it.

**Three draws from each runner, looked at after 8,192 tokens.** Two more seeds of the
engine at an 8,192 cap, and the 32,768 runs of both cut at 8,192:

| | Correct | Still running at 8,192 | Ending in a loop |
|---|---|---|---|
| The study's runner, its 8,192 run | 13 | 87 | 18 |
| The study's runner, its unmerged-edit run | 15 | 85 | 21 |
| The study's runner, its 32,768 run cut | 18 | 82 | 13 |
| Row engine, its 32,768 run cut | 12 | 88 | 24 |
| Row engine, seed 1 | 15 | 85 | 15 |
| Row engine, seed 2 | 15 | 84 | 18 |

The engine's first draw sits at the unlucky end of every column and its other two in the
middle. On average the two runners are 15.3 and 14.0 correct, 84.7 and 85.7 still
running, 17.3 and 19.0 looping. (The loop and still-running counts of the two cut rows
are a quick check, not saved; the rest is in `bench/results/phase1.json` and the
study's `runs/summary.md`.) Seed 1 is also the seed of the second 32,768 draw, so that
row is the second draw's opening and not a further sample: 98 of the 100 rows are the
same tokens.

**At an 8,192 cap** the two extra seeds took 12.1 minutes each (1,058 and 1,046 tokens a
second), against 18.1 minutes on record for the study's runner on the same condition
(716 tokens a second, 32 rows at once): 1.5x. The base model with thinking off at the
same cap took 5.8 minutes against 8.3 on record. That draw also wrote 14% fewer tokens
(20 rows at the cap against 28), so per token it is 1.2x: 815 tokens a second against
661. 43 of 100 were correct in both runners.

### What one long run gives besides its own number

- **Accuracy at every smaller cap**, by cutting the same rows short (the grader reads
  the last boxed answer in whatever text exists). First draw: 7% at 4,096 tokens, 12%
  at 8,192, 28% at 16,384, 42% at 24,576, 47% at 32,768. Second draw: 9%, 14%, 33%,
  46%, 55%. A separate run at the smaller cap is not always the same text: the
  8,192-cap run with the second draw's seed wrote the same first 8,192 tokens for 98 of
  100 rows, and has 15 correct where the cut has 14. Its batch narrows at other steps
  ("Branches that wait for a slot" below).
- **The loop rule, measured again on rows it was not tuned on**:

  | Stop after this many looping tokens | 512 | 896 | 1,408 | 2,432 | 4,480 | 8,576 |
  |---|---|---|---|---|---|---|
  | First draw: tokens saved | 39.6% | 32.4% | 28.6% | 20.7% | 15.0% | 11.1% |
  | First draw: correct answers lost (of 47) | 6 | 2 | 2 | 1 | 1 | 0 |
  | Second draw: tokens saved | 32.9% | 25.1% | 20.7% | 15.6% | 13.0% | 7.9% |
  | Second draw: correct answers lost (of 55) | 13 | 5 | 3 | 3 | 1 | 0 |

  The same picture as on the saved run: only the strictest rule loses nothing.

### Forced exits as branches

The study's next phase asks what happens when a row that is still thinking at 4,096,
8,192, 16,384 or 24,576 tokens is made to stop and answer. In the engine such a row
branches: the branch takes the study's 25-token exit phrase and answers, and the trunk
carries on. Here the branches were added to the finished run from its saved tokens
(`--exits` on a ledger that already holds the rows), each with an answer budget of 4,000
tokens.

| Forced exit at | Rows forced | Correct | Tokens per problem | Correct with an answer budget of 500, 1,000, 2,000 | Branches that used all 4,000 |
|---|---|---|---|---|---|
| 4,096 | 93 | 39% | 6,535 | 17%, 37%, 39% | 58 |
| 8,192 | 85 | 37% | 10,360 | 20%, 35%, 37% | 58 |
| 16,384 | 56 | 45% | 16,769 | 33%, 43%, 44% | 40 |
| 24,576 | 36 | 52% | 20,480 | 46%, 51%, 52% | 26 |
| never (the run itself) | | 47% | 22,037 | | |

- **Four more conditions in 31.3 minutes**: 270 branches, 2.9 million tokens of the
  rows' own text read back in and 0.8 million new tokens written. Run from scratch the
  four would need about 5.4 million tokens of writing: 2.1 to 4.4 hours at the two
  rates on record for the study's runner (716 tokens a second at an 8,192 cap, 339 at
  32,768). That is an estimate, not a run.
- **The answer budget matters up to about 2,000 tokens.** The study's own forced exit
  let the answer run on to the overall cap. Here most forced rows were still writing
  when their 4,000 tokens ran out, but going from 2,000 to 4,000 added at most one
  correct answer per condition.
- **Read these accuracies as a by-product of a timing run**: one draw of 100 problems,
  where a condition's own spread is about 3 points. They are not the study.

### The base model, with its forced exits in the same run

The model without the edit, thinking on, 32,768 cap, branching at the same four exit
points as it goes (`--arm base --exits 4096,8192,16384,24576`), with 32 rows at once
and a 10 GB pool. The study has no run of this condition to compare with.

| | |
|---|---|
| Wall-clock for the five conditions | 132.9 min |
| New tokens | 3.97 million: 2.99 million in the 100 rows, 0.99 million in 364 branches |
| Tokens read back in | 5.09 million |
| Tokens per second, whole run | 498 |
| Peak memory | 13.5 GB |

- **The 100 rows alone would take the study's runner about 147 minutes** (2.99 million
  tokens at its 339 a second), and the four forced exits as separate runs another 2.3
  to 4.9 hours (5.96 million tokens at 716 or 339 a second). Both are estimates.
- **Nearly every row runs to the cap**: 89 of 100, against 35 for the edited model. A
  full row holds 0.4 GB, so the pool ran short 13 times and a row was set aside and
  replayed.
- **Only 1 of the 364 branches shared its trunk's pages.** In this run a branch shared
  only if a slot was free at the moment it was asked for, and with every slot busy that
  almost never happened. The other 363 were read in later from saved tokens. Reading
  one row back runs at 4,870 tokens a second for 4,096 tokens and 3,548 for 24,576 (a
  quick check, not saved), which comes to about 22 of the 133 minutes here and 12 of
  the 31 minutes for the edited model's exits above. The next subsection changes the
  rule.

| | Base | Edited (first draw, from above) |
|---|---|---|
| Correct | 13 | 47 |
| Rows at the cap | 89 | 35 |
| Rows ending in a loop | 88 | 35 |
| Median tokens | 32,768 | 22,371 |

Cut at a smaller cap the base model has 7% at 4,096 tokens, 10% at 8,192 and 13% from
16,384 on. Its forced exits, each branch with an answer budget of 4,000 tokens:

| Forced exit at | Rows forced | Correct | Tokens per problem | Correct with an answer budget of 500, 1,000, 2,000 | Branches that used all 4,000 |
|---|---|---|---|---|---|
| 4,096 | 94 | 40% | 6,619 | 20%, 38%, 40% | 58 |
| 8,192 | 92 | 42% | 10,335 | 22%, 41%, 42% | 56 |
| 16,384 | 89 | 43% | 17,670 | 23%, 42%, 43% | 54 |
| 24,576 | 89 | 44% | 24,978 | 25%, 42%, 44% | 55 |
| never (the run itself) | | 13% | 29,858 | | |

The loop rule on these rows:

| Stop after this many looping tokens | 512 | 896 | 1,408 | 2,432 | 4,480 | 8,576 |
|---|---|---|---|---|---|---|
| Tokens saved | 73.8% | 71.4% | 69.7% | 66.7% | 60.5% | 48.7% |
| Correct answers lost (of 13) | 0 | 0 | 0 | 0 | 0 | 0 |

- **On the base model the rule costs nothing on this draw**: none of its looping rows
  came back to a correct answer. On the edited model the same rule loses answers. So
  one setting does not fit both models, and a row stopped early has nothing to branch
  from at the later exit points.
- These accuracies are again a by-product of a timing run: one draw of 100 problems.

### Branches that wait for a slot

In the runs above nearly every branch was read back in, because a branch could share its
trunk's history only if a slot was free at the moment it was asked for. Now such a branch
**waits**. It keeps the trunk's pages (they stay in the pool while it waits) and a copy of
the trunk's small per-row state, about 25 MB, and takes the next slot that comes free,
ahead of rows that have not started. As many branches may wait as there are slots. One
more than that is read in later, as before, and so is a waiting branch whose trunk has
ended when the pool runs short.

**Same tokens.** A waiting branch starts from the numbers its trunk had when the branch
was asked for, so it writes what a branch started at once would have written: 400 of
400 sampled tokens in `bench.engine_checks`. Both go through the same code, so a second
check does not: with no randomness a branch must write what its trunk goes on to write,
and it does, for 200 of 200 tokens started at once and for 200 of 200 after a wait that
ended while the trunk was still running (most sharing branches start after a wait:
221 of 224 in the run below). The same sampled branch read in from saved tokens parted
after 23 tokens. mlx-lm's reading of a text is taken to round differently from writing
it token by token. That was not tested on its own.

**The edited model's rows and all four forced exits in one run**
(`--arm edited --exits 4096,8192,16384,24576`, 48 slots, 10 GB pool, the first draw's seeds):

| | Two runs (above) | One run, branches waiting |
|---|---|---|
| Wall-clock | 90.5 min (59.2 + 31.3) | **68.4 min** |
| New tokens | 3.00 million | 2.97 million |
| Tokens read back in | 2.90 million | 0.44 million |
| Branches sharing their trunk's history | 0 of 270 | 224 of 270, 221 of them after a wait |
| Steps | 84,510 | 69,039 |
| Peak memory | 15.2 GB with a 12 GB pool, then 13.6 GB | 13.4 GB |

- **22 minutes less for the same five conditions**, 17 of them from the change. The
  two earlier runs were made on the first evening, when long runs took 3 to 7% longer
  than the planner's clock (below). On that one clock the two runs come to 84.9
  minutes and the one run to 68.1.
- **The four forced exits add about 13 minutes to the rows alone** (12.6 against the
  second draw, made the same night; 13.0 by the planner). Reading them in afterwards
  added 31.
- **46 branches were still read in**: the waiting list was full (48, one per slot)
  when the first 48 rows reached 8,192 and then 16,384 tokens together. None lost its
  place because the pool ran short.

**The base model's rows and all four forced exits in one run**
(`--arm base --exits 4096,8192,16384,24576 --slots 48 --keep-free 16`, 10 GB pool). Nearly
every base row runs to the cap, so a waiting branch would wait a long time for a slot.
16 of the 48 slots are therefore kept for branches, and 32 rows run at once as before.

| | Branches read in (above) | Branches waiting, 16 slots kept for them |
|---|---|---|
| Wall-clock | 132.9 min | **117.2 min** |
| New tokens | 3.97 million | 4.06 million |
| Tokens read back in | 5.09 million | 0.40 million |
| Branches sharing their trunk's history | 1 of 364 | 375 of 375, 137 of them after a wait |
| Rows set aside and replayed | 13 | 19 |
| Steps | 127,446 | 131,072 |
| Peak memory | 13.5 GB | 13.8 GB |

- **16 minutes less (12%) while writing 2% more tokens**, 13 of them from the change
  by the planner's one clock (129.3 against 116.4 minutes). No branch was read in. What
  was read back are the 19 rows the pool had no room for: 10 trunks and 9 branches,
  0.39 million tokens.
- **The pool, not the slot count, limits this model.** A row at the cap holds 128
  pages and a 10 GB pool has 3,413, enough for 26 full rows. The 32 rows that start
  together reach the cap together, so near the end of a round the youngest rows
  running are set aside.
- **These are other rows than the 32-slot run's, from the same seeds.** The engine's
  code computes a row the same way in both runs (the edited model's rows below came out
  identical across the change). What differs is the width of the batch, 32 slots then
  and 48 now, and all 100 rows part from their 32-slot twins within the first 310
  tokens, half of them within 54. The first 32 rows sat in the same slots and started
  at the same step in both runs, so for them nothing else differs. This is a second
  draw of the base condition:

  | | 32-slot run | 48-slot run |
  |---|---|---|
  | Correct | 13 | 9 |
  | Rows at the cap | 89 | 92 |
  | Correct with a forced exit at 4,096, 8,192, 16,384, 24,576 | 40, 42, 43, 44% | 39, 38, 43, 45% |
  | Tokens saved by stopping after 512 to 8,576 looping tokens | 74 to 49% | 74 to 49% |
  | Correct answers that stopping loses | none of 13 | none of 9 |

  In the 48-slot run an answer budget of 2,000 tokens gives the same four accuracies
  as 4,000.

**The forced exits, drawn a second time.** A branch that shares its trunk's history
starts from the trunk's own numbers. One read in from saved tokens starts from mlx-lm's
reading of that text, which rounds differently. So the 224 sharing branches of this run
are a new draw:

| Forced exit at | Correct, read in afterwards | Correct, sharing in the one run |
|---|---|---|
| 4,096 | 39% | 40% |
| 8,192 | 37% | 40% |
| 16,384 | 45% | 49% |
| 24,576 | 52% | 54% |
| never (the run itself) | 47% | 49% |

With an answer budget of 2,000 tokens instead of 4,000 this run has 39, 40, 48 and 52%:
the larger budget added at most two correct answers per condition (at most one in the
earlier run).

**The rows of this run are the first draw's rows, except where a batch narrowed.** Both
runs used the same seeds, and 78 of the 100 rows came out identical, token for token.
Replaying both schedules on the CPU shows which (a comparison of the two ledgers, not
saved):

- **76 rows never met a narrowing of the batch in either run, and all 76 are
  identical.** 26 of them started 1,357 to 15,335 steps later in this run than in the
  first, next to other rows, in a smaller pool, and 191 branches were taken from the 76.
  47 of them wrote 16,384 tokens or more.
- **24 rows were still running when the first draw narrowed its batch** from 48 slots
  to 24, at step 43,723 (12 of them also when this run narrowed, at step 60,998). 22
  of the 24 differ between the two runs, each from a point 6 to 1,456 tokens after the
  narrowing and none before it. The other two ran on to the cap in both runs.
- **It is the width, not the move to another slot.** The narrowing moved 17 of the 24
  rows and left 7 in their slots, and 5 of those 7 differ as well. And 27 of the 78
  identical rows sat in different slots in the two runs.
- So a row's text does not depend on its neighbours, on its slot, on when it starts, on
  the pool or on its branches. It does depend on the width of the batch it is computed
  in. Which operation behaves differently at another width was not looked for.
- **The same on one text** (`bench.engine_checks`): forced through a batch of 48 that
  keeps its width, and through one that narrows to 24 after 200 steps, its scores are
  equal to the last digit for those 200 steps. After them they are up to 0.44 apart,
  with the same top token at 99.0% of steps, whether the narrowing moves the row or
  not. That is the size of the yardstick in "Is it the same computation?": the same
  arithmetic done in another order. Narrowing 8 slots to 4 changed nothing at all in
  the one quick check made of it (not saved).
- **To repeat a row its seed is not enough.** The batch must have the same width at
  every step the row lives through, and the row must not be read back in on the way
  (set aside, or a branch read in from saved tokens). A row's record in the ledger now
  says all three: the widths it ran at, how often it was set aside, whether it was read
  in (the nine runs here were made before that). `bench.phase1 --keep-width` holds
  the width to the end of a run. It has only been tried on a 12-problem run, where the
  batch then never narrowed. By the planner below it costs 6% on a 100-problem run of
  rows alone, 2 to 3% with the forced exits in the same run and under 1% on a run of
  1,000 problems (the same 100 rows ten times over).

### A planner for runs that were not made

`bench.plan_sim` takes the row lengths of a finished run and steps through the engine's
rules for them on the CPU, in about 5 seconds for everything below: slots, slots kept
free, branches that start at once, wait or are read in, the pool page by page, rows set
aside, the narrowing. Time comes from three measured costs: a step's fixed part by batch
width (23.6, 33.0 and 53.7 ms at 16, 48 and 96 slots), 45 billionths of a second per
token of history per row per step (the second draw, worked out above), and the time to
read a saved row back in (0.84 s for 4,096 tokens, 6.93 s for 24,576; a quick check, not
saved).

| Run | Started | Replay | Measured |
|---|---|---|---|
| Edited rows, first draw | 19:12 | 55.1 min | 59.2 min |
| Edited, 8,192 cap, seed 1 and seed 2 | 20:49, 21:01 | 12.0 min each | 12.1 min each |
| Edited forced exits, read in afterwards | 21:14 | 29.8 min | 31.3 min |
| Base rows and exits, 32 slots, branches read in | 21:46 | 129.3 min | 132.9 min |
| Base, thinking off | 00:20 | 5.6 min | 5.8 min |
| Edited rows, second draw | 00:34 | 55.0 min | 55.8 min |
| Edited rows and exits in one run | 01:40 | 68.1 min | 68.4 min |
| Base rows and exits in one run, 16 slots kept free | 02:50 | 116.4 min | 117.2 min |

- **The schedule is reproduced exactly.** For each of the nine runs the replay has the
  run's number of steps (10,177 to 131,072), of rows set aside (13 and 19 in the two
  base runs), of branches sharing, waiting and read in, and of tokens read back in, to
  the token (5,089,471 in the 32-slot base run). The script prints both sides. It is
  not a forecast: both runs in which the pool ran short were used to get the page
  counting right.
- **Nine runs do not exercise every rule.** A second review compared the planner with
  the engine line by line and found two rules written down wrongly that no real run
  had met: the order the engine takes its rows in after a narrowing, and that it
  narrows at the very next step. Both are corrected. One estimate moved, by 0.3%. A
  third rule has not fired in any real run either, a waiting branch giving up its
  pages when the pool runs short after its trunk has gone. The test suite now runs 29
  random small schedules through the engine itself, on a tiny model, and through the
  planner, and requires the same counts. 87 others agreed while it was written.
- **The clock is within 1.5% of the three long runs made after midnight**: the second
  draw, which the history cost was measured on (1.4% over the replay), and the two
  runs with waiting branches, which it was not (0.4 and 0.7%). The two 8,192-cap runs
  are within 0.5%.
- **The three long runs of the first evening took 3 to 7% longer than the replay.**
  For the first draw its log shows something else using the Mac (above). For the other
  two the cause was not established. The short thinking-off run took 3% longer, 0.2
  minutes.

Two runs were predicted before they were made, by the planner as it then was (the
predictions are in `bench/results/plan_sim.json` as it was committed before the runs
were started):

| | Predicted | Measured |
|---|---|---|
| Edited, rows and exits in one run | 73.0 min, 69,561 steps, 222 branches sharing, 48 read in | 68.4 min, 69,039 steps, 224 sharing, 46 read in |
| Base, rows and exits in one run | 114.5 min, 114,176 steps, 11 rows set aside, 363 branches sharing, 1 read in | 117.2 min, 131,072 steps, 19 set aside, 375 sharing, 0 read in |

- **Edited: steps within 1%, branch counts within 2, minutes 7% too high.** Its clock
  read too high: the history cost then came from the first draw's uneven minutes (51
  billionths of a second). With the second draw's 45 the same forecast is 69.1 minutes,
  1% too high.
- **Base: minutes 2% too low, steps 13% too low**, and the first is two errors that
  cancel. The clock read 5% too high, as above. And the schedule was too short. It was
  worked out from the rows of the 32-slot run, and the 48-slot run drew other rows
  (above): 92 at the cap instead of 89 and 375 branches instead of 364, enough to need
  a fourth round of trunks. Today's planner puts those old rows at 108.5 minutes, 7%
  under what the run took, and the run's own rows at 116.4.
- **The planner also counted the pool loosely then** (pages rounded up, the pool
  looked at once per stretch of up to 256 steps). Given the run's own rows it was 6%
  low in steps and set 16 rows aside where the engine set 19. It now counts pages as
  the engine does, before every step, which is what makes the replays above exact.
- **So the planner says how long given rows take, to within 1.5% on five runs and 3 to
  7% short on the first evening's long ones. It cannot know the rows of a run not yet
  made.** For a 100-problem base run that was worth 7%, because a few more rows at the
  cap added a round of trunks. A run of 1,000 problems depends less on any one row.
  And it knows nothing about other programs on the Mac. `docs/phase1-plan.md` uses it
  for sizes larger than 100 problems.

### The 500-problem run

The study's planned run (`docs/phase1-prereg.md`, results in `docs/phase1-results.md`)
was three arms on all 500 problems, each row computed at a batch width of 48 from start
to end (`--keep-width`). The plan gave 11.1 to 12.6 hours, from the planner fed with the
saved 100-problem rows repeated five times. Its three arms took 12.51 hours and wrote
31.5 million tokens, about 700 a second over the whole of it.

| Arm | Forecast before the run | Replay of its own rows | Measured |
|---|---|---|---|
| Base, thinking off, 32,768 cap | 22 to 100 min | 88.1 min | 88.4 min |
| Edited rows and four forced exits | 4.9 h | 288.0 min | 297.7 min |
| Base rows to 24,577 and four forced exits, 16 slots kept free | 5.9 to 6.1 h | 365.0 min | 364.6 min |

- **The schedule is reproduced exactly at five times the size.** The replays have the
  runs' 102,062, 287,608 and 412,771 steps. In the edited arm 1,305 branches shared
  their row's history (1,277 of them after waiting for a slot) and 45 were read in. In
  the base arm all 1,875 shared it (157 after a wait). No row was set aside.
- **The clock was within 0.3% on two arms and 3.4% short on the edited arm.** The Mac
  was not kept free of other work during this run, and the cause was not established.
  A one-line script that edits a notes file ran in the second minute of the first arm.
- **The forecast for thinking off was a wide range** because no saved run showed how
  many rows keep writing to a 32,768 cap with thinking off. 43 of 500 did.

### What was tried on the way

- **The first kernel looked at the page table once per token** and read the history at
  122 GB/s, less than half of MLX's (a quick check, not saved). Dealing each row's
  tokens to 64 groups so that a group takes 4 tokens of every page, with one look at
  the table per page and no bounds checks inside a page, brought it level with MLX's.
- **Not reading the value of a token with almost no weight** (under a billionth of the
  largest) left every score unchanged and saved nothing: 30.4 ms a step against 29.1
  on 16 rows with real 8,000-token histories. Dropped.
- **Holding half the slots back for branches** would nearly double the steps the
  trunks need. For the edited model none is held back and branches wait instead. For
  the base model, whose rows do not end before the cap, a third is held back: by the
  planner, with none held back the run takes 3% longer, sets 42 rows aside and reads
  135 of the 364 branches back in.

Limits: one model family (Qwen3.5 text models), one Mac, and single runs except where a
repeat is shown. The 32,768-cap comparison is two draws of the engine against one of
the study's runner (its second run wrote the same text as its first). A row's text
repeats only at the same batch width and if it is not read back in on the way. Two
CPU jobs of under a minute ran inside timed runs, going by file times: an analysis of
saved transcripts in the first minute of the first draw, and a unit test of the
statistics code in the third minute of the 32-slot base run. Files of this repository
were also edited and imported while the study's runner was timed again, and its time
still matched the record of 2026-11-03 within 0.6%.

Verdict: **keep**, for runs of many long generations. For one chat nothing changes.
