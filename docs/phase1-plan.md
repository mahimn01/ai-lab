# Phase 1 on this Mac: what the row engine makes possible

A proposal, written 2026-11-12. Nothing here has been run beyond the 100-problem timing
runs of `SPEED-RESULTS.md` section 19. The study is `neural-harness`; its recommended
Phase 1 arms are in its `RESULTS-P0.md`, section 6. The decisions marked "yours" at the
end are not made.

Words: a **row** is one generation, a **condition** a set of rows with the same
settings, the **cap** the most tokens a row may write, a **trunk** a full-length row and
a **branch** a row that continues a trunk from some point in another way.

## What was measured

All with Qwen3.5-0.8B, the study's 100 problems, one sample each, the study's sampling,
on the M1 Max with 48 rows at once (32 for the base model).

| Condition | New tokens | Minutes | With the study's own runner |
|---|---|---|---|
| Edited model, thinking on, 32,768 cap | 2.20 million | 59.2 | 103.5 (measured the same day) |
| Its four forced exits (at 4,096, 8,192, 16,384 and 24,576 tokens, 4,000-token answers), added from the saved rows | 0.80 million, plus 2.9 million read back in | 31.3 | 2.1 to 4.4 hours as four separate runs (estimate) |
| Edited model, 8,192 cap | 0.76 million | 12.1 | 18.1 (on record) |
| Base model, thinking on, 32,768 cap, with its four forced exits in the same run | 3.97 million, plus 5.1 million read back in | 132.9 | 4.8 to 7.3 hours as five separate runs (estimate) |
| Base model, thinking off, 8,192 cap | 0.28 million | 5.8 | 8.3 (on record) |

The estimates are the tokens each separate run would write, divided by the two rates
on record for the study's runner: 716 tokens a second at an 8,192 cap and 339 at
32,768. The study ran its Phase 0 budget conditions as runs of their own, so that is
the comparison.

While all 48 slots are busy the engine writes 668 tokens a second on 32,768-cap rows of
the edited model, against 337 to 339 for the study's runner. A 100-row run is slower
than that on average (620), because its last 25 minutes run with slots emptying. A
larger run does not have that tail.

## What one trunk gives

- **The accuracy at every smaller cap.** The grader reads the last boxed answer in
  whatever text exists, so "the model at an 8,192 cap" is the trunk's first 8,192
  tokens. One 32,768-token trunk gives the whole accuracy-against-tokens curve.
- **Each forced exit**, as a branch that starts where the trunk stood at that token
  count. A branch needs its answer tokens and, if it is made later from saved rows, one
  reading of the trunk's text up to that point.
- **Loop marks** for every row: when its first loop began and its longest loop.

## The decisive test as rows

`RESULTS-P0.md` 6.1 asks whether a decoding fix on the base model matches the edited
model at the same token budget.

| What the study wants | How it is obtained |
|---|---|
| Edited model at 4k, 8k, 16k, 24k, 32k | one edited trunk, cut |
| Edited model with a forced exit at 4k, 8k, 16k, 24k | four branches per edited trunk |
| Base model with a forced exit at 4k, 8k, 16k, 24k | four branches per base trunk |
| Base model, thinking on, at each cap | the base trunk, cut |
| Base model, thinking off | its own short rows |

Two trunks per problem and sample, their branches, and one short row. Before, each
line of the study's list was a run of its own.

## Sizes and hours

One unit is one problem with one sample, run through everything in the table above:
both trunks, their eight forced exits and the short thinking-off row. Measured, a
hundred units take 3.8 hours on this Mac: 90.5 minutes for the edited model, 132.9 for
the base model and 5.8 for thinking off.

| Problems x samples | Row engine | The study's runner (estimate) |
|---|---|---|
| 100 x 1 | 3.8 hours | 9 to 14 hours |
| 100 x 2 | 7.6 hours | 17 to 27 hours |
| 250 x 1 | 9.6 hours | 22 to 34 hours |
| 100 x 4 | 15.3 hours | 1.5 to 2.3 days |
| 500 x 1 | 19.1 hours | 1.8 to 2.8 days |
| 500 x 4 | 3.2 days | 7 to 11 days |

- **Overnight (10 hours) fits 100 problems with 2 samples, or 250 problems with 1.**
  The full 500 x 4 does not: it is three days and a night of this Mac doing nothing
  else.
- **The hours scale with rows, not with conditions**, because the conditions of one
  model now come from one trunk. Adding an exit point adds branches, not a run.
- **Rules of thumb for other arms**: per 100 rows, an edited-model condition at the
  32,768 cap costs about an hour, a base-model one about an hour and a half (it nearly
  always runs to the cap), and an edited-model one at an 8,192 cap 12 minutes.
- **Two choices would shorten it** (both are in the list at the end):
  - An answer budget of 2,000 tokens instead of 4,000 removes 45% of the branch
    tokens. On the timing runs it changed at most one answer per condition.
  - Stopping base trunks at 24,576 tokens, the last exit point, removes 24% of the
    base trunk tokens, and those are the slowest ones to write.
- **Not yet as fast as it could be**: in the timing runs nearly every branch was read
  back in from saved tokens instead of sharing its trunk's memory, because no slot was
  free when it was asked for. That reading was about 21 of the base model's 133
  minutes and 12 of the edited model's 31.

## Stopping when the answer is in

`lab/stats/sequential.py` has a paired test that may be looked at after every problem
without raising its false-alarm rate. Problems are taken in an order fixed before the
run, and each problem counts once however many samples it has.

- A large effect ends early: on the study's saved runs "the base model loops more than
  the edited one" would have been decided at problem 18 and at problem 26 of 100.
- For Phase 0's open question (edited at 32,768 against base with thinking off, 54%
  against 43%), a simulation with that gap decides within 500 problems in 92% of runs,
  at a median of problem 195. One test at 500 problems would find it in 99%. With no
  gap the early test stops wrongly in 3% of runs (its limit is 5%).
- The price: a real effect is found a little less often than by one test at the
  planned size, and a small one much less often (a 3-point gap: 34% against 74% of
  simulated runs at 2,000 pairs). So the early-decision test fits the main question,
  and the planned-size test stays for everything else.

## What stays expensive

- **A row's history must be read for every token it writes**, at the speed of this
  Mac's memory. That is half of a step at long histories and nothing found tonight
  shortens it.
- **Memory decides how many rows run at once.** A full 32,768-token row holds 0.4 GB.
  The base model, where nearly every row runs to the cap, fits about 26 rows in a 10 GB
  pool whatever the number of slots.
- **Stopping looping rows early is not free.** On two full-length runs of the edited
  model, a rule strict enough to lose no correct answer saved 6.5% and 11% of the
  tokens. Looser rules save up to 40% and lose up to 6 of about 50 correct answers.
  On the base model every rule lost nothing and saved 49 to 74%, because its looping
  rows do not come back. But a row stopped early cannot be branched at the later exit
  points, so the rule does not fit the decisive test as it is written.
- **The answer budget of a forced exit.** With 4,000 tokens most forced rows were
  still writing when the budget ran out. Going from 2,000 to 4,000 added at most one
  correct answer per condition, so 2,000 looks enough, but that rests on one draw.

## The other arms of section 6

| Arm | What the engine offers | Not built |
|---|---|---|
| 6.2 Gradient LoRA at matched compute | training through `lab.finetune.lora`; evaluating both adapters in one batch as per-row edits | the training objectives |
| 6.3 Directional ablation at four strengths | four edits as per-row edits, one batch | finding the direction (the study's `sd1/` code does it) |
| 6.4 Dose-response, a timed adapter, composing with decoding fixes | the strength is a per-row setting, so a sweep is one batch; forced exits compose as branches | an edit that switches on part-way through a row (a branch may not change its trunk's edit yet) |
| 6.5 Held-out benchmarks | rows are just prompts with a cap; the grader is the caller's | the benchmarks' prompts and graders |
| 6.6 Presence-penalty sweep | the penalty is a per-row setting, so three penalties and two models are one batch | nothing |

## Moving to a cluster

Row definitions, the ledger, the loop test, the stopping rules and the statistics do
not depend on the Mac. The engine itself does. On NVIDIA GPUs the same `run(rows,
ledger, ...)` could be served by a thin client of a vLLM server, which already keeps
its history in pages. That client is not built and could not be tested here. Its
sampler order would need the same check against the study's sampler as the engine got
(`bench/engine_checks.py`) before a number from it is trusted. The published run of
this model on an H200 wrote 14,190 tokens a second, 21 times this Mac with the engine.

## Yours to decide

1. **Go or no go** on any of the sizes above. Nothing larger than the 100-problem
   timing runs has been started.
2. **The answer budget of a forced exit.** The study's own ran to the overall cap; the
   timing runs used 4,000 tokens.
3. **Which base condition the edited model is compared with first** (thinking off, or
   a forced exit at one budget), fixed before the run.
4. **The grader.** Section 6.1 names math-verify. The timing runs used the study's own
   grader. math-verify is not installed here.
5. **Whether base trunks run to 32,768 or stop at 24,576.** The last exit point is at
   24,576, so the final 8,192 tokens of a base trunk only serve "base at a 32,768 cap".
6. **Samples per problem.** With 4, the early-decision test still counts each problem
   once.
7. **Whether the base model needs all four exit points.** On the one timing draw its
   forced-exit accuracy was 40, 42, 43 and 44% from the earliest point to the latest.
   The two late points wrote half of its branch tokens and need the trunks to run on
   to 24,576.
