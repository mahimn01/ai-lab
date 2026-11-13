# Phase 1 on this Mac: what the row engine makes possible

A proposal, written 2026-11-12 and brought up to date with the runs of the following
night. Nothing here has been run beyond the 100-problem timing runs of
`SPEED-RESULTS.md` section 19. The study is `neural-harness`; its recommended Phase 1
arms are in its `RESULTS-P0.md`, section 6. The decisions marked "yours" at the end are
not made.

Words: a **row** is one generation, a **condition** a set of rows with the same
settings, the **cap** the most tokens a row may write, a **trunk** a full-length row and
a **branch** a row that continues a trunk from some point in another way.

## What was measured

All with Qwen3.5-0.8B, the study's 100 problems, one sample each, the study's sampling,
on the M1 Max with 48 slots.

| Condition | New tokens | Minutes | With the study's own runner |
|---|---|---|---|
| Edited model, thinking on, 32,768 cap (two draws) | 2.20 and 2.19 million | 59.2 and 55.8 | 103.5 (measured the same day) |
| The same with its four forced exits in the same run (at 4,096, 8,192, 16,384 and 24,576 tokens, 4,000-token answers) | 2.97 million | 68.4 | 3.8 to 6.2 hours as five separate runs (103.5 minutes of it measured, the rest an estimate) |
| Edited model, 8,192 cap (two seeds) | 0.76 million | 12.1 each | 18.1 (on record) |
| Base model, thinking on, 32,768 cap, with its four forced exits in the same run | 4.06 million | 117.2 | 4.9 to 7.5 hours as five separate runs (estimate) |
| Base model, thinking off, 8,192 cap | 0.28 million | 5.8 | 8.3 (on record, for 0.33 million tokens) |

The estimates are the tokens each separate run would write, divided by the two rates
on record for the study's runner: 716 tokens a second at an 8,192 cap and 339 at
32,768. The study ran its Phase 0 budget conditions as runs of their own, so that is
the comparison.

Before branches could wait for a slot the same conditions took 90.5 minutes for the
edited model (59.2 for the rows, 31.3 to add the exits from saved tokens) and 132.9 for
the base model. Those runs were made on the first evening, with other programs in use.
On the planner's one clock the change is worth 17 and 13 minutes of the 22 and 16.

While all 48 slots were busy the engine wrote 762 and 809 tokens a second in the two
draws, against 359 for the study's runner while its 16 were. A 100-row run is slower
than that on average (620 and 654), because its last 25 minutes run with slots
emptying. A larger run has less of that tail.

## What one trunk gives

- **The accuracy at every smaller cap.** The grader reads the last boxed answer in
  whatever text exists, so "the model at an 8,192 cap" is the trunk's first 8,192
  tokens. One 32,768-token trunk gives the whole accuracy-against-tokens curve. (A
  separate run at an 8,192 cap with the same seed matched 98 of 100 rows token for
  token, and had 15 correct where the cut has 14: its batch narrows at other steps.)
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
hundred units take 3.2 hours on this Mac: 68.4 minutes for the edited model, 117.2 for
the base model and 5.8 for thinking off.

| Problems x samples | Row engine | The study's runner (estimate) |
|---|---|---|
| 100 x 1 | 3.2 hours | 9 to 14 hours |
| 100 x 2 | 6.4 hours | 17 to 27 hours |
| 250 x 1 | 8.0 hours | 22 to 34 hours |
| 300 x 1 | 9.6 hours | 26 to 41 hours |
| 100 x 4 | 12.8 hours | 1.5 to 2.3 days |
| 500 x 1 | 15.9 hours | 1.8 to 2.8 days |
| 500 x 4 | 2.7 days | 7 to 11 days |

The engine's column is the measured hundred units multiplied up. Those three runs were
made after midnight, with nothing else noticed on the Mac. The first evening's long
runs, with other programs in use, took 3 to 7% longer than the planner's clock
(`bench.plan_sim`, `SPEED-RESULTS.md` section 19), so allow that much on a Mac in use.

As one long run it should be shorter. The planner puts 1,000 problems (the same 100
rows ten times over) at 2.90 hours per hundred units, 9% under the table: about 5%
because the slow end of a run is a smaller share of a long one, the rest because the
hundred-problem base run needed a fourth round of trunks for its last few rows.

- **Overnight (10 hours) fits about 300 units**: 300 problems with 1 sample, or 100
  problems with 3. By the table that is 9.6 hours, so there is room only if the Mac is
  left alone. As one long run the planner makes it 8.7 hours. The full 500 x 4 does
  not fit: 2.7 days by the table, 2.4 by the planner.
- **The hours scale with rows, not with conditions**, because the conditions of one
  model now come from one trunk. Adding an exit point adds branches, not a run.
- **Rules of thumb for other arms**: per 100 rows, an edited-model condition at the
  32,768 cap costs about an hour, and 68 minutes with four forced exits. The base
  model with its four exits costs about two hours (it nearly always runs to the cap).
  An edited-model condition at an 8,192 cap costs 12 minutes.
- **Two choices would shorten it** (both are in the list at the end; the percentages
  are the planner's, for a run of 1,000 problems):
  - An answer budget of 2,000 tokens instead of 4,000: 13% less time for the edited
    model and 7% for the base model. On the four timing runs with forced exits it
    changed at most two answers of a hundred per condition.
  - Stopping base trunks at 24,577 tokens, one past the last exit point: with the
    shorter answers the base model's part falls by a third. On both timing draws the
    base model gained no correct answer after 16,384 tokens. `bench.phase1` cannot do
    this yet: its one cap covers trunks and branches alike, so it needs a cap for
    trunks alone first.
  - Both together: 2.21 hours per hundred units instead of 2.90. Then about 450 units
    fit in 10 hours, 100 x 4 takes 8.9 hours and 500 x 4 about 44 hours.
- **Settings**: 48 slots and a 10 GB pool. The base model's run also keeps 16 slots
  back for branches (`--keep-free 16`). The planner puts 12 kept back at 2 to 8% less
  for a 100-problem run, depending on the draw (36 trunks at once can finish a hundred
  rows in three rounds instead of four), and at about 2% less for a run of 1,000
  problems. Neither was run.

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
  Mac's memory. That is half of a step at long histories and nothing tried shortens
  it.
- **Memory decides how many rows run at once.** A full 32,768-token row holds 0.4 GB.
  The base model, where nearly every row runs to the cap, fits about 26 rows in a 10 GB
  pool whatever the number of slots.
- **Stopping looping rows early is not free.** On four full-length runs of the edited
  model (the study's saved one and three of the engine's), a rule strict enough to
  lose no correct answer saved 6.5 to 11% of the tokens. Looser rules save up to 40%
  and lose up to 13 of about 50 correct answers. On both draws of the base model every
  rule lost nothing and saved 49 to 74%: none of its looping rows came back to a
  correct answer (22 correct rows in all, so this is not much evidence). But a row
  stopped early cannot be branched at the later exit points, so the rule does not fit
  the decisive test as it is written.
- **The answer budget of a forced exit.** With 4,000 tokens most forced rows were
  still writing when the budget ran out. Going from 2,000 to 4,000 added at most two
  correct answers per condition, so 2,000 looks enough. That rests on four runs of one
  draw each.
- **A row's text cannot always be repeated from its seed.** Near the end of a run the
  batch narrows as rows run out, and the rows still running then continue with other
  rounding. A row that is set aside and read back does the same: 19 rows in the
  100-problem base run, and by the planner about as many in a run of 1,000 problems.
  A branch read in from saved tokens does too (46 in the edited model's run). From now
  on a row's record in the ledger says which of these happened to it. Every such row
  is still a draw from the same model, so rates are not affected. Repeating one row
  exactly needs the same batch width at every step: `--keep-width`, under 1% more time
  on a large run by the planner. It has been tried only on a 12-problem run.

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
this model on an H200 wrote 14,190 tokens a second, about 22 times this Mac with the
engine.

## Yours to decide

1. **Go or no go** on any of the sizes above. Nothing larger than the 100-problem
   timing runs has been started.
2. **The answer budget of a forced exit.** The study's own ran to the overall cap; the
   timing runs used 4,000 tokens. 2,000 would save 13% of the edited model's time and
   7% of the base model's.
3. **Which base condition the edited model is compared with first** (thinking off, or
   a forced exit at one budget), fixed before the run.
4. **The grader.** Section 6.1 names math-verify. The timing runs used the study's own
   grader. math-verify is not installed here.
5. **Whether base trunks run to 32,768 or stop at 24,577.** The last exit point is at
   24,576, so the final 8,192 tokens of a base trunk only serve "base at a 32,768 cap".
   Stopping early needs a small change to the runner first (a cap for trunks alone).
6. **Samples per problem.** With 4, the early-decision test still counts each problem
   once.
7. **Whether the base model needs all four exit points.** On the two timing draws its
   forced-exit accuracy was 40, 42, 43 and 44%, then 39, 38, 43 and 45%, from the
   earliest point to the latest. The two late points wrote half of its branch tokens
   and need the trunks to run on to 24,576.
8. **Whether single rows must be repeatable** (`--keep-width`). It costs under 1% on a
   large run and 2 to 6% on a 100-problem one.
9. **Whether the timing runs' rows count toward the study.** They were made as
   benchmarks, with no plan fixed before them, and their accuracies have been looked
   at. Starting the study in a new ledger under a plan file (`--plan`, whose hash the
   ledger records) keeps it clean, and costs the 3.2 hours again.
