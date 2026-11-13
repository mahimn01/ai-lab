# Phase 1, the decisive test: the plan, fixed before the run

Written on 2026-11-13, before any row of this run exists. The commit that adds this file
comes before the run starts, and `bench.phase1` writes the file's sha256 into every ledger
it starts (`--plan`). A ledger started under this file cannot be continued or reported
under another. The file is not edited once the run has started. Whatever turns out
different is listed with the results as a deviation from this plan.

The study is `neural-harness`. Its Phase 0 results and its list of Phase 1 arms are in its
`RESULTS-P0.md`. This plan covers the first arm of that list (6.1), which it calls the
decisive test.

Words: a **row** is one generation for one problem. The **cap** is the most tokens a row
may write. A **forced exit** at 8,192 means that a row still thinking after 8,192 tokens
is made to write the exit phrase of the Qwen3 report, which closes its thinking, and then
answers. A **condition** is one way of producing an answer for every problem. An **arm**
is one run of the engine, which gives several conditions.

## The question

The base model, Qwen3.5-0.8B, loops while it thinks on most MATH problems and runs out of
tokens. The edited model, the same weights with two matrices changed by hand, does not.
Phase 0 found that most of the edited model's gain is also there without touching the
weights, if the base model is simply made to stop thinking. What it left open is whether
the edit adds anything beyond that. So: with about the same number of tokens spent on a
problem, is the edited model right more often than the base model with a decoding fix?

## The decisions (2026-11-13)

1. **Size**: all 500 problems of MATH-500, one sample each.
2. **Primary comparison**: the edited model at its 32,768 cap against the base model
   with a forced exit at 24,576 tokens.
3. **Thinking off** runs with the same 32,768 cap as the other arms. (Phase 0 used
   8,192. That number still comes out, by cutting the rows.)
4. **A forced exit may write 2,000 tokens** after the exit phrase.
5. **Base rows stop at 24,577 tokens**, one past the last exit point. The rest of a base
   row would only serve "the base model at a 32,768 cap", which is not part of this test.
6. **Grader**: math-verify, as the blog used.
7. **The batch keeps one width** (`--keep-width`), so that a row can be repeated from its
   seed.
8. **New ledgers under this plan.** The timing runs of 2026-11-12 and 13 do not count:
   they were made as benchmarks, with no plan, and their accuracies have been looked at.
9. **When to stop**: if anything fails, or the run would take much longer than about 13
   hours, it is stopped and written up. Nothing is patched on the way.

## Further details fixed here

These fill in what the decisions above leave open.

- **Run seed 20261113.** Seeds 0, 1, 2 and 99 were used here before, this one never. A
  row's random numbers come from the run seed, the problem's id and the sample number.
- **Order of the problems**: `random.Random(0).sample(range(500), 500)`, the study's own
  way of drawing. Its first 100 are the 100 problems of Phase 0.
- **Both models get all four exit points**: 4,096, 8,192, 16,384 and 24,576.
- **Engine settings**: 48 rows at once, a 10 GB history pool, and in the base arm 16
  slots kept free for branches.
- **Order of the arms**: thinking off, then edited, then base.
- **Hard time limit**: 15 hours, enforced by `lab.run`.

## What is run

| Arm | Model | Thinking | A row stops at | Forced exits |
|---|---|---|---|---|
| `p1-nothink` | base | off | 32,768 | none |
| `p1-edited` | edited | on | 32,768 | 4,096, 8,192, 16,384, 24,576 |
| `p1-base` | base | on | 24,577 | 4,096, 8,192, 16,384, 24,576 |

Sampling is the study's: temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5.
The prompt, the exit phrase and the stop tokens are imported from the study's code.

The whole run is one command. The script holds the three `bench.phase1` commands and the
analysis command, and is part of the commit that adds this plan.

```
cd ~/Documents/Dev/ai-lab
uv run python -m lab.run --name phase1 --max-hours 15 -- \
    zsh bench/phase1_study.zsh p1 docs/phase1-prereg.md 500 1 20261113
```

## Conditions

Twenty-three conditions come out of the three arms. Each has a name used in the results.

- `edited@C` for C in 4,096, 8,192, 16,384, 24,576, 32,768: the edited model's rows cut
  after C tokens. The last one is the row as it ran.
- `edited+exit@B` for B in 4,096, 8,192, 16,384, 24,576: the edited model with a forced
  exit at B. A row that had finished, or closed its thinking, before B counts as it is.
- `base@C` for C in 4,096, 8,192, 16,384, 24,576, and `base@24577` for the row as it ran:
  the base model with nothing done about its looping. For reference only.
- `base+exit@B` for the same four B: the base model with a forced exit.
- `nothink@C` for the five C: the base model with thinking off.

For each condition the results give the share of problems right and the mean number of
tokens a problem used. Those two numbers per condition are the study's accuracy against
tokens curves.

## Grading

math-verify 0.9.0, in the blog's own protocol, written from the blog's code
(`bench/grading.py`):

- Only the text after the last `</think>` is read, or the whole text if thinking never
  closed.
- The answer is what the last `\boxed{}` holds. No closed box means no answer, and that
  is wrong.
- It is right when `verify(parse("\boxed{gold}"), parse("\boxed{answer}"))` says so.

Checked before the run: it gives the published verdict on all 40 published generations
(`bench/results/grader_check.json`). Each arm is also graded with the study's Phase 0
grader, to hold against Phase 0. That second number is reported next to the first and
decides nothing.

## Analysis

The analysis is `bench/phase1_analysis.py`, committed before the run and called by the
script with the comparisons below.

**Primary.** `edited@32768` against `base+exit@24576`. On the timing runs the edited
model used about 22,000 tokens a problem and this base condition about 25,000 (with
4,000-token answers), which made it the base fix closest in cost.

- The number reported is the difference in the share of problems right, edited minus
  base, with a 95% interval.
- The test is the exact McNemar test, two-sided, at the 5% level. It looks only at the
  problems that one of the two got right and the other got wrong.

**Secondary**, seven comparisons, each reported the same way, with its p-value also
corrected for there being seven (Holm):

1. `edited@32768` against `nothink@32768`.
2. `edited@32768` against `nothink@8192`, which repeats Phase 0's open comparison.
3. `edited@32768` against `base+exit@16384`, the next cheaper base fix.
4. to 7. `edited+exit@B` against `base+exit@B` for each of the four B: the two sets of
   weights under the same decoding fix.

**What the primary result will be taken to mean.**

- p below 0.05 and the edited model ahead: at this size the edit adds accuracy beyond
  what this decoding fix gives the base model.
- p below 0.05 and the base model ahead: the decoding fix alone does better than the
  edit.
- Otherwise: no difference was found. The interval then says how large a difference
  either way the data still allow. This is not evidence that the two are the same.

**Rules.**

- No row is left out. A row that hits its cap with no boxed answer is wrong. So is a
  forced exit that uses up its 2,000 tokens without one.
- There is no early stopping on results. Nobody looks at accuracies while the run is
  going. The three arms run to the end, or to the time limit.
- If the run is cut short, only the front of the problem order is analysed: the problems
  before the first one that is not complete in all three arms. The rows that happen to
  have finished are not a fair sample, because short rows finish first.
- Intervals: Wilson's for a condition's accuracy. For a difference, the mean of the
  per-problem differences plus and minus 1.96 standard errors.
- Also reported, without a test: rows at the cap, forced exits that used up their answer
  budget, rows set aside and read back in, wall-clock time and tokens per arm.

**What this size can show.** On the saved 100-problem runs the primary pair disagreed
on 14 problems of 100. If that holds, 500 problems give an interval of about plus or
minus 3.3 points. A true difference of 3.3 points would then be found about half the
time, and it takes about 4.7 points for a four in five chance.

## What has already been seen

- **The timing runs**: the 100 Phase 0 problems, run several times with seeds 0, 1
  and 2. Regraded with math-verify, the last pair of runs gave `edited@32768` 50%,
  `base+exit@24576` 46% and `base+exit@16384` 43% (both with 4,000-token answers). Two
  other draws of `edited@32768` gave 48% and 57%, and `nothink@8192` gave 45%. The
  answer budget, the base cap and the choice of the primary comparison were made
  knowing these numbers.
- **The smoke run** of this script: 12 problems, seed 99, in `bench/results/phase1.json`
  as `smoke-*`. It was a test of the pipeline.

This run uses another seed, so every row is a new draw. 399 of its 500 problems have
never been run here.

## Time

The run planner (`bench.plan_sim`), fed with the saved rows repeated to 500 problems,
gives 11.1 to 12.6 hours: 4.9 for the edited arm, 5.9 to 6.1 for the base arm, and 22 to
100 minutes for thinking off. The spread there is the unknown share of rows that keep
writing to the cap. On the smoke run the planner was within 2.1% on each arm.

## Kept the same as Phase 0, and what differs

- Same: the models and their hashes (the study's `MANIFEST.md`), MATH-500 at revision
  `6e4ed1a2`, the prompt, the sampling and the exit phrase.
- Different: the row engine of this repository instead of mlx-lm's batch generator, so
  random streams differ from Phase 0's. The grader. A forced exit has an answer budget
  where Phase 0's ran on to the overall cap.
- A row's text depends on the width of the batch, which is held at 48 here. It also
  changes if the engine sets the row aside for lack of memory and reads it back in. Each
  row's record says whether that happened. Such a row is still a draw from the same
  model.
- Software: mlx 0.32.4, mlx-lm 0.32.2, math-verify 0.9.0 with latex2sympy2_extended
  1.11.0, antlr4 4.13.2 and sympy 1.14.0, Python 3.13. Machine: Mac Studio, M1 Max,
  32 GB.
