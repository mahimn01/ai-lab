# Phase 1, the decisive test: results

The run was made on 2026-11-13 and 14 under `docs/phase1-prereg.md`. GitHub recorded the push of that plan at 19:31:52 UTC, the run's log starts at 19:32:03, and the plan's sha256 is in all three ledgers. Words mean what they mean in the plan. A row is one generation for one problem. The cap is the most tokens a row may write. A forced exit at 8,192 makes a row that is still thinking after 8,192 tokens write the exit phrase and then answer.

## The answer

The edited model was right on 50.6% of the 500 problems, and the base model with a forced exit at 24,576 tokens on 35.8%. The difference is 14.8 points with a 95% interval of 10.8 to 18.8, and the exact McNemar test gives p = 2.0 x 10^-12. That was the plan's primary comparison. By the reading fixed in the plan, the edit adds accuracy beyond what this decoding fix gives the base model.

A planned secondary comparison limits what that means. The base model with thinking switched off was right on 50.4%, and it used 7,918 tokens a problem against the edited model's 21,745. The difference is 0.2 points with an interval of -4.0 to +4.4, so no difference was found. Thinking off is also a fix that leaves the weights alone. The run shows the edit far ahead of one such fix and not ahead of another.

A forced answer could be 2,000 tokens long, where the study's Phase 0 write-up had asked for a generous allowance. Two thirds of the base model's forced rows used all 2,000 without boxing an answer. Right answers had nearly stopped arriving by then, and at that rate twice the room would have added about 2 points to the base model's figure. Far longer answers were not tried.

## What ran

| Arm | Model | Thinking | Time | Tokens written | Rows that reached the cap |
|---|---|---|---|---|---|
| `p1-nothink` | base | off | 88.4 min | 3,958,984 | 43 of 500, at 32,768 |
| `p1-edited` | edited | on | 4.96 h | 13,050,457 | 150 of 500, at 32,768 |
| `p1-base` | base | on | 6.08 h | 14,461,352 | 464 of 500, at 24,577 |

Every problem finished in every arm. The run started at 14:32 EST and ended at 03:04, which is 12 hours and 32 minutes and inside the plan's estimate of 11.1 to 12.6 hours. Peak memory was 13.4 GB. The code was commit `e7443c0`, the one that added the plan.

## Every condition

A condition's name gives the arm and the cut. `edited@8192` is the edited model's rows cut after 8,192 tokens, and `base+exit@8192` is the base model with a forced exit there. "Right" is by math-verify, the plan's grader, with Wilson's interval. The study's Phase 0 grader is the second grading the plan asks for, and it decides nothing. The tokens are the mean a problem used.

| Condition | Right | 95% interval | Study's grader | Tokens a problem |
|---|---|---|---|---|
| `nothink@4096` | 39.8% | 35.6% to 44.2% | 38.6% | 2,322 |
| `nothink@8192` | 45.0% | 40.7% to 49.4% | 43.4% | 3,852 |
| `edited@4096` | 5.2% | 3.6% to 7.5% | 5.4% | 4,011 |
| `base@4096` | 5.0% | 3.4% to 7.3% | 4.8% | 4,025 |
| `edited+exit@4096` | 36.4% | 32.3% to 40.7% | 34.8% | 5,473 |
| `base+exit@4096` | 34.4% | 30.4% to 38.7% | 32.8% | 5,504 |
| `nothink@16384` | 48.8% | 44.4% to 53.2% | 47.2% | 6,000 |
| `nothink@24576` | 50.2% | 45.8% to 54.6% | 48.6% | 7,158 |
| `edited@8192` | 13.6% | 10.9% to 16.9% | 13.6% | 7,736 |
| `base@8192` | 7.2% | 5.2% to 9.8% | 6.8% | 7,908 |
| `nothink@32768` | 50.4% | 46.0% to 54.8% | 48.6% | 7,918 |
| `edited+exit@8192` | 41.0% | 36.8% to 45.4% | 39.0% | 9,224 |
| `base+exit@8192` | 38.6% | 34.4% to 42.9% | 37.0% | 9,350 |
| `edited@16384` | 28.0% | 24.2% to 32.1% | 27.2% | 14,103 |
| `base@16384` | 8.2% | 6.1% to 10.9% | 7.8% | 15,542 |
| `edited+exit@16384` | 48.2% | 43.9% to 52.6% | 46.2% | 15,703 |
| `base+exit@16384` | 38.2% | 34.0% to 42.5% | 36.6% | 16,951 |
| `edited@24576` | 42.4% | 38.1% to 46.8% | 40.8% | 18,734 |
| `edited+exit@24576` | 51.0% | 46.6% to 55.4% | 49.4% | 19,985 |
| `edited@32768` | 50.6% | 46.2% to 55.0% | 49.0% | 21,745 |
| `base@24576` | 8.2% | 6.1% to 10.9% | 7.8% | 23,144 |
| `base@24577` | 8.2% | 6.1% to 10.9% | 7.8% | 23,145 |
| `base+exit@24576` | 35.8% | 31.7% to 40.1% | 34.2% | 24,599 |

The second grader is within 2 points of the first on every condition. It reads the last box anywhere in the text and compares strings. On the rows as they ran it marked 19 wrong that math-verify marked right, and none the other way. Under it the primary difference is again 14.8 points (94 problems against 20), and the seven other differences move by at most 0.4.

## The planned comparisons

Differences are in percentage points, first minus second. The two counts are the problems that only one of the pair got right. The exact McNemar test looks at those problems alone and asks how likely a split at least this uneven would be if each were as likely to fall to either side. The seven secondary p-values are also corrected for there being seven (Holm).

| | First | Second | Difference | 95% interval | Only the first right | Only the second right | p | p, Holm |
|---|---|---|---|---|---|---|---|---|
| Primary | `edited@32768` | `base+exit@24576` | +14.8 | +10.8 to +18.8 | 95 | 21 | 2.0 x 10^-12 | |
| 1 | `edited@32768` | `nothink@32768` | +0.2 | -4.0 to +4.4 | 58 | 57 | 1 | 1 |
| 2 | `edited@32768` | `nothink@8192` | +5.6 | +1.6 to +9.6 | 68 | 40 | 0.009 | 0.036 |
| 3 | `edited@32768` | `base+exit@16384` | +12.4 | +8.6 to +16.2 | 82 | 20 | 4.3 x 10^-10 | 2.6 x 10^-9 |
| 4 | `edited+exit@4096` | `base+exit@4096` | +2.0 | -1.4 to +5.4 | 42 | 32 | 0.295 | 0.638 |
| 5 | `edited+exit@8192` | `base+exit@8192` | +2.4 | -1.1 to +5.9 | 45 | 33 | 0.213 | 0.638 |
| 6 | `edited+exit@16384` | `base+exit@16384` | +10.0 | +6.4 to +13.6 | 69 | 19 | 7.8 x 10^-8 | 3.9 x 10^-7 |
| 7 | `edited+exit@24576` | `base+exit@24576` | +15.2 | +11.4 to +19.0 | 91 | 15 | 1.9 x 10^-14 | 1.3 x 10^-13 |

## What the numbers say

The edited model beats the base model's forced exit. It is 14.8 points ahead of the exit at 24,576 and 12.4 ahead of the exit at 16,384. The base model's four exits lie between 34.4% and 38.6%, so it is at least 12 points ahead of each exit point that was run.

A forced exit usually ends without a boxed answer, in either model. Of the 464 base rows forced at 24,576, 143 gave the right answer, 7 boxed a wrong one, and 314 wrote all 2,000 tokens and boxed nothing. Of those 314, 128 end in a loop by the study's test and the other 186 do not. The edited model's forced rows fare the same with less looping: of 470 forced at 4,096, 152 were right, 6 boxed a wrong answer and 312 boxed nothing, and 9 of them end in a loop.

So the answer budget is part of what a forced exit scores here. The study's Phase 0 write-up had asked for a generous allowance, and the plan fixed 2,000 tokens because the earlier 100-problem runs, which had 4,000, gained almost nothing after 2,000. This run agrees with them as far as it reaches. In the last 500 tokens of the budget the base model's four exits gained 1, 2, 0 and 3 right answers. At that rate twice the room adds about 2 points at the 24,576 exit, against a lead of 14.8. What a forced answer would do with much more room, as in Phase 0 where it ran on to the overall cap, this run cannot say.

Under the same forced exit the two models are close early and far apart late. At 4,096 and 8,192 the edited model is ahead by 2.0 and 2.4 points, and neither interval excludes zero. At 16,384 and 24,576 it is ahead by 10.0 and 15.2. A row that is forced is right about as often in either model, 30 to 34% of the time in seven of the eight cases (the eighth is the edited model's 161 rows forced at 24,576, at 21%). The two models differ in how many rows are forced at all. By 24,576 tokens 339 edited rows had finished or closed their thinking by themselves, and 221 of those were right. Only 36 base rows had, all of them right.

No difference was found between thinking off and the edited model, and thinking off used a little over a third of the tokens. The two are right on 50.4% and 50.6% of the problems, and partly on different ones: 58 that only the edited model got, 57 that only thinking off got, and 195 that both did. Against thinking off cut at 8,192 tokens, which was Phase 0's open comparison, the edited model is 5.6 points ahead (p = 0.009, and 0.036 after Holm's correction). That lead comes from the unequal caps. With thinking off a third of the rows (166) write more than 8,192 tokens, and with the edited model's 32,768 the same rows go from 45.0% to 50.4%.

So the plan's question has two answers. It asked whether the edited model, at about the same number of tokens, is right more often than the base model with a decoding fix. Against a forced exit with a 2,000-token answer it is. Against thinking off, which costs fewer tokens, no difference was found, and the interval leaves room for about 4 points either way. All of this is one small model on one benchmark with one sample a problem, where Phase 0 had asked for four or more.

## Also reported

The plan lists four things to report without a test.

Rows at the cap are in the first table: 43 with thinking off, 150 for the edited model, and 464 for the base model, whose cap was 24,577.

Forced exits that used up their answer budget:

| Exit at | Edited rows forced | Of them, wrote all 2,000 tokens | Base rows forced | Of them, wrote all 2,000 tokens |
|---|---|---|---|---|
| 4,096 | 470 | 311 | 479 | 318 |
| 8,192 | 420 | 283 | 468 | 302 |
| 16,384 | 299 | 202 | 464 | 298 |
| 24,576 | 161 | 122 | 464 | 314 |

None of those 2,150 branches boxed an answer. With a budget of 1,000 tokens the eight forced-exit conditions would have scored 1.0 to 4.2 points lower.

No row was set aside in any arm. 45 forced-exit branches of the edited arm, all at the 8,192 exit, were read in from the row's saved tokens, which the engine does when a branch can neither start at once nor wait for a slot. 13 of the 45 were right (29%), against 116 of the other 375 branches at that exit (31%).

Time and tokens for each arm are in the first table. Given each arm's own row lengths afterwards, the run planner reproduces the engine's schedule exactly (the same number of steps and of branches sharing, waiting and read in) and puts the three arms at 88.1, 288.0 and 365.0 minutes, against 88.4, 297.7 and 364.6 measured. The edited arm took 3.4% longer than its replay. The Mac was not kept free of other work during the run, and the cause was not established.

## Checks made after the run

`bench/phase1_checks.py` reads the three ledgers again and saves what it finds in `bench/results/phase1_checks.json`. All 30 checks passed.

- Each ledger carries the plan's sha256, has one finished row for each of the 500 problems and holds nothing else. Every row and branch was computed at a batch width of 48. The settings in the ledgers are the plan's: seed 20261113, a cap of 32,768 with thinking off and for the edited model and of 24,577 for the base model, the four exit points, and 2,000-token answers. They were held against the plan by eye, because the plan is prose.
- Every row still thinking at an exit point has its branch (1,350 in the edited arm, 1,875 in the base arm), and there is no other branch. Each one starts with the exit phrase and has the 2,000-token budget. The run's own summary found each by its key, a hash of the row's prompt, its tokens up to the exit point, the seed and the budget.
- Every verdict was made again from the saved tokens, looked up by the problem's id, and equals the one the analysis used. All eight comparisons, counted again from those verdicts, give the saved counts and p-values.
- The forced answers are text the model could have written. A branch continues from its row's history inside the engine, and damaged history would still give readable text that is only wrong more often. The base model's exit at 24,576 is made one token before its row stops, a path added for this run. So 60 of those 464 answers were read again by mlx-lm without the engine, along with 8 at each other exit point of both arms and 8 of the 45 branches that were read in, 124 in all. At every sampled token the check asks whether mlx-lm's own scores could have produced it. Of the 93,742 tokens of the 60 answers, 36 could not, which is 1 in 2,604. The other groups together have 1 in 2,624, and the rows' own last tokens before an exit 1 in 4,535. Rates of this size are what rounding gives when the same scores are computed two ways (`bench/engine_checks.py` counted 1 and 2 in 8,192 on other text). Two answers of each group were also scored after another row's text, and there 1 token in 58 could not have been drawn.
- If the exit at 24,576 is set aside all the same, the edited model is 12.4 points ahead of the exit at 16,384, which is not made at a row's end.

## Deviations from the plan

Nothing that was run or analysed differs from the plan. The script, the seed, the order of the arms and the analysis code are those of commit `e7443c0`, and the run computed the comparisons itself as its last step. The second grading was made after the run, from the saved tokens, with the commands at the end of this file.

One rule of the plan cannot be vouched for. It says nobody looks at accuracies while the run is going. The log printed each arm's summary as the arm ended, at 16:00 and 20:59, and no record was kept of who read it. Nothing could be steered by it: the run was one command with no stopping rule, from code committed before it started.

Two things came out differently from what the plan expected. The primary pair disagreed on 116 of the 500 problems (23%), where the plan had guessed 14% from an earlier 100-problem draw, so the interval is plus or minus 4.0 points where the plan expected 3.3. And the Mac was not reserved for the run, which matters for its timings only.

## Looked at after the results

None of this was planned, so none of it is a test.

- The 100 problems of Phase 0 scored higher than the rest. They are the first 100 of the order. On them the edited model is right on 56 against 197 of the other 400 (49%), thinking off on 56 against 196 (49%), and the base model's forced exit at 24,576 on 42 against 137 (34%). Gaps of 7 or 8 points between 100 problems and 400 can be chance (Fisher's exact p is 0.26, 0.22 and 0.16), and by MATH's own levels the 100 are no easier (mean level 3.55 against 3.41). The primary difference is 14.0 points on the first 100 and 15.0 on the other 400, of which 399 had never been run here.
- One draw of 100 problems says little about this difference. An earlier pair of runs on the same 100, with another seed and a 4,000-token answer budget, had the primary pair at 50% and 46%. This run's draw of those problems has it at 56% and 42%.
- The base model's forced exit improves from 4,096 to 8,192 and not after that. The earlier draw had 39, 39, 43 and 46% at the four exit points, which looked like a steady rise. On 500 problems the four are 34.4, 38.6, 38.2 and 35.8%. The last is the exit made one token before its row stops. Its forced rows are right 143 times in 464, against 155 in 464 at 16,384, and its answers are the ones checked hardest above.
- The answer budget. The earlier 100-problem runs had 4,000 tokens. Cut to 2,000, the base model lost none of its right answers at any exit point, and the edited model lost 4 across the four. In this run, of the 464 rows the base model's exit at 24,576 forced, 65, 133, 140 and 143 were right within 500, 1,000, 1,500 and 2,000 tokens. Its other three exits gained 1, 2 and 0 in the last 500, and the edited model's four gained 1, 0, 6 and 2. A row that answers with thinking off from the start is slower and keeps gaining: cut at 500, 1,000 and 2,000 tokens those rows are right on 12.6%, 28.8% and 35.8%, and on 50.4% in the end.
- Not every answer that ran out of budget was looping. By the study's test, the base model's count is 99 of 318, 135 of 302, 134 of 298 and 128 of 314 at the four exit points. The edited model's is 9 of 311, 19 of 283, 28 of 202 and 7 of 122.

## Run it again

```
uv run python -m lab.run --name phase1 --max-hours 15 -- \
    zsh bench/phase1_study.zsh p1 docs/phase1-prereg.md 500 1 20261113
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 uv run python -m bench.phase1_checks --name p1 --plan docs/phase1-prereg.md

# the second grading, from the saved tokens (add the two HF_ variables as above)
uv run python -m bench.phase1 --tag p1-nothink --arm base --no-thinking \
    --plan docs/phase1-prereg.md --n 500 --run-seed 20261113 --report --grader study --save-as "p1-nothink, study grader"
uv run python -m bench.phase1 --tag p1-edited --arm edited --exits 4096,8192,16384,24576 --answer-tokens 2000 \
    --plan docs/phase1-prereg.md --n 500 --run-seed 20261113 --report --grader study --save-as "p1-edited, study grader"
uv run python -m bench.phase1 --tag p1-base --arm base --trunk-tokens 24577 --exits 4096,8192,16384,24576 --answer-tokens 2000 \
    --plan docs/phase1-prereg.md --n 500 --run-seed 20261113 --report --grader study --save-as "p1-base, study grader"
```

The first command needs the study's folder and about 12.5 hours. The ledgers it writes hold every token (124 MB for this run) and are not in the repository. `bench/results/phase1.json` has a verdict for every row under every condition, in entries `p1-nothink`, `p1-edited` and `p1-base`, so the comparisons can be made again without the ledgers by running the last command of `bench/phase1_study.zsh`. The second grading is in the entries whose names end in `, study grader`, and it needs the ledgers.
