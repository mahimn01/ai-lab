# Design: a row engine for long-reasoning experiments

Written 2026-11-12, before any of it was built. Section 9 lists what was predicted, so the
results can be held against it.

Words used here:

- **Token**: a piece of a word. The 32,768 "cap" is the most tokens one answer may use.
- **Row**: one generation, from prompt to stop. One problem, one sample, one set of settings.
- **Condition**: a set of rows that share settings (model, edit, cap, sampling).
- **Step**: one pass of the model that writes one new token for every row in the batch.
- **Attention history**: what the model's 6 attention layers store for every earlier
  token of a row, about 12 KB per token (the "KV cache"). A 32,768-token row holds 0.4 GB.
- **Trunk and branch**: a trunk is a full-length generation. A branch starts from a point
  on a trunk and continues differently (for example, forced to stop thinking and answer).

## 1. The problem

The study in `neural-harness` asks whether a small model (Qwen3.5-0.8B) can be repaired
from looping by a weight edit. Its evaluations are long: up to 32,768 tokens of thinking
per problem. Phase 0 needed 1.74 hours for one condition of 100 problems at that cap
(2.1 million tokens at 337 tokens a second). The recommended Phase 1 has about ten such
conditions on 500 problems with 4 samples each. At Phase 0's speed one such condition
takes about 36 hours, so the study was out of reach on this Mac.

## 2. What was measured before designing

All on the study's own model (dense bfloat16 weights, 1.6 GB), with mlx-lm 0.32.2, on
the M1 Max. Scripts and raw output are exploration, not part of the repo.

**Where one step goes.** The model pass alone, 16 rows:

| History length per row | Step | Tokens/s |
|---|---|---|
| 64 tokens | 22.4 ms | 715 |
| 4,096 | 25.5 ms | 628 |
| 16,384 | 35.1 ms | 456 |
| 32,768 | 48.9 ms | 327 |

1. **Most of a step is a fixed cost that does not grow with the number of rows.** At a
   short history the step takes 22 ms for 16 rows, 25 ms for 32 and 29 ms for 64. The
   arithmetic for 16 rows needs about 3 ms. The rest is the cost of issuing about 1,100
   small GPU operations, about 20 millionths of a second each. Compiling the step
   (`mx.compile`) removed 4% of it. No single operation stands out.
2. **The other cost is reading the attention history**, and it is already near the
   Mac's memory speed: MLX's attention kernel reads 230 to 295 GB/s against about 375
   GB/s measured for the chip. It costs about 42 billionths of a second per row per token
   of history per step. A hand-regrouped version was 2.5x slower.
3. **mlx-lm pads every row's history to the longest row in the batch.** With rows
   starting and ending at different times, the longest is nearly always close to 32,768,
   so every row pays for a full history at every step. Phase 0's 337 tokens a second is
   exactly the 16-row, 32,768-token line of the table. The rows' real average history
   was about 12,600 tokens.
4. **Picking the next token costs as much as a third of the model pass.** The study's
   sampler finds the top 20 of 248,320 scores with a full sort: 5.0 ms for 32 rows. An
   exact method that sorts only the 20 best blocks of 256 takes 0.9 ms. The presence
   penalty runs as a Python loop over rows.
5. **float16 gains little here**: 5 to 8% (20.9 against 22.4 ms). This model is small and
   dense, so the fixed cost dominates and the number format hardly matters.

**What the saved transcripts say** (3,432 transcripts from Phase 0 and SD1, no GPU):

6. **Loops are soft, so guessing ahead fails.** Using the text's own earlier
   repetitions as a guess for the next tokens, 45% of the 32k run's tokens could be
   guessed, but only 4.7 at a time, and 63% of guesses broke early. That is 1.56 tokens
   per model pass, not enough to pay for a second lane. The idea is rejected.
7. **Rows escape long loops.** With the study's own loop test (a 512-token window whose
   text compresses below 0.2), stopping a row after 512 looping tokens would save 28% of
   the 32k run's tokens and lose 4 of its 54 correct answers. Stopping after 4,480 still
   loses 1. Only at 8,576 looping tokens is none lost, and that saves 6.5%.
8. **A shorter cap does not rank edits the same way.** Loops form late: at a 2,048 cap
   the base model shows 2.5 looping rows of 64, at 8,192 it shows 40. The 38 saved edit
   variants rank differently at 2,048 (rank agreement 0.39) and 4,096 (0.71) than at
   8,192. The idea is rejected.

**What follows.** A faithful 32,768-token run cannot be made ten times faster on this
Mac: the history must be read for every token, at memory speed, and the fixed cost per
step can only be shared among more rows. The estimate for a faithful run is about 3x.
The larger gain is in the study design: most Phase 1 conditions are prefixes or branches
of two trunks, and they should cost prefixes and branches, not ten full runs.

## 3. Design

One engine, `lab/gen/`, that treats the **row** as the unit of work. A condition is a
list of rows. Rows from different conditions share the GPU.

### 3.1 Rows carry everything that differs

Each row has its own prompt, cap, stop tokens, sampling settings (temperature, top-k,
top-p, presence penalty), random stream, optional weight edit, and rules (stop on a
loop, branch at a token count). So a sweep over penalties or edit strengths is one
stream of rows, with no model reloads and no half-empty batches.

**Each row draws from its own random stream**, made from a hash of (run seed, problem,
sample, branch). mlx-lm draws once for the whole batch, so a row's text depends on
which rows share the batch. Here a row's draws do not, which is what lets a run be
resumed, a row be restarted, and a branch be compared with its trunk.

### 3.2 Attention history in pages

The history lives in a pool of fixed pages of 256 tokens. A row owns a list of pages
plus a small buffer for its newest tokens (up to 255). When the buffer fills it becomes
a page. Finished rows give their pages back.

- **No padding.** A custom attention kernel walks each row's own pages and buffer, so a
  step reads what the rows really hold. It follows the method of MLX's single-query
  attention kernel (running maximum and sum, 32-bit accumulation) with a page table
  added. This is the part with the most risk (section 8).
- **Memory follows the real total, not rows x cap.** 16 padded rows at a 32,768 cap
  need 6.4 GB whatever they hold. In pages, 48 to 64 rows fit in about 12 GB because
  rows are at different lengths. The engine admits a new row only if the pool has room,
  and if the pool ever runs short it sets the youngest row aside and replays it later
  (its tokens are saved, and re-reading them is far faster than writing them).
- **Pages are never changed once full**, so two rows can share them.

### 3.3 Branches that share their trunk

A branch is a new row that points at its trunk's pages and copies only the small
buffer and the recurrent layers' state (about 20 MB). It costs nothing to start.

This is what makes the Phase 1 curves cheap:

- **Accuracy at every smaller cap is free.** The grader reads the last boxed answer in
  whatever text exists, so "the model at a 4,096 cap" is the first 4,096 tokens of the
  32,768 trunk, graded. One trunk gives the whole accuracy-against-tokens curve.
- **A forced exit at 4k, 8k, 16k and 24k is four short branches.** When the trunk
  passes each point with its thinking still open, a branch takes the study's
  25-token exit phrase and writes its answer. The trunk carries on.

Branches can also be made later from a saved trunk by re-reading its tokens.

### 3.4 Sampling on the GPU, same distribution

The study's rule, in its order: presence penalty on the raw scores for every token the
row has written, divide by the temperature, keep the top 20, keep tokens while the
probability before them is below 0.95, draw. The engine does this for all rows at once
with per-row settings. The top 20 come from the block method of finding 4 (exact: the
20 largest values must lie in the 20 blocks with the largest maxima). The draw uses the
row's own random number against the cumulative probabilities.

### 3.5 Several edits in one batch

The study's edits are rank-1 changes to two matrices. Kept unmerged, an edit is two
vectors per matrix, so each row can carry its own (or none): the batch computes the
shared matrix product once and adds each row's small term. Base and edited rows, or
38 edit variants, run in one stream. (Phase 0's "edited" arm used the merged published
tensors. The benchmark uses those, unchanged, for the like-for-like run.)

### 3.6 Watching loops live, stopping only by rule

The engine applies the study's loop test to every row every 128 tokens and records
when each loop starts and ends. That is a measurement and changes nothing.

Stopping is a separate, opt-in rule: stop after N looping tokens in a row. Finding 7
shows it changes grades, so:

- the default is off, and the full-budget run stays the number of record;
- a run that uses it also runs a random share of its rows to the full budget (chosen by
  a hash of the row, not by its content), which measures how many grades the rule
  changed in that very run;
- the calibration table (tokens saved against grades changed, for each N) is part of
  the results.

### 3.7 Ledger and resume

Every finished row is appended to a ledger: its key (a hash of everything that defines
it), its token ids, why it stopped, loop marks, and timing. Token ids are kept (Phase 0
kept only text, and re-tokenizing was inexact). A run that is stopped or crashes
restarts by skipping rows already in the ledger. A plan file written before a run
(conditions, sample sizes, stopping rules, the analysis) is hashed into the ledger, so
a result can be shown to match what was planned.

### 3.8 Stopping a comparison when it is decided

`lab/stats/sequential.py`: a test for paired right/wrong outcomes that stays valid when
you look after every problem (a mixture likelihood ratio on the discordant pairs), and
a confidence band for an accuracy that holds at every sample size at once. Problems
are analysed in a fixed random order set before the run, because short problems finish
first and would bias a look. A large effect (looping 66% against 17%) is decided in a
few dozen problems. A small one runs to the planned size.

### 3.9 Moving to a GPU cluster later

Nothing above the engine depends on the Mac: row definitions, the ledger, the loop
test, the stopping rules, the statistics and the plan file. The engine sits behind a
small interface (admit rows, step, collect finished rows). On NVIDIA GPUs the same
interface can be served by vLLM, which already has paged history and continuous
batching: a row maps to one request (seed, temperature, top-k, top-p, presence penalty,
token cap, stop tokens, adapter), and a branch maps to a request that continues a
prefix. That backend is not built here and cannot be tested without a cluster account.
Its sampler order would need the same distribution test as section 7 before any number
from it is trusted.

## 4. What it is not

- Not faster for one chat. One row alone still pays the whole fixed cost.
- Not a general server. It supports the Qwen3.5 text models through mlx-lm's own layer
  code. Another architecture needs its attention layer wired to the page pool.
- Not a shortcut on the science. Section 2's findings 6 to 8 are three shortcuts that
  were checked and turned down.

## 5. Ideas from the brief, and what happens to each

| Idea | Decision | Why |
|---|---|---|
| Continuous batching, paged history | build | findings 1 and 3: the old run paid for padding and for too few rows |
| Loop detection and early stop | build as an opt-in rule with an audit | finding 7: it changes grades |
| Sharing history across conditions | build (branches) | most Phase 1 conditions are prefixes or branches of two trunks |
| Sequential testing | build | cheap, and exact |
| Activation caching across sweeps | exists (`hooks.run_to`, `run_from`, `save_activations`) | section 15 of `SPEED-RESULTS.md` |
| Speculative decoding, new angle | reject | finding 6: 1.56 tokens per pass |
| Many edits in one batch | build | rank-1 edits are two vectors per row |
| Shorter-cap proxy | reject | finding 8 |
| Scheduler with checkpoints | build (ledger, resume, replay) | Phase 0 lost a whole arm on a crash |
| Pre-registration and ledger | build | |
| Path to a cluster | design the seam, do not build the backend | cannot be tested here |
| float16 | optional, off by default | finding 5: 5 to 8% here, and it needs a headroom check |

## 6. Build order

1. Sampler, with its distribution test.
2. Page pool and attention kernel, tested against MLX's own attention on the same data.
3. Engine: slots, admission, the model pass, stop rules. Tested against mlx-lm.
4. Ledger and resume.
5. Branches and per-row edits.
6. Loop marks and the stopping rule.
7. Sequential statistics.
8. The Phase 1 benchmark.

## 7. How each speed-up is checked

Correctness comes first: nothing below is reported as a speed-up without its check.

| Piece | Check |
|---|---|
| Attention kernel | same output as `mx.fast.scaled_dot_product_attention` on the same keys and values, for rows of different lengths, to rounding |
| Model pass | the engine's scores equal mlx-lm's own scores at every step of real prompts, to rounding; with no randomness (always the top token) the engine and mlx-lm write the same tokens |
| Batch independence | a row's tokens are the same alone and in a full batch |
| Sampler | for the same scores, the engine's probabilities over tokens equal the study's sampler's; drawn frequencies match those probabilities |
| Branch | a branch's scores equal those of a fresh row given the same tokens |
| Replay | a row set aside and replayed continues with the same scores |
| Per-row edit | scores equal those of the model with the edit attached the study's way |
| Whole condition | accuracy, cap-hit share, loop share and the token-count distribution of a 100-problem run against Phase 0's run of the same condition (samples differ, so this is a comparison of distributions) |
| Stopping rule | grades changed, counted on full-budget transcripts |
| Sequential test | false-alarm rate under repeated looks, by simulation |

## 8. Risks

- **The attention kernel may not reach MLX's speed.** Fallback: call MLX's attention
  once per row on that row's own history (correct, about 8 to 10 ms more per step at 64
  rows).
- **In-place writes may copy.** MLX updates an array in place only when nothing else
  holds it. The pool is written in its own evaluation, after everything that read it,
  and the test checks that step time does not grow with pool size.
- **Bit-level differences.** bfloat16 has about 3 digits. Two correct ways of computing
  the same score differ in the last digit, and sampled text then diverges. This is why
  the whole-condition check compares distributions, as Phase 0 itself had to.
- **Memory.** The pool is sized up front (default 12 GB) and never grows.

## 9. Predictions, written before building

For the Phase 0 condition (edited model, thinking on, 32,768 cap, 100 problems, about
2.1 million tokens):

| | Phase 0 | Predicted for the engine |
|---|---|---|
| Wall-clock | 104 min | 35 to 45 min |
| Tokens per second, whole run | 337 | 800 to 1,000 |
| Rows at once | 16 | 48 to 64 |
| Where the time goes | 16 rows padded to 32,768 | about half fixed cost, half history reads |

For 100 problems the run cannot be shorter than its longest row: 32,768 steps. At about
25 ms a step that floor is 14 minutes, and the tail of a run (few rows left) is slow in
tokens per second. A larger run does not have this problem.

Four forced-exit conditions as branches: predicted to add under 15% to the trunk's time.

If the engine misses these numbers, section 19 of `SPEED-RESULTS.md` will say so.
