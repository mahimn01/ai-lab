# ai-lab guide

A measured AI research lab for one Mac (Mac Studio, M1 Max, 32 GB). It lets you train
small models from scratch, fine-tune existing ones, edit weights and look inside
models, run local models fast, and run an automated experiment loop. Everything runs
on Apple's MLX framework.

Numbers live in two files, both built from real runs on this Mac:

- `BASELINE.md`: how fast this Mac runs, trains and fine-tunes models today.
- `SPEED-RESULTS.md`: each speed idea that was tested, with a keep or reject verdict.

All commands start from the project folder:

```
cd ~/Documents/Dev/ai-lab
```

## The rules this Mac taught us

1. **Check memory first.** A job that needs more memory than is free pushes the Mac
   into swap (using the disk as slow memory) and runs 5-10x slower. Run
   `uv run python -m lab.preflight` before anything heavy.
2. **Keep models on the internal disk.** Loading from the USB hard drive took 23
   minutes for a 13 GB model; from the internal disk it takes seconds.
3. **Switch the lab's GPU paths on with one line, `qmm.install(model)`.** With MLX's own
   GPU code on the M1, one decoding step costs the same for 8, 16 or 32 prompts, and 2-6
   prompts each re-read all the weights (`SPEED-RESULTS.md` section 1). The lab's kernel
   removes that for 2-18 prompts: 1.4-1.6x faster at 4 prompts, 2.1-2.6x at 8 (section
   9). The same line makes prompts 1.2-1.33x faster to read (section 11). 32 at once
   still gives the most tokens per second; 8 at once now gives over half of that with a
   quarter of the memory. 64 is no faster than 32. One chat at a time is unchanged.
4. **For inference, run the model in float16**: `float16.convert(model, tokenizer)`.
   MLX's GPU code converts the models' bfloat16 numbers in software; float16 is native.
   Prompts are read 1.4x faster and 32 prompts at once decode 1.16-1.31x faster; a
   64-question eval finished 1.29x sooner end to end. For one prompt the writing is
   unchanged, but with the lab's paths on as well a request with a 2,048-token prompt
   finishes 1.29 to 1.32x sooner. `convert` first checks on sample prompts that no
   value inside the model comes near float16's limit of 65,504, and refuses otherwise.
   Not for fine-tuning (section 16).
5. **Train from scratch with a small vocabulary and the default number format.** 8,192
   tokens instead of GPT-2's 50,257 made training 2.6x faster. The trainer's default
   format (`mixed16`: 32-bit weights, 16-bit arithmetic) trains 24% more tokens per
   second than 32-bit in 41% less memory and scores better in the same time (section 12).
6. **Fine-tune through `lab.finetune.lora`, not `mlx_lm.lora` directly.** It is mlx-lm's
   trainer with the lab's GPU paths, a fast training path for Qwen3.5-type models, and
   batches of similar-length examples. On short examples: Qwen3-4B 289 → 422 tokens/s,
   Qwen3.5-4B 86 → 338, Qwen3.5-9B 64 → 166. Qwen3.5-4B on full 512-token examples ran
   out of memory with mlx-lm alone and now runs at 345 tokens/s in 10 GB (section 13).
   The script decides by itself when to trade speed for memory.
7. **For several passes or a sweep on the same data, use `lab.finetune.cached`.** It
   computes the frozen bottom layers once: 1.2x faster per step when the last 16 layers
   are trained and up to 2.6x for the last 4, after a cache that takes one to two
   minutes to build (section 14).
8. **Pick the number of bits by the job.** At 32 prompts at once, 4-bit, 6-bit and 8-bit
   copies of a model run at the same speed, so use 8-bit (no measurable quality loss)
   when memory allows. For one chat, 4-bit is about 60% faster than 8-bit and costs
   about 6% in quality (perplexity).
9. **Put the shared part of the prompts first.** `generate_batch` reads a common opening
   (system prompt, worked examples, a document) once instead of once per prompt: 5x
   faster end to end on prompts that share 2,700 tokens.
10. **For one prompt whose answer copies earlier text, decode with lookahead.**
    `lab.infer.lookahead.generate` guesses the next tokens from the text so far and has
    the model check them in one pass: 1.75-2.5x faster when rewriting code or quoting a
    passage, no change on fresh text (section 10). It needs the lab kernel switched on.
11. **When looking inside a model, do not run layers you do not need.** `hooks.run_to`
    stops after a layer, `hooks.run_from` restarts from stored results, and
    `last_only=True` skips the output layer for all but the last position:
    collecting an early layer runs 6x faster than a full pass and a sweep from three
    quarters of the way up 5.3x faster (section 15).

12. **For many long generations at once, use the row engine (`lab.gen`).** mlx-lm's
    batch generator pads every row to the longest one and sorts the whole vocabulary at
    each step. The engine keeps each row's history in pages and picks tokens on the GPU,
    which also lets 48 rows run at once where 16 fitted: 100 problems at a 32,768-token
    cap took 56 to 59 minutes instead of 104, and at an 8,192 cap 12 instead of 18
    (section 19).
    - On this Mac use 48 rows at once and a 10 GB pool (the defaults): with 12 GB a run
      that also read long texts back in ran out of memory. The 56 and 59 minutes were
      measured with 12 GB. With 10 the planner adds under a minute, for a dozen rows
      set aside and read back.
    - One long run also gives the accuracy at every smaller cap, by cutting its rows
      short.
    - A condition that continues a row another way (a forced answer at a token count)
      runs as a branch in the same run (`--exits`): the edited model's rows and four
      such conditions took 68 minutes together. For a model whose rows nearly all run
      to the cap (the unedited one), keep slots back for the branches with
      `--keep-free 16`: 117 minutes for its five conditions.
    - `bench.plan_sim` says how long a run would take with other settings or more
      rows, from the row lengths of a run already made. On a Mac doing nothing else
      its clock has been within 1.5%. With other programs in use, runs took 3 to 7%
      longer than it says.
    - Near the end of a run the batch narrows, which changes the last digit of the
      scores of the rows still running. `--keep-width` switches that off, so a row
      comes out the same in any run of that width, unless it is set aside or read
      back in on the way. By the planner it costs a few percent of the time. It has
      been tried on a 12-problem run only.
    - Stopping rows that loop is possible but off by default, because it loses correct
      answers.

## Commands

| What you want | Command | What it changes |
|---|---|---|
| Is the Mac ready for a heavy job? | `uv run python -m lab.preflight` | nothing |
| Run any long job safely (no sleep, log, stop if memory runs out) | `uv run python -m lab.run --name myjob -- <command>` | one log under `runs/logs/` |
| Prepare the training text (once) | `uv run python -m lab.train.data` | writes `data/fineweb_edu/` |
| Train a small GPT for 5 minutes | `uv run python -m lab.train.train --name try1` | `runs/train/try1/result.json` |
| Fine-tune a model with LoRA | `uv run python -m lab.finetune.lora --model mlx-community/Qwen3-4B-Instruct-2507-4bit --iters 200` | adapter under `runs/lora/example` |
| The same, bottom layers computed once (sweeps, many passes) | `uv run python -m lab.finetune.cached --model mlx-community/Qwen3-4B-Instruct-2507-4bit --num-layers 8 --iters 200` | adapter under `runs/lora/cached`, cache under `runs/prefix-cache/` |
| Chat with a model | `uv run mlx_lm.chat --model mlx-community/Qwen3.5-9B-4bit` | nothing |
| Serve a model to other programs (lab paths on) | `uv run python -m lab.serve --model mlx-community/Qwen3.5-9B-4bit --port 8642 --decode-concurrency 8 --float16` | nothing |
| Compare the lab kernel with MLX's on a model | `uv run python -m bench.kernel_rows --model LABEL=PATH` | `bench/results/kernel_rows.json` |
| Check what float16 does for a model | `uv run python -m bench.float16 --model LABEL=PATH` | `bench/results/float16.json` |
| Time one whole request, four set-ups | `uv run python -m bench.request --model LABEL=PATH` | `bench/results/request.json` |
| Run the example eval | see the top of `examples/inspect_local.py` | one log under `runs/inspect-logs/` |
| Start the experiment loop | `uv run python -m lab.loop --rounds 10 --agent "claude -p --permission-mode acceptEdits"` | a `loop/...` git branch, `results.tsv` |
| Re-measure the baseline | `uv run python -m bench.inference --model LABEL=PATH`, then `uv run python -m bench.report` | `bench/results/`, `BASELINE.md` |
| Run a long-reasoning eval condition through the row engine | `uv run python -m bench.phase1 --arm edited --tag edited-32k` (needs the `neural-harness` study folder) | ledger under `runs/phase1/edited-32k/`, an entry in `bench/results/phase1.json` |
| Estimate how long a row-engine run will take (no GPU, 5 seconds; needs the ledgers of timing runs made on this Mac under `runs/phase1/`) | `uv run python -m bench.plan_sim` | `bench/results/plan_sim.json` |
| Check the row engine against mlx-lm on the real model | `uv run python -m bench.engine_checks` | `bench/results/engine_checks.json` |
| See what one decode step costs at long history | `uv run python -m bench.long_step --rows 16` | `bench/results/long_step.json` |
| Re-run the checks on the study's saved transcripts (no GPU) | `uv run python -m bench.transcripts` | `bench/results/transcripts.json` |
| See how early a comparison would be decided | `uv run python -m lab.stats.sequential` | nothing |
| Run the tests | `uv run pytest` (about 3 minutes; add `-m "not slow"` for the 35-second set) | nothing |

Every script starts with a plain-words header saying what it does, how to run it and
what it changes.

## Using the pieces from Python

Answer many prompts at once (the main speed lever for evals and sweeps):

```python
from mlx_lm import load
from lab.infer.batch import chat_tokens, generate_batch

model, tokenizer = load("mlx-community/Qwen3.5-4B-4bit")
prompts = [chat_tokens(tokenizer, q, system="Answer in one sentence.") for q in questions]
result = generate_batch(model, tokenizer, prompts, max_tokens=256, batch_size=32)
print(result.texts[0], result.generation_tps)
```

Switch on the lab's GPU paths (faster for 2-18 prompts at once and for reading prompts),
run in float16, and decode one prompt with lookahead:

```python
from lab.infer import float16
from lab.kernels import qmm
from lab.infer.lookahead import generate

report = float16.convert(model, tokenizer)            # checks on sample prompts first; float16.revert(model) undoes it
qmm.install(model)                                    # qmm.uninstall(model) switches back
result = generate_batch(model, tokenizer, prompts, max_tokens=256, batch_size=8)
one = generate(model, tokenizer, chat_tokens(tokenizer, question), max_tokens=400)
print(one.text, one.tokens_per_s, f"{one.accepted} of {one.guessed} guesses accepted")
```

Look inside a model, steer it, and edit its weights:

```python
import mlx.core as mx
from lab.edit import hooks, weights

logits, acts = hooks.capture(model, tokens)             # what every layer wrote
with hooks.steer(model, layer=12, vector=direction, scale=4.0):
    steered = model(tokens)                              # output with a direction added
with hooks.hook(model, 8, lambda out: mx.zeros_like(out)):
    ablated = model(tokens)                              # output with layer 8 silenced

# Skip the layers whose result is not needed or already known (SPEED-RESULTS.md section 15):
states = hooks.run_to(model, tokens, 20)                 # layer 20's output; later layers never run
for direction in directions:                             # try many changes, re-running only layers 21 up
    with hooks.steer(model, layer=24, vector=direction, scale=4.0):
        scores = hooks.run_from(model, states, 20, last_only=True)
hooks.save_activations(model, batches, layer=12, path="runs/acts-layer12.npy")  # a dataset of one layer's output

weights.list_weights(model, contains="layers.15.mlp")    # find a weight by name
log = weights.EditLog(model="Qwen3.5-0.8B")
weights.apply_delta(model, "language_model.model.layers.15.mlp.down_proj", delta, log=log)
a, b, info = weights.delta_to_lora(delta)                # the same change as a tiny LoRA file
undo = weights.attach_lora(model, path, a, b, scale=1.0) # try it without touching the weights
```

`apply_delta` needs a full-precision model. `attach_lora` and all the hooks also work
on quantized (4-bit) models.

Run many long generations at once, each row with its own settings (section 19; design
in `docs/design-row-engine.md`). Works for the Qwen3.5 text models:

```python
from lab.gen.engine import Engine, Row, seed_for
from lab.gen.ledger import Ledger

engine = Engine(model, slots=48, pool_gb=10, max_tokens=8192, decode=tokenizer.decode)
rows = [Row(prompt=p, max_tokens=8192, seed=seed_for("run-1", i), stop_tokens=(eos_id,), meta={"i": i})
        for i, p in enumerate(prompts)]                    # temperature, top_k, top_p, presence_penalty can differ per row
records = engine.run(rows, Ledger("runs/my-eval"))        # each finished row is on disk at once; run again to continue
```

A row may carry its own weight edit (`edit=`, with `edits=` given to the engine), tokens
it must write first (`forced=`), a loop-stopping rule (`stop_after_loop=`, off by
default because it changes grades) and branch points (`branch_at=` with a `brancher`
function passed to `run`). A branch shares its trunk's history and waits for a slot if
none is free. `keep_free=` on `run` holds slots back for branches. `lab/stats/sequential.py`
has a paired test that stays valid when it is looked at after every problem.

## What fits on this Mac

MLX may use about 25 GB of the 32 GB for models, and other apps need their share.

- **Running models:** up to about 27B parameters at 4-bit (14 GB of weights). 4B-9B
  models leave plenty of room for batching. The 27B model runs 1-8 prompts at once
  (21.6 GB at 8); 32 at once ran out of memory. With the lab kernel on, 8 at once
  gives it about twice its best speed without (section 9), and in float16 it reads
  prompts 1.4x faster (154 against 109 tokens/s, section 16).
- **Training from scratch:** 14M-91M parameter GPTs. The 91M size needs 16-bit
  arithmetic (the default) or a batch of 16 to stay under 12 GB.
- **Fine-tuning (LoRA, batch 4, short examples, through `lab.finetune.lora`):**
  Qwen3-4B at 422 tokens/s in 8 GB; Qwen3.5-0.8B at 1,172 in 4 GB; Qwen3.5-4B at 338 in
  11 GB; Qwen3.5-9B at 166 in 14 GB. Full 512-token examples: Qwen3-4B at 516 tokens/s
  in 14 GB, Qwen3.5-4B at 345 in 10 GB.

## Free compute for confirming a result on a real GPU

A result found here is "best on this Mac". Before claiming more, confirm it once on a
standard NVIDIA GPU. Free routes, checked 2026-11-11:

- **Digital Research Alliance of Canada / SciNet.** Free national clusters with real
  GPUs. A UofT faculty member must sponsor the account. This is the most valuable
  option: ask a professor whose course or lab fits the project.
- **Modal.** $30 of free credit each month on the Starter plan (about 7 hours of one
  H100 at the price listed that day).
- **Kaggle notebooks.** Free GPU sessions; the `kaggle` command-line tool is already
  installed. The weekly quota was not verified, so check it on Kaggle's site.
- **Google Colab free tier.** Works for short runs; limits are not published.

The trainer is MLX-only. A PyTorch twin of the model already exists in
`bench/training.py` and is the starting point for a GPU version when one is needed.

## Layout

```
lab/preflight.py      memory, swap, disk and heat check
lab/run.py            watchdog job runner
lab/train/            data.py (text -> tokens), model.py (the GPT), train.py (fixed-time training),
                      program.md (instructions the loop gives the agent)
lab/finetune/         lora.py (LoRA / QLoRA fine-tuning), cached.py (the same with the frozen bottom layers
                      computed once), gated_delta.py (the fast training path for Qwen3.5's recurrent layers)
lab/edit/             weights.py (read, change, checksum, LoRA conversion), hooks.py (capture, hook, steer,
                      stop after a layer, resume from a layer)
lab/infer/batch.py    batched generation with shared-prefix reuse
lab/infer/float16.py  switch a bfloat16 model to float16 after a safety check
lab/infer/lookahead.py  one prompt, several tokens per pass by guessing ahead
lab/kernels/qmm.py    the lab's GPU paths for quantized layers: a kernel for 2-18 rows, decode-then-dense for 256+
lab/gen/              the row engine for many long generations: engine.py (rows, slots, branches), kv.py (attention
                      history in pages and its GPU kernel), sampler.py, loops.py (loop marks), ledger.py (restartable runs)
lab/stats/            sequential.py (a paired test and an accuracy band that may be looked at after every problem)
lab/serve.py          mlx-lm's server with those paths switched on
lab/loop.py           the automated experiment loop
bench/                benchmark and experiment scripts; results in bench/results/*.json
docs/                 design-row-engine.md (the design of lab/gen, written before it was built, with its corrections)
examples/             an Inspect AI eval against the local server
tests/                fast tests (seconds) and slow tests (real model, real training)
```

## What is deliberately left alone

`neural-harness`, `stockbench-inspect`, the StockBench Ollama server and its models on
the external drive, and the old conda environments. Nothing outside this folder and
the Hugging Face cache was changed. No system setting was changed.
