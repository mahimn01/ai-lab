# ai-lab

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
3. **Switch the lab kernel on, then batch.** With MLX's own GPU code on the M1, one
   decoding step costs the same for 8, 16 or 32 prompts, and 2-6 prompts each re-read
   all the weights (`SPEED-RESULTS.md` section 1). The lab's kernel removes that for
   2-18 prompts with one line, `qmm.install(model)`: 1.4-1.6x faster at 4 prompts,
   2.1-2.6x at 8 (section 9). 32 at once still gives the most tokens per second; 8 at
   once now gives over half of that with a quarter of the memory. 64 is no faster than
   32. One chat at a time is unchanged.
4. **Small vocabulary for from-scratch training.** 8,192 tokens instead of GPT-2's
   50,257 made training 2.6x faster and cut memory by 40% for the same model body.
5. **Fine-tune Qwen3.5-type models with gradient checkpointing** (the lab turns it on
   by itself). Without it a 0.8B model needed 18 GB; with it, 4 GB at the same speed.
6. **Pick the number of bits by the job.** At 32 prompts at once, 4-bit, 6-bit and 8-bit
   copies of a model run at the same speed, so use 8-bit (no measurable quality loss)
   when memory allows. For one chat, 4-bit is about 60% faster than 8-bit and costs
   about 6% in quality (perplexity).
7. **Put the shared part of the prompts first.** `generate_batch` reads a common opening
   (system prompt, worked examples, a document) once instead of once per prompt: 5x
   faster end to end on prompts that share 2,700 tokens.
8. **For one prompt whose answer copies earlier text, decode with lookahead.**
   `lab.infer.lookahead.generate` guesses the next tokens from the text so far and has
   the model check them in one pass: 1.75-2.5x faster when rewriting code or quoting a
   passage, no change on fresh text (section 10). It needs the lab kernel switched on.

## Commands

| What you want | Command | What it changes |
|---|---|---|
| Is the Mac ready for a heavy job? | `uv run python -m lab.preflight` | nothing |
| Run any long job safely (no sleep, log, stop if memory runs out) | `uv run python -m lab.run --name myjob -- <command>` | one log under `runs/logs/` |
| Prepare the training text (once) | `uv run python -m lab.train.data` | writes `data/fineweb_edu/` |
| Train a small GPT for 5 minutes | `uv run python -m lab.train.train --name try1` | `runs/train/try1/result.json` |
| Fine-tune a model with LoRA | `uv run python -m lab.finetune.lora --model mlx-community/Qwen3-4B-Instruct-2507-4bit --iters 200` | adapter under `runs/lora/example` |
| Chat with a model | `uv run mlx_lm.chat --model mlx-community/Qwen3.5-9B-4bit` | nothing |
| Serve a model to other programs (lab kernel on) | `uv run python -m lab.serve --model mlx-community/Qwen3.5-9B-4bit --port 8642 --decode-concurrency 8` | nothing |
| Compare the lab kernel with MLX's on a model | `uv run python -m bench.kernel_rows --model LABEL=PATH` | `bench/results/kernel_rows.json` |
| Run the example eval | see the top of `examples/inspect_local.py` | one log under `runs/inspect-logs/` |
| Start the experiment loop | `uv run python -m lab.loop --rounds 10 --agent "claude -p --permission-mode acceptEdits"` | a `loop/...` git branch, `results.tsv` |
| Re-measure the baseline | `uv run python -m bench.inference --model LABEL=PATH`, then `uv run python -m bench.report` | `bench/results/`, `BASELINE.md` |
| Run the tests | `uv run pytest` (add `-m "not slow"` for the 3-second set) | nothing |

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

Switch on the lab's GPU kernel (faster for 2-18 prompts at once) and decode one prompt
with lookahead:

```python
from lab.kernels import qmm
from lab.infer.lookahead import generate

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

weights.list_weights(model, contains="layers.15.mlp")    # find a weight by name
log = weights.EditLog(model="Qwen3.5-0.8B")
weights.apply_delta(model, "language_model.model.layers.15.mlp.down_proj", delta, log=log)
a, b, info = weights.delta_to_lora(delta)                # the same change as a tiny LoRA file
undo = weights.attach_lora(model, path, a, b, scale=1.0) # try it without touching the weights
```

`apply_delta` needs a full-precision model. `attach_lora` and all the hooks also work
on quantized (4-bit) models.

## What fits on this Mac

MLX may use about 25 GB of the 32 GB for models, and other apps need their share.

- **Running models:** up to about 27B parameters at 4-bit (14 GB of weights). 4B-9B
  models leave plenty of room for batching. The 27B model runs 1-8 prompts at once
  (21.6 GB at 8); 32 at once ran out of memory. With the lab kernel on, 8 at once
  gives it about twice its best speed without (section 9).
- **Training from scratch:** 14M-91M parameter GPTs. The 91M size needs 16-bit weights
  or a batch of 16 to stay under 12 GB.
- **Fine-tuning (LoRA, batch 4, 512 tokens):** Qwen3-4B at 313 tokens/s in 5 GB;
  Qwen3.5-4B at 88 tokens/s in 9 GB; Qwen3.5-9B at 56 tokens/s in 12 GB. Qwen3.5-type
  models fine-tune 3.5-7x slower per token than attention-only models of the same size.

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
lab/finetune/lora.py  LoRA / QLoRA fine-tuning
lab/edit/             weights.py (read, change, checksum, LoRA conversion), hooks.py (capture, hook, steer)
lab/infer/batch.py    batched generation with shared-prefix reuse
lab/infer/lookahead.py  one prompt, several tokens per pass by guessing ahead
lab/kernels/qmm.py    the lab's GPU kernel for 2-18 prompts at once
lab/serve.py          mlx-lm's server with that kernel switched on
lab/loop.py           the automated experiment loop
bench/                benchmark and experiment scripts; results in bench/results/*.json
examples/             an Inspect AI eval against the local server
tests/                fast tests (seconds) and slow tests (real model, real training)
```

## What is deliberately left alone

`neural-harness`, `stockbench-inspect`, the StockBench Ollama server and its models on
the external drive, and the old conda environments. Nothing outside this folder and
the Hugging Face cache was changed. No system setting was changed.
