"""Train a small GPT from scratch for a fixed amount of time, then score it.

What it does
  Trains the model in lab/train/model.py on the FineWeb-Edu tokens for exactly
  --budget-s seconds of training time (the first step, where MLX compiles the GPU
  code, is not counted), then measures the loss on the held-out validation tokens.

Why a fixed time budget
  Every run gets the same wall-clock time on the same Mac, so any two runs are
  directly comparable: the better idea is the one with the lower validation score.
  This is the protocol of Karpathy's autoresearch. A result found this way is "best on
  this Mac in this time"; it may not hold on bigger hardware.

The score
  val_bpb = validation bits per byte of text (lower is better). It is the average loss
  per token, converted so it does not depend on which tokenizer is used.

Run:     uv run python -m lab.train.train --name baseline [--size s] [--budget-s 300]
         (first run `uv run python -m lab.train.data` once to prepare the tokens)
Changes: writes runs/train/<name>/result.json and prints one RESULT line. With
         --save it also writes runs/train/<name>/model.safetensors. Nothing else.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from mlx.utils import tree_flatten, tree_map

from lab.train.model import GPT, SIZES, n_params

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "fineweb_edu"
DTYPES = {"float32": mx.float32, "bfloat16": mx.bfloat16, "float16": mx.float16}
# "mixed16": 32-bit weights and optimizer, 16-bit arithmetic in the model. The loss is multiplied by
# GRAD_SCALE per token before the backward pass so that the 16-bit gradients stay inside the range
# 16-bit floats can hold (their smallest normal value is 6e-5), and divided out again afterwards.
MIXED = {"mixed16": mx.float16, "mixedbf16": mx.bfloat16}
GRAD_SCALE = 8.0
SCALED_TOKENS = 32 * 512  # the batch the scale was measured at; bigger batches are not scaled further
WARMUP_FRACTION = 0.03  # of the time budget
VAL_BATCHES = 20


def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", default="run")
    ap.add_argument("--size", choices=SIZES, default="s")
    ap.add_argument("--budget-s", type=float, default=300.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--optimizer", choices=["muon", "adamw"], default="muon",
                    help="muon = Muon on the block matrices + AdamW on the rest; adamw = AdamW on everything")
    ap.add_argument("--lr", type=float, default=3e-3, help="AdamW learning rate")
    # 0.04 beat 0.02 by 0.05 val_bpb (noise 0.007) and 0.08 was worse: SPEED-RESULTS.md section 5.
    ap.add_argument("--muon-lr", type=float, default=0.04)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    # mixed16 trained 24% more tokens in the same time than float32 and scored 0.05 val_bpb better
    # (two seeds each, noise 0.007): SPEED-RESULTS.md section 12.
    ap.add_argument("--dtype", choices=[*DTYPES, *MIXED], default="mixed16",
                    help="number format; mixed16 (default) = 32-bit weights with 16-bit arithmetic")
    ap.add_argument("--no-compile", action="store_true", help="run the step without mx.compile (for comparison)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", action="store_true", help="also save the trained weights")
    return ap.parse_args()


def make_optimizers(args: argparse.Namespace) -> tuple[optim.Optimizer, list[tuple[optim.Optimizer, float]]]:
    """Return the optimizer plus (sub-optimizer, peak learning rate) pairs for the schedule."""
    adamw = optim.AdamW(learning_rate=args.lr, betas=[0.9, 0.95], weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        return adamw, [(adamw, args.lr)]
    # Muon is made for 2-D weight matrices inside the blocks. The embedding table and
    # the 1-D normalisation weights go to AdamW, as the MLX docs recommend.
    muon = optim.Muon(learning_rate=args.muon_lr, weight_decay=args.weight_decay)
    is_block_matrix = lambda path, w: w.ndim == 2 and "wte" not in path  # noqa: E731
    return optim.MultiOptimizer([muon, adamw], [is_block_matrix]), [(muon, args.muon_lr), (adamw, args.lr)]


def lr_scale(progress: float) -> float:
    """Learning-rate multiplier by fraction of the time budget used: short ramp up, then cosine down to 10%."""
    if progress < WARMUP_FRACTION:
        return progress / WARMUP_FRACTION
    p = (progress - WARMUP_FRACTION) / (1 - WARMUP_FRACTION)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


def main() -> None:
    args = get_args()
    meta = json.loads((DATA_DIR / "meta.json").read_text())
    train = np.memmap(DATA_DIR / "train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(DATA_DIR / "val.bin", dtype=np.uint16, mode="r")
    B, T = args.batch_size, args.seq_len
    rng = np.random.default_rng(args.seed)
    mx.random.seed(args.seed)

    model = GPT(dataclasses.replace(SIZES[args.size], vocab_size=meta["vocab_size"]))
    model.set_dtype(DTYPES.get(args.dtype, mx.float32))
    model.compute_dtype = MIXED.get(args.dtype)
    mx.eval(model.parameters())
    optimizer, schedule = make_optimizers(args)
    # Per token the error signal is multiplied by GRAD_SCALE, so that small values survive 16 bits. A weight's
    # gradient is a sum over the batch and grows with it, so beyond the measured batch size the scale stops growing.
    grad_scale = GRAD_SCALE * min(B * T, SCALED_TOKENS) if args.dtype in MIXED else 1.0

    def loss_fn(model: nn.Module, x: mx.array, y: mx.array) -> tuple[mx.array, mx.array]:
        # The loss itself is always computed in 32-bit: adding up 16,384 losses overflows a 16-bit float.
        loss = mx.fast.cross_entropy(model(x).astype(mx.float32), y).mean()
        return loss * grad_scale, loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def step(x: mx.array, y: mx.array) -> tuple[mx.array, mx.array]:
        """One training step. Returns the loss and whether every gradient value fitted in 16 bits."""
        (_, loss), grads = loss_and_grad(model, x, y)
        fitted = mx.array(True)
        if grad_scale != 1.0:  # undo the scaling; a value that overflowed 16 bits counts as zero, and the step is counted
            fitted = mx.stack([mx.isfinite(g).all() for _, g in tree_flatten(grads)]).all()
            grads = tree_map(lambda g: mx.nan_to_num(g * (1.0 / grad_scale), nan=0.0, posinf=0.0, neginf=0.0), grads)
        optimizer.update(model, grads)
        return loss, fitted

    def eval_loss(x: mx.array, y: mx.array) -> mx.array:
        return loss_fn(model, x, y)[1]

    if not args.no_compile:
        # Compiling turns the whole step (forward, backward, weight update) into one GPU
        # program. The model and optimizer state must be declared as inputs and outputs.
        state = [model.state, optimizer.state]
        step = partial(mx.compile, inputs=state, outputs=state)(step)
        eval_loss = partial(mx.compile, inputs=[model.state])(eval_loss)

    def batch(data: np.memmap, starts: np.ndarray) -> tuple[mx.array, mx.array]:
        rows = np.stack([data[s:s + T + 1] for s in starts]).astype(np.int32)
        return mx.array(rows[:, :-1]), mx.array(rows[:, 1:])

    def train_batch() -> tuple[mx.array, mx.array]:
        return batch(train, rng.integers(0, len(train) - T - 1, B))

    # First step: compiles the GPU code. Not counted in the budget or the speed.
    mx.eval(step(*train_batch()), model.parameters(), optimizer.state)
    mx.reset_peak_memory()

    steps, overflowed, last_log = 0, 0, 0.0
    t0 = time.perf_counter()
    while (elapsed := time.perf_counter() - t0) < args.budget_s:
        scale = lr_scale(elapsed / args.budget_s)
        for opt, peak in schedule:
            opt.learning_rate = peak * scale
        loss, fitted = step(*train_batch())
        mx.eval(loss, fitted, model.parameters(), optimizer.state)
        steps += 1
        overflowed += not fitted.item()
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"training loss became {loss.item()} at step {steps}: the run diverged")
        if elapsed - last_log >= 20:
            last_log = elapsed
            print(f"  {elapsed:5.0f}s step {steps:6d} loss {loss.item():.4f} "
                  f"{steps * B * T / elapsed:,.0f} tok/s", flush=True)
    train_s = time.perf_counter() - t0
    peak_gb = mx.get_peak_memory() / 2**30  # of training; the scoring below is not part of it
    if overflowed:
        print(f"  note: in {overflowed} of {steps} steps some gradient values were too large for 16 bits and were left out. "
              "If that is more than a few, use a smaller batch or --dtype bfloat16.", flush=True)

    # Score on the first VAL_BATCHES * B windows of the validation tokens (always the same ones).
    model.eval()
    if args.dtype in MIXED:
        model.compute_dtype = None  # score in the weights' own 32-bit format, as the float32 runs do
    val_losses = []
    for i in range(VAL_BATCHES):
        starts = (np.arange(B) + i * B) * T
        val_losses.append(eval_loss(*batch(val, starts)).item())
    val_loss = float(np.mean(val_losses))

    result = {
        "name": args.name,
        "val_bpb": val_loss / math.log(2) / meta["val_bytes_per_token"],
        "val_loss": val_loss,
        "train_tokens_per_s": steps * B * T / train_s,
        "steps": steps,
        "train_s": train_s,
        "params": n_params(model),
        "peak_memory_gb": peak_gb,
        "overflowed_steps": overflowed,
        "config": {k: v for k, v in vars(args).items() if k not in ("name", "save")},
    }
    out_dir = ROOT / "runs" / "train" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.save:
        model.save_weights(str(out_dir / "model.safetensors"))
    print("RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
