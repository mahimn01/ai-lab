"""Training baseline: how many tokens per second can this Mac train a small GPT on?

What it does
  Trains the same GPT shape on random tokens for a fixed number of seconds and reports
  tokens/s, in two frameworks:
    MLX      (lab/train/model.py)  step compiled or not; weights in 32-bit or 16-bit
    PyTorch  (a twin model below, on the Mac GPU via "MPS")  32-bit, or 16-bit autocast
  Both use plain AdamW, batch 32 x 512 tokens, and wait for the GPU after every step,
  so the two frameworks are timed the same way. Random tokens are fine here: this
  measures speed only, not learning.

Run:     uv run --group torch python -m bench.training [--sizes s,m,gpt2] [--seconds 20]
Changes: writes bench/results/training.json. Nothing else.
"""

from __future__ import annotations

import argparse
import gc
import math
import time
from functools import partial

import numpy as np

from bench.common import environment, free_gpu_memory, save
from lab.train.model import SIZES, GPTConfig

B, T = 32, 512
WARMUP_STEPS = 3


def bench_mlx(cfg: GPTConfig, dtype: str, compiled: bool, seconds: float) -> dict:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    from lab.train.model import GPT

    mx.random.seed(0)
    model = GPT(cfg)
    model.set_dtype(getattr(mx, dtype))
    mx.eval(model.parameters())
    optimizer = optim.AdamW(learning_rate=3e-4)
    loss_and_grad = nn.value_and_grad(model, lambda m, x, y: mx.fast.cross_entropy(m(x), y).mean())

    def step(x, y):
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        return loss

    if compiled:
        state = [model.state, optimizer.state]
        step = partial(mx.compile, inputs=state, outputs=state)(step)
    rng = np.random.default_rng(0)

    def run_step() -> float:
        tokens = mx.array(rng.integers(0, cfg.vocab_size, (B, T + 1)).astype(np.int32))
        loss = step(tokens[:, :-1], tokens[:, 1:])
        mx.eval(loss, model.parameters(), optimizer.state)
        return loss.item()

    for _ in range(WARMUP_STEPS):
        run_step()
    mx.reset_peak_memory()
    steps, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        loss = run_step()
        steps += 1
    dt = time.perf_counter() - t0
    out = {"tokens_per_s": steps * B * T / dt, "steps": steps, "final_loss": loss,
           "loss_is_finite": math.isfinite(loss), "peak_memory_gb": mx.get_peak_memory() / 2**30}
    del model, optimizer
    free_gpu_memory()
    return out


def bench_torch(cfg: GPTConfig, autocast: str | None, seconds: float) -> dict:
    import torch
    import torch.nn.functional as F
    from torch import nn

    def rope(x: torch.Tensor) -> torch.Tensor:
        """Rotary position embedding on [B, heads, T, D]: rotate each pair of channels by a position-dependent angle."""
        d, t = x.shape[-1], x.shape[-2]
        freqs = 1.0 / (10000 ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
        angles = torch.arange(t, device=x.device, dtype=torch.float32)[:, None] * freqs[None, :]
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            n = cfg.n_embd
            self.norm1, self.norm2 = nn.RMSNorm(n), nn.RMSNorm(n)
            self.qkv = nn.Linear(n, 3 * n, bias=False)
            self.attn_out = nn.Linear(n, n, bias=False)
            self.fc = nn.Linear(n, 4 * n, bias=False)
            self.mlp_out = nn.Linear(4 * n, n, bias=False)

        def forward(self, x):
            b, t, c = x.shape
            q, k, v = self.qkv(self.norm1(x)).split(c, dim=-1)
            q, k, v = (a.view(b, t, cfg.n_head, -1).transpose(1, 2) for a in (q, k, v))
            a = F.scaled_dot_product_attention(rope(q), rope(k), v, is_causal=True)
            x = x + self.attn_out(a.transpose(1, 2).reshape(b, t, c))
            return x + self.mlp_out(F.gelu(self.fc(self.norm2(x))))

    class GPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
            nn.init.normal_(self.wte.weight, std=0.02)  # same start as the MLX model
            self.blocks = nn.ModuleList(Block() for _ in range(cfg.n_layer))
            self.norm = nn.RMSNorm(cfg.n_embd)

        def forward(self, tokens):
            x = self.wte(tokens)
            for block in self.blocks:
                x = block(x)
            return F.linear(self.norm(x), self.wte.weight)

    torch.manual_seed(0)
    device = torch.device("mps")
    model = GPT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(0)

    def run_step() -> float:
        tokens = torch.from_numpy(rng.integers(0, cfg.vocab_size, (B, T + 1))).to(device)
        with torch.autocast("mps", dtype=getattr(torch, autocast), enabled=autocast is not None) if autocast else torch.enable_grad():
            logits = model(tokens[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size).float(), tokens[:, 1:].reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return loss.item()  # waits for the GPU, like mx.eval on the MLX side

    for _ in range(WARMUP_STEPS):
        run_step()
    steps, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        loss = run_step()
        steps += 1
    dt = time.perf_counter() - t0
    out = {"tokens_per_s": steps * B * T / dt, "steps": steps, "final_loss": loss,
           "loss_is_finite": math.isfinite(loss),
           "driver_allocated_gb": torch.mps.driver_allocated_memory() / 2**30,
           "torch": torch.__version__}
    del model, optimizer
    gc.collect()
    torch.mps.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sizes", default="s,m,gpt2")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--skip-torch", action="store_true")
    args = ap.parse_args()

    for size in args.sizes.split(","):
        cfg = SIZES[size]
        arms = {
            "mlx float32 compiled": lambda: bench_mlx(cfg, "float32", True, args.seconds),
            "mlx float32 not compiled": lambda: bench_mlx(cfg, "float32", False, args.seconds),
            "mlx bfloat16 compiled": lambda: bench_mlx(cfg, "bfloat16", True, args.seconds),
            "mlx float16 compiled": lambda: bench_mlx(cfg, "float16", True, args.seconds),
        }
        if not args.skip_torch:
            arms |= {
                "torch-mps float32": lambda: bench_torch(cfg, None, args.seconds),
                "torch-mps autocast float16": lambda: bench_torch(cfg, "float16", args.seconds),
                "torch-mps autocast bfloat16": lambda: bench_torch(cfg, "bfloat16", args.seconds),
            }
        for label, run in arms.items():
            try:
                entry = run()
            except Exception as e:  # one arm failing (unsupported type, out of memory) must not hide the others
                entry = {"error": f"{type(e).__name__}: {e}"[:300]}
            entry["environment"] = environment()
            save("training", f"{size} | {label}", entry)
            shown = f"{entry['tokens_per_s']:>9,.0f} tok/s  loss {entry['final_loss']:.3f}" if "error" not in entry else entry["error"]
            print(f"{size:5s} {label:30s} {shown}", flush=True)


if __name__ == "__main__":
    main()
