"""A small GPT language model in MLX, built to be read and changed.

It is the standard modern recipe: token embeddings, then blocks of (causal attention
with rotary positions, then a 4x-wide MLP), each with a pre-normalisation and a
residual connection, then a final normalisation and an output layer that shares its
weights with the embedding table.

Sizes (total parameters with the lab's 8,192-entry vocabulary, shared embedding counted once):
  s     6 layers, width 384    about 14M
  m     8 layers, width 512    about 29M
  gpt2  12 layers, width 768   about 91M (the GPT-2 small shape; 124M with GPT-2's own 50,257 vocabulary)
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384


SIZES = {
    "s": GPTConfig(n_layer=6, n_head=6, n_embd=384),
    "m": GPTConfig(n_layer=8, n_head=8, n_embd=512),
    "gpt2": GPTConfig(n_layer=12, n_head=12, n_embd=768),
}


class Block(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.n_head = c.n_head
        self.norm1 = nn.RMSNorm(c.n_embd)
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd, bias=False)
        self.attn_out = nn.Linear(c.n_embd, c.n_embd, bias=False)
        self.rope = nn.RoPE(c.n_embd // c.n_head)
        self.norm2 = nn.RMSNorm(c.n_embd)
        self.fc = nn.Linear(c.n_embd, 4 * c.n_embd, bias=False)
        self.mlp_out = nn.Linear(4 * c.n_embd, c.n_embd, bias=False)
        # Start each block as "do nothing": its two output layers begin at zero.
        self.attn_out.weight = mx.zeros_like(self.attn_out.weight)
        self.mlp_out.weight = mx.zeros_like(self.mlp_out.weight)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, C = x.shape
        q, k, v = mx.split(self.qkv(self.norm1(x)), 3, axis=-1)
        # [B, T, C] -> [B, heads, T, C/heads]
        q, k, v = (a.reshape(B, T, self.n_head, -1).transpose(0, 2, 1, 3) for a in (q, k, v))
        q, k = self.rope(q), self.rope(k)
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5, mask="causal")
        x = x + self.attn_out(a.transpose(0, 2, 1, 3).reshape(B, T, C))
        return x + self.mlp_out(nn.gelu(self.fc(self.norm2(x))))


class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.wte = nn.Embedding(c.vocab_size, c.n_embd)
        self.wte.weight = mx.random.normal(self.wte.weight.shape, scale=0.02)
        self.blocks = [Block(c) for _ in range(c.n_layer)]
        self.norm = nn.RMSNorm(c.n_embd)

    def __call__(self, tokens: mx.array) -> mx.array:
        """tokens [B, T] of ints -> logits [B, T, vocab] (a score for every possible next token)."""
        x = self.wte(tokens)
        for block in self.blocks:
            x = block(x)
        return self.wte.as_linear(self.norm(x))


def n_params(model: nn.Module) -> int:
    return sum(v.size for _, v in tree_flatten(model.parameters()))
