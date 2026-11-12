"""Run a model in float16 instead of the bfloat16 it was shipped in: the same model, faster GPU code.

Why
  The Qwen models in MLX format store their 16-bit numbers as bfloat16. Apple GPUs have
  no bfloat16 type, so MLX's GPU code converts every such number in software. float16 is
  a native type. Measured on this Mac (SPEED-RESULTS.md section 16): prompts are read
  1.4x faster with MLX's own kernels, 32 prompts are decoded at once 1.25-1.3x faster,
  and one prompt alone is unchanged (that case waits on memory, not arithmetic).

The catch, and the check
  float16 keeps more digits than bfloat16 (11 bits against 8) but a far smaller range:
  its largest number is 65,504. If any value inside the model went beyond that, the
  output would turn into nonsense without an error. `convert` therefore
    1. runs sample prompts through the model as it is and notes the largest value
       written anywhere (every layer's output, every weight table's input and output),
    2. refuses if that value comes within `min_headroom` (4x) of the limit,
    3. converts, runs the same prompts again, and undoes the conversion if any score is
       not a number or the two versions disagree on the most likely next token too often.
  The check covers the prompts it is given. Pass prompts like the ones you will use; a
  very different input could still overflow, and the only sign would be garbled output.

Not for training
  The error signals that training passes backwards are far smaller than the values going
  forwards and would round to zero in float16. Fine-tune in bfloat16.

Use:     from lab.infer import float16
         report = float16.convert(model, tokenizer)        # or prompts=[token ids, ...]
         float16.revert(model)                             # back to bfloat16
Changes: the model in memory only. Nothing on disk.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import Iterator

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from lab.edit import hooks
from lab.kernels import qmm

LIMIT = 65504.0  # the largest float16 number
MIN_HEADROOM = 4.0
MIN_AGREEMENT = 0.95  # share of positions where both versions must pick the same next token
# Different kinds of text for the check when no prompts are given: chat, prose, code, another script, numbers.
SAMPLE_TEXTS = (
    "The mitochondria of a cell convert the energy stored in nutrients into a form the cell can use. "
    "They have their own small genome, inherited from the mother, and divide independently of the cell.",
    "def merge(left, right):\n    out = []\n    while left and right:\n        out.append(left.pop(0) if left[0] <= right[0] "
    "else right.pop(0))\n    return out + left + right\n\nprint(merge([1, 4, 9], [2, 3, 10]))",
    "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。 Привет, как дела? ¿Dónde está la biblioteca?",
    "3.14159 26535 89793 23846 26433 83279 50288 41971 69399 37510 | 1e-9 6.02e23 -273.15 0x7fff 65504 1000000007",
)
SAMPLE_QUESTION = "Explain in three sentences why the sky is blue, then give the answer as one line of Python."


def sample_prompts(tokenizer) -> list[list[int]]:
    """Token ids for the built-in sample: four kinds of plain text, a chat prompt, and one long repetitive prompt."""
    prompts = [tokenizer.encode(text) for text in SAMPLE_TEXTS]
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "system", "content": "You are a careful assistant."}, {"role": "user", "content": SAMPLE_QUESTION}]
        prompts.append(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True))
    prompts.append((prompts[0] + prompts[1]) * 6)
    return prompts


@contextmanager
def _recording(model: nn.Module) -> Iterator[list[mx.array]]:
    """While the block runs, every weight table of the model notes the largest value it reads and writes."""
    seen: list[mx.array] = []
    kinds = (nn.Linear, nn.QuantizedLinear, nn.Conv1d)
    originals = {cls: cls.__call__ for cls in {type(m) for _, m in model.named_modules() if isinstance(m, kinds)}}

    def recording(call):
        def run(self, x, *args, **kwargs):
            y = call(self, x, *args, **kwargs)
            seen.append(mx.maximum(mx.abs(x).max(), mx.abs(y).max()).astype(mx.float32))
            return y
        return run

    for cls, call in originals.items():
        cls.__call__ = recording(call)
    try:
        yield seen
    finally:
        for cls, call in originals.items():
            cls.__call__ = call


def _batch_of_one(prompt) -> mx.array:
    """Token ids (a list or an array, of Python or numpy integers) as tokens [1, T]."""
    return mx.array(np.asarray(prompt, dtype=np.int32))[None]


def largest_value(model: nn.Module, prompts: list[list[int]]) -> tuple[float, list[mx.array]]:
    """The largest number the model writes anywhere for these prompts, and its scores [T, vocab] for each prompt."""
    largest, scores = 0.0, []
    for prompt in prompts:
        with _recording(model) as seen, ExitStack() as layer_hooks:
            def note(out: mx.array) -> mx.array:
                seen.append(mx.abs(out).max().astype(mx.float32))
                return out

            for i in range(len(hooks.layer_list(model))):
                layer_hooks.enter_context(hooks.hook(model, i, note))
            logits = model(_batch_of_one(prompt))[0].astype(mx.float32)
            top = mx.max(mx.nan_to_num(mx.stack(seen), nan=float("inf")))
            mx.eval(logits, top)  # together, so nothing is kept in memory longer than in a normal pass
        largest = max(largest, top.item())
        scores.append(logits)
    return largest, scores


@contextmanager
def _kernel_set_aside(model: nn.Module) -> Iterator[None]:
    """The lab kernel prepares its code for one number format: take it off for the block, set it up again after."""
    installed = qmm.uninstall(model) > 0
    try:
        yield
    finally:
        if installed:
            qmm.install(model)


def _cast(model: nn.Module, source: mx.Dtype, target: mx.Dtype) -> int:
    """Convert every parameter stored as `source` to `target`. Returns how many were converted."""
    count = sum(1 for _, value in tree_flatten(model.parameters()) if value.dtype == source)
    model.apply(lambda value: value.astype(target) if value.dtype == source else value)
    mx.eval(model.parameters())
    return count


def convert(model: nn.Module, tokenizer=None, *, prompts: list[list[int]] | None = None,
            min_headroom: float = MIN_HEADROOM) -> dict:
    """Switch a bfloat16 model to float16 after checking it is safe on sample prompts. Returns what the check found.

    Give either `prompts` (lists of token ids, ideally like the ones you will use) or a tokenizer
    for the built-in sample. Raises ValueError and leaves the model unchanged if the check fails.
    """
    if prompts is None:
        if tokenizer is None:
            raise ValueError("give prompts=[token ids, ...] or a tokenizer for the built-in sample")
        prompts = sample_prompts(tokenizer)
    with _kernel_set_aside(model):
        largest, before = largest_value(model, prompts)
        headroom = LIMIT / max(largest, 1e-30)
        if headroom < min_headroom:
            raise ValueError(f"not converted: a value of {largest:,.0f} was seen inside the model, within {headroom:.1f}x of "
                             f"float16's limit of {LIMIT:,.0f} (at least {min_headroom:.0f}x of room is required)")
        originals = model.parameters()  # the arrays themselves, to put back exactly if the check fails
        converted = _cast(model, mx.bfloat16, mx.float16)
        after = [model(_batch_of_one(prompt))[0].astype(mx.float32) for prompt in prompts]
        finite = all(mx.isfinite(scores).all().item() for scores in after)
        same = sum((mx.argmax(a, -1) == mx.argmax(b, -1)).sum().item() for a, b in zip(after, before)) / sum(len(p) for p in prompts)
        gap = sum(mx.abs(a - b).mean().item() for a, b in zip(after, before)) / len(prompts)
        if not finite or same < MIN_AGREEMENT:
            model.update(originals)
            raise ValueError("not converted: in float16 the model " + ("produced scores that are not numbers" if not finite else
                             f"picked the same next token as before at only {same:.0%} of positions"))
    return {"converted_parameters": converted, "largest_value": largest, "headroom": headroom, "same_top_token": same,
            "mean_score_difference": gap, "prompts_checked": len(prompts), "tokens_checked": sum(len(p) for p in prompts)}


def revert(model: nn.Module) -> int:
    """Switch a model converted by `convert` back to bfloat16. Returns how many parameters were converted."""
    with _kernel_set_aside(model):
        return _cast(model, mx.float16, mx.bfloat16)
