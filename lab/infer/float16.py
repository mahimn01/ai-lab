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
  Values are recorded at every layer's output and around every weight table of the usual
  kinds (linear, quantized linear, convolution, mlx-lm's switch layers); anything else
  is covered only by step 3.

Not for training
  The error signals that training passes backwards are far smaller than the values going
  forwards and would round to zero in float16. Fine-tune in bfloat16.

Use:     from lab.infer import float16
         report = float16.convert(model, tokenizer)        # or prompts=[token ids, ...]
         float16.revert(model)                             # every float16 parameter back to bfloat16
Changes: the model in memory only. Nothing on disk. While it checks, it holds a second
         copy of the parameters it converts (small for a quantized model; as much again
         as the weights for one that is not quantized).
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
        user = {"role": "user", "content": SAMPLE_QUESTION}
        for messages in ([{"role": "system", "content": "You are a careful assistant."}, user], [user]):
            try:
                prompts.append(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True))
                break
            except Exception:  # some templates refuse a system message: try without one, else go on without a chat prompt
                continue
    prompts.append((prompts[0] + prompts[1]) * 6)
    return prompts


def _table_kinds() -> tuple:
    kinds = (nn.Linear, nn.QuantizedLinear, nn.Conv1d)
    try:
        from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchLinear
    except ImportError:
        return kinds
    return kinds + (SwitchLinear, QuantizedSwitchLinear)


@contextmanager
def _recording(model: nn.Module) -> Iterator[list[mx.array]]:
    """While the block runs, every weight table of the model notes the largest value it reads and writes."""
    seen: list[mx.array] = []
    kinds = _table_kinds()
    # None marks a class that inherits __call__: it gets its own for the block and loses it again after
    originals = {cls: cls.__dict__.get("__call__") for cls in {type(m) for _, m in model.named_modules() if isinstance(m, kinds)}}

    def recording(call):
        def run(self, x, *args, **kwargs):
            y = call(self, x, *args, **kwargs)
            seen.append(mx.maximum(mx.abs(x).max(), mx.abs(y).max()).astype(mx.float32))
            return y
        return run

    for cls in originals:
        cls.__call__ = recording(cls.__call__)
    try:
        yield seen
    finally:
        for cls, own in originals.items():
            if own is None:
                del cls.__call__
            else:
                cls.__call__ = own


def _batch_of_one(prompt) -> mx.array:
    """Token ids (a list or an array, of Python or numpy integers) as tokens [1, T]."""
    return mx.array(np.asarray(prompt, dtype=np.int32))[None]


def _answers(model: nn.Module, prompt) -> tuple[mx.array, mx.array, mx.array]:
    """One prompt through the model: the next token it picks at every position, its scores after the last
    position, and whether every score is a number. (Not the full scores: those are hundreds of MB per prompt.)"""
    scores = model(_batch_of_one(prompt))[0]
    return mx.argmax(scores, axis=-1), scores[-1].astype(mx.float32), mx.isfinite(scores).all()


def largest_value(model: nn.Module, prompts: list[list[int]]) -> tuple[float, list[tuple[mx.array, mx.array, mx.array]]]:
    """The largest number the model writes anywhere for these prompts, and its answers to each (see _answers)."""
    largest, answers = 0.0, []
    for prompt in prompts:
        with _recording(model) as seen, ExitStack() as layer_hooks:
            def note(out: mx.array) -> mx.array:
                seen.append(mx.abs(out).max().astype(mx.float32))
                return out

            for i in range(len(hooks.layer_list(model))):
                layer_hooks.enter_context(hooks.hook(model, i, note))
            answer = _answers(model, prompt)
            top = mx.max(mx.nan_to_num(mx.stack(seen), nan=float("inf")))
            mx.eval(answer, top)  # together, so nothing is kept in memory longer than in a normal pass
        largest = max(largest, top.item())
        answers.append(answer)
    return largest, answers


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
        originals = model.parameters()  # the arrays themselves, to put back exactly if anything below fails
        converted = _cast(model, mx.bfloat16, mx.float16)
        try:
            after = [_answers(model, prompt) for prompt in prompts]
            mx.eval(after)
            finite = all(ok.item() for _, _, ok in after)
            tokens = sum(len(prompt) for prompt in prompts)
            same = sum((a[0] == b[0]).sum().item() for a, b in zip(after, before)) / tokens
            gap = sum(mx.abs(a[1] - b[1]).mean().item() for a, b in zip(after, before)) / len(prompts)
            if not finite or same < MIN_AGREEMENT:
                raise ValueError("not converted: in float16 the model " + ("produced scores that are not numbers" if not finite else
                                 f"picked the same next token as before at only {same:.0%} of positions"))
        except BaseException:  # a failed check, an interrupt, running out of memory: the model goes back as it was
            model.update(originals)
            raise
    return {"converted_parameters": converted, "largest_value": largest, "headroom": headroom, "same_top_token": same,
            "last_position_score_difference": gap, "prompts_checked": len(prompts), "tokens_checked": tokens}


def revert(model: nn.Module) -> int:
    """Switch every float16 parameter of a model back to bfloat16 (so only use it on a model `convert` changed).

    Returns how many parameters were converted.
    """
    with _kernel_set_aside(model):
        return _cast(model, mx.float16, mx.bfloat16)
