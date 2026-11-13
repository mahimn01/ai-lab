"""Grade a MATH answer the way the blog's own evaluation does: math-verify on the last boxed answer.

Why
  The neural-harness study graded Phase 0 with the original MATH string comparison plus three
  extensions (its harness/grader.py). For Phase 1 it names math-verify (RESULTS-P0.md 6.1), which
  the blog and the published re-run of the edited model used. This file is that protocol, written
  from the blog's code (github.com/RikiyaT/lm_understand-logs, exp8_math/code/mathqa.py: `extract`
  and `is_correct`), with the versions the published run names: math-verify 0.9.0,
  latex2sympy2_extended 1.11.0, antlr4 4.13.2.

The protocol
  - Only the text after the last "</think>" is read. If thinking never closed, the whole text is.
  - The answer is what the last \\boxed{...} holds (the last \\fbox{...} if there is no \\boxed at
    all), braces matched. "\\boxed 5" without braces is read up to the next "$" or line end.
    A box that is not closed, or is empty, is no answer.
  - It is right when verify(parse("\\boxed{gold}"), parse("\\boxed{answer}")) says so, each call
    with a 5-second limit. An answer that does not parse is wrong.

How it differs from the study's grader
  In what it reads, not only in how it compares: the study's grader takes the last box anywhere,
  thinking included. So a row that boxed an answer while thinking and none after it is right there
  and wrong here.

Checked: tests/test_grading.py, and bench/grader_check.py against the published samples and verdicts.
Use:     from bench.grading import grade
         answer, right = grade(text, gold)
Changes: nothing. math-verify's time limits work only in the main thread.
"""

from __future__ import annotations

from functools import lru_cache

from math_verify import parse, verify

THINK_CLOSE = "</think>"
TIME_LIMIT_S = 5


def boxed_answer(text: str) -> str | None:
    """What the last box after the thinking holds, or None."""
    text = text.rsplit(THINK_CLOSE, 1)[-1]
    for command in ("\\boxed", "\\fbox"):
        start = text.rfind(command)
        if start >= 0:
            break
    else:
        return None
    rest = text[start + len(command):]
    body = rest.lstrip()
    if body.startswith("{"):
        depth = 0
        for i, ch in enumerate(body):
            depth += (ch == "{") - (ch == "}")
            if depth == 0:
                return body[1:i].strip() or None
        return None          # not closed: the text was cut inside the box
    if command == "\\boxed" and rest.startswith(" "):
        return rest.strip().split("$")[0].split("\n")[0].strip() or None
    return None


@lru_cache(maxsize=1 << 16)
def _parsed(answer: str) -> list:
    return parse("\\boxed{" + answer + "}", parsing_timeout=TIME_LIMIT_S)


@lru_cache(maxsize=1 << 16)
def same_answer(answer: str | None, gold: str) -> bool:
    """Does math-verify take the answer and the gold answer for the same thing?"""
    if answer is None:
        return False
    g, a = _parsed(gold), _parsed(answer)
    return bool(g and a and verify(g, a, timeout_seconds=TIME_LIMIT_S))


def grade(text: str, gold: str) -> tuple[str | None, bool]:
    """(the answer read from the text, whether it is right): the same shape as the study's `grade`."""
    answer = boxed_answer(text)
    return answer, same_answer(answer, gold)
