"""The blog's grading protocol: the last box after the thinking, checked with math-verify (bench/grading.py)."""

from __future__ import annotations

import pytest

from bench.grading import boxed_answer, grade, same_answer


@pytest.mark.parametrize("text, answer", [
    ("so \\boxed{1} and later \\boxed{ \\frac{1}{2} }.", "\\frac{1}{2}"),       # the last box, spaces trimmed
    ("maybe \\boxed{7}</think>The answer is 9.", None),                         # a box inside closed thinking does not count
    ("still thinking, maybe \\boxed{7}", "7"),                                   # thinking never closed: the whole text is read
    ("a</think>\\boxed{1}</think>\n\\boxed{2}", "2"),                            # after the last </think>
    ("\\boxed{\\frac{a}{b}", None),                                              # cut inside the box
    ("\\boxed{1} then \\boxed{2", None),                                         # the last box decides, closed or not
    ("\\boxed{}", None),
    ("\\boxed {x^{2}}", "x^{2}"),
    ("the answer is $\\boxed 5$ so", "5"),
    ("\\boxed5", None),
    ("\\fbox{3}", "3"),
    ("\\fbox{3} and \\boxed{4} and \\fbox{5}", "4"),                             # \fbox only when there is no \boxed at all
    ("no box", None),
])
def test_the_answer_is_the_last_box_after_the_thinking(text, answer):
    assert boxed_answer(text) == answer


@pytest.mark.parametrize("answer, gold, right", [
    ("(3, \\pi/2)", "\\left( 3, \\frac{\\pi}{2} \\right)", True),
    ("0.5", "\\frac{1}{2}", True),
    ("x^2+2x+1", "(x+1)^2", True),
    ("10000", "10{,}000", True),
    ("Evelyn", "\\text{Evelyn}", True),
    ("x=5", "5", True),
    ("2, 3", "3, 2", True),
    ("4.6667", "\\frac{14}{3}", False),
    ("[2,5]", "[2,5)", False),
    ("\\text{east}", "\\text{west}", False),
    ("garbage {{", "5", False),
    (None, "5", False),
])
def test_math_verify_decides_whether_two_answers_are_the_same(answer, gold, right):
    assert same_answer(answer, gold) is right


def test_grade_reads_the_answer_and_checks_it():
    assert grade("thinking \\boxed{3}</think>\nSo the answer is \\boxed{\\frac12}.", "\\frac{1}{2}") == ("\\frac12", True)
    assert grade("thinking \\boxed{\\frac12}</think>\nI am not sure.", "\\frac{1}{2}") == (None, False)
