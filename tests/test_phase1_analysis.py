"""The planned Phase 1 comparisons (bench/phase1_analysis.py)."""

from __future__ import annotations

import pytest

from bench.phase1_analysis import analyse, holm, sign_flip_p, wilson
from lab.stats.sequential import mcnemar_exact


def entry(samples: int, order: list[str], rows: list[str], caps: dict[str, str], exits: dict[str, str] | None = None) -> dict:
    """One arm's entry as bench/phase1.py saves it, with only what the analysis reads."""
    return {"samples": samples, "problems_in_order": order, "conditions": {"arm": {
        "correct_by_problem": dict.fromkeys(rows, True),
        "at_smaller_caps": {cap: {"by_row": marks, "mean_tokens": float(cap)} for cap, marks in caps.items()},
        "forced_exit": {b: {"by_row": marks, "mean_tokens": float(b) + 1} for b, marks in (exits or {}).items()}}}}


def test_the_sign_flip_test_is_mcnemars_with_one_sample_per_problem():
    for only_first, only_second in ((21, 10), (0, 0), (5, 5), (1, 9), (30, 3)):
        differences = [1] * only_first + [-1] * only_second + [0] * 7
        assert sign_flip_p(differences) == pytest.approx(mcnemar_exact(only_first, only_second), abs=1e-12)


def test_the_sign_flip_test_counts_every_way_the_signs_could_have_fallen():
    assert sign_flip_p([2, 1]) == 0.5               # sums 3, 1, -1, -3: two of the four are as far out as 3
    assert sign_flip_p([2, -1]) == 1.0
    assert sign_flip_p([3, 3, 3, 3, 3]) == 2 / 32


def test_holm_multiplies_by_the_tests_left_and_never_falls():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.9]) == [1.0, 1.0] and holm([]) == []


def test_wilson_interval():
    low, high = wilson(54, 100)
    assert (round(low, 2), round(high, 2)) == (0.44, 0.63)       # Phase 0: the edited model at 32,768, 54% of 100


def test_only_the_front_of_a_run_that_was_cut_short_is_analysed():
    order = ["a", "b", "c", "d"]
    rows = [f"{p}#0" for p in order]
    out = analyse({"nothink": entry(1, order, rows, {"8": "1100"}),
                   "edited": entry(1, order, rows, {"32": "1110"}, {"8": "1011"}),
                   "base": entry(1, order, rows[:3], {"24": "000"}, {"8": "01-"})},      # c's branch is missing, d never finished
                  [("edited@32", "base+exit@8"), ("edited@32", "nothink@8"), ("edited+exit@8", "base+exit@8")])
    assert (out["problems_planned"], out["problems_analysed"]) == (4, 2)
    assert out["conditions"]["edited@32"]["accuracy"] == 1.0 and out["conditions"]["base+exit@8"]["accuracy"] == 0.5
    assert out["conditions"]["base+exit@8"]["mean_tokens"] == 9.0
    assert out["primary"]["difference"] == 0.5 and out["primary"]["problems_where_the_first_did_better"] == 1
    assert [c["first"] for c in out["secondary"]] == ["edited@32", "edited+exit@8"] and all("p_holm" in c for c in out["secondary"])


def test_samples_of_one_problem_count_as_one_piece_of_evidence():
    order = ["a", "b", "c"]
    rows = [f"{p}#{k}" for p in order for k in range(2)]
    out = analyse({"nothink": entry(2, order, rows, {"8": "000000"}),
                   "edited": entry(2, order, rows, {"32": "111110"}),
                   "base": entry(2, order, rows, {"24": "100000"})}, [("edited@32", "base@24")])
    assert out["problems_analysed"] == 3 and out["conditions"]["edited@32"]["accuracy"] == pytest.approx(5 / 6)
    assert out["primary"]["difference"] == pytest.approx(4 / 6)
    assert out["primary"]["p"] == 0.25               # differences of 1, 2 and 1 samples: all of one sign in 2 of 8 ways
