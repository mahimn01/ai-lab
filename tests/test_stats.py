"""Statistics that may be looked at after every problem (lab/stats/sequential.py)."""

from __future__ import annotations

import random

from lab.stats.sequential import PairedTest, accuracy_band, mcnemar_exact, simulate


def test_mcnemar_reproduces_the_studys_published_p_value():
    assert round(mcnemar_exact(21, 10), 3) == 0.071          # Phase 0: edited at 32k against base with thinking off
    assert mcnemar_exact(0, 0) == 1.0 and mcnemar_exact(5, 5) == 1.0


def test_paired_test_rarely_fires_without_a_difference_however_often_it_is_looked_at():
    nothing = simulate(0.10, 0.10, n=1000, runs=400, seed=1)
    assert nothing["decided_share"] <= 0.05


def test_paired_test_decides_a_large_difference_early_and_keeps_the_decision():
    big = simulate(0.52, 0.03, n=100, runs=200, seed=2)
    assert big["decided_share"] == 1.0 and big["median_problems_when_decided"] < 30
    test = PairedTest()
    for _ in range(12):
        test.add(True, False)
    assert test.decided and test.p_value < 0.05
    for _ in range(12):                                       # evidence falling back later does not undo the decision
        test.add(False, True)
    assert test.decided and test.pairs == 24 and test.only_a == test.only_b == 12


def test_several_samples_of_a_problem_count_as_one_piece_of_evidence():
    # Half the problems favour each side (0.95 against 0.25 correct), so the two conditions are equal on average,
    # but the 4 samples of a problem agree with each other. Fed one by one they would raise false alarms.
    rng = random.Random(4)
    alarms_by_problem = alarms_by_sample = 0
    for _ in range(300):
        by_problem, by_sample = PairedTest(), PairedTest()
        for _ in range(125):
            pa, pb = (0.95, 0.25) if rng.random() < 0.5 else (0.25, 0.95)
            a, b = [rng.random() < pa for _ in range(4)], [rng.random() < pb for _ in range(4)]
            by_problem.add_problem(sum(a), sum(b))
            for x, y in zip(a, b):
                by_sample.add(x, y)
        alarms_by_problem += by_problem.decided
        alarms_by_sample += by_sample.decided
    assert alarms_by_problem / 300 <= 0.05 < alarms_by_sample / 300


def test_accuracy_band_contains_the_truth_at_every_look():
    rng, misses = random.Random(3), 0
    for _ in range(300):
        right, missed = 0, False
        for n in range(1, 301):
            right += rng.random() < 0.3
            low, high = accuracy_band(right, n)
            missed |= not low <= 0.3 <= high
        misses += missed
    assert misses / 300 <= 0.05
    low, high = accuracy_band(54, 100)
    assert low < 0.54 < high and high - low < 0.4
    assert accuracy_band(0, 0) == (0.0, 1.0) and accuracy_band(0, 30)[0] == 0.0 and accuracy_band(30, 30)[1] == 1.0
