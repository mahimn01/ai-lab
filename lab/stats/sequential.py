"""Statistics that stay valid when you look at the results after every problem.

Why
  An eval of 500 problems with 4 samples each is planned at a fixed size, but the answer
  is often clear long before the end. Looking early and stopping when a normal test says
  "significant" is not allowed: the more often you look, the more often noise passes the
  test. The two tools here are built to be looked at as often as you like.

What is here
  PairedTest       two conditions answered the same problems; is one right more often?
                   Only the problems where exactly one of them is right carry information
                   (the same idea as McNemar's test). The evidence is a likelihood ratio
                   averaged over all possible sizes of the difference; under "no
                   difference" it exceeds 1/alpha with probability at most alpha, however
                   many times you look (Ville's inequality).
  accuracy_band    a range for one condition's true accuracy that holds at every sample
                   size at once, from the same kind of averaged likelihood ratio.
  mcnemar_exact    the ordinary fixed-size test, for the planned final analysis.

  Problems must be analysed in an order fixed before the run, and only as far as every
  problem in that order has finished: short problems finish first, so "whatever has
  finished" is a biased sample. bench/phase1.py does this with --decide.

Run:     uv run python -m lab.stats.sequential          (prints a simulation: false alarms
         and how early typical differences are decided)
Changes: nothing. Read-only.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact p-value for 'A and B are right equally often', from the problems only one got right."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n
    return min(1.0, 2 * tail)


@dataclass
class PairedTest:
    """Evidence that two conditions differ in accuracy on the same problems, valid at every look.

    Feed one pair at a time with `add`. `decided` turns True once the evidence passes 1/alpha;
    after that the comparison may stop. `prior` is the Beta(prior, prior) weight on how
    lopsided the disagreements are (0.5 spreads it widely).
    """

    alpha: float = 0.05
    prior: float = 0.5
    only_a: int = 0
    only_b: int = 0
    pairs: int = 0
    peak_log_evidence: float = 0.0

    def add(self, a_right: bool, b_right: bool) -> None:
        """One independent pair of outcomes (one problem, one answer from each condition)."""
        self.pairs += 1
        self.only_a += bool(a_right and not b_right)
        self.only_b += bool(b_right and not a_right)
        self.peak_log_evidence = max(self.peak_log_evidence, self.log_evidence)

    def add_problem(self, a_right: int, b_right: int) -> None:
        """One problem answered several times by each condition: counts as a win for whichever was right more often.

        Samples of one problem rise and fall together, so they are not separate pieces of evidence;
        feeding them to `add` one by one would raise false alarms. The problem is the unit.
        """
        self.add(a_right > b_right, b_right > a_right)

    @property
    def log_evidence(self) -> float:
        """Log of the likelihood ratio against 'no difference' (0 = no evidence either way)."""
        n = self.only_a + self.only_b
        return (_log_beta(self.only_a + self.prior, self.only_b + self.prior) - _log_beta(self.prior, self.prior)
                + n * math.log(2))

    @property
    def decided(self) -> bool:
        return self.peak_log_evidence >= math.log(1 / self.alpha)

    @property
    def p_value(self) -> float:
        """A p-value that remains valid under repeated looks: 1 over the largest evidence seen so far."""
        return min(1.0, math.exp(-self.peak_log_evidence))


def accuracy_band(right: int, n: int, alpha: float = 0.05, prior: float = 0.5) -> tuple[float, float]:
    """A range for the true accuracy that is valid at every n at once (wider than a fixed-n interval)."""
    if n == 0:
        return 0.0, 1.0
    log_mix = _log_beta(right + prior, n - right + prior) - _log_beta(prior, prior)

    def inside(p: float) -> bool:
        if p <= 0.0 or p >= 1.0:
            return (p <= 0.0 and right == 0) or (p >= 1.0 and right == n)
        return log_mix - (right * math.log(p) + (n - right) * math.log(1 - p)) < math.log(1 / alpha)

    def edge(lo: float, hi: float) -> float:
        """The point between lo (inside the band) and hi (outside it) where the band ends."""
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if inside(mid) else (lo, mid)
        return lo

    centre = min(max(right / n, 1e-12), 1 - 1e-12)
    return (0.0 if inside(0.0) else edge(centre, 0.0)), (1.0 if inside(1.0) else edge(centre, 1.0))


def simulate(p_only_a: float, p_only_b: float, n: int, alpha: float = 0.05, runs: int = 2000, seed: int = 0) -> dict:
    """How the paired test behaves when each problem is 'only A right' / 'only B right' with these chances.

    Returns the share of runs decided within n problems, the median problem count at the
    decision, and the share a fixed-size McNemar test at n would call significant.
    """
    rng = random.Random(seed)
    stops, fixed_hits = [], 0
    for _ in range(runs):
        test, stop = PairedTest(alpha=alpha), None
        for i in range(n):
            r = rng.random()
            test.add(r < p_only_a, p_only_a <= r < p_only_a + p_only_b)
            if stop is None and test.decided:
                stop = i + 1
        if stop is not None:
            stops.append(stop)
        fixed_hits += mcnemar_exact(test.only_a, test.only_b) < alpha
    stops.sort()
    return {"decided_share": len(stops) / runs, "median_problems_when_decided": stops[len(stops) // 2] if stops else None,
            "fixed_size_test_share": fixed_hits / runs}


def main() -> None:
    print("Paired test, alpha 0.05, looked at after every problem (2,000 simulated runs each):")
    cases = [("no difference (10% vs 10% disagreements)", 0.10, 0.10, 2000),
             ("looping 66% vs 17% (about 52% vs 3%)", 0.52, 0.03, 100),
             ("accuracy 54% vs 43% as in Phase 0 (21% vs 10%)", 0.21, 0.10, 100),
             ("the same difference, 2,000 problem-samples", 0.21, 0.10, 2000),
             ("a 3-point difference (14% vs 11%)", 0.14, 0.11, 2000)]
    for name, a, b, n in cases:
        s = simulate(a, b, n)
        print(f"  {name}: decided in {s['decided_share']:.1%} of runs within {n}"
              + (f", median at problem {s['median_problems_when_decided']}" if s["median_problems_when_decided"] else "")
              + f"; a fixed-size test at {n} is significant in {s['fixed_size_test_share']:.1%}")


if __name__ == "__main__":
    main()
