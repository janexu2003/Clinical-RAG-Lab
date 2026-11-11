"""
Statistics for SMALL evaluation sets.

With 30 questions, one question is 3.3 percentage points. Before claiming "config B beats A",
check the uncertainty:

  bootstrap_ci(values)            95% interval for a mean (resample questions with replacement)
  paired_bootstrap(a, b)          compare two configs on the SAME questions; the paired design
                                  removes question difficulty and gives much tighter intervals
  sign_flip_pvalue(a, b)          exact-ish permutation test on paired differences
  cohen_kappa(x, y)               agreement between the LLM judge and your own labels
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Dict, List, Sequence, Tuple


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(values: Sequence[float], n: int = 5000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float, float]:
    vals = list(values)
    if not vals:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    means = sorted(mean([vals[rng.randrange(len(vals))] for _ in vals]) for _ in range(n))
    return mean(vals), means[int(alpha / 2 * n)], means[int((1 - alpha / 2) * n) - 1]


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n: int = 5000, seed: int = 0) -> Dict:
    assert len(a) == len(b), "paired comparison needs the same questions in both runs"
    diffs = [y - x for x, y in zip(a, b)]
    m, lo, hi = bootstrap_ci(diffs, n=n, seed=seed)
    return {"mean_diff": m, "ci_low": lo, "ci_high": hi,
            "wins": sum(d > 0 for d in diffs), "losses": sum(d < 0 for d in diffs),
            "ties": sum(d == 0 for d in diffs), "p_value": sign_flip_pvalue(a, b, seed=seed)}


def sign_flip_pvalue(a: Sequence[float], b: Sequence[float], n: int = 10000, seed: int = 0) -> float:
    diffs = [y - x for x, y in zip(a, b)]
    obs = abs(sum(diffs))
    if obs == 0:
        return 1.0
    rng = random.Random(seed)
    hits = sum(abs(sum(d if rng.random() < 0.5 else -d for d in diffs)) >= obs - 1e-12 for _ in range(n))
    return (hits + 1) / (n + 1)


def cohen_kappa(x: Sequence, y: Sequence) -> float:
    assert len(x) == len(y) and x
    labels = set(x) | set(y)
    po = sum(a == b for a, b in zip(x, y)) / len(x)
    cx, cy = Counter(x), Counter(y)
    pe = sum(cx[l] * cy[l] for l in labels) / (len(x) ** 2)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0
