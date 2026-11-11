"""
Paired comparison of two runs on the same questions.

    python scripts/compare_runs.py results/offline_baseline_retrieval results/offline_improved_retrieval

For each metric: mean A, mean B, difference with a paired-bootstrap 95% CI, wins/losses/ties,
and a sign-flip permutation p-value. If the CI includes 0, you have NOT shown an improvement.
"""
import argparse
import json
import os

import _path  # noqa: F401

from ragkit.eval.stats import paired_bootstrap

DEFAULT = ["hit@1", "hit@3", "hit@5", "mrr", "ndcg@5", "correctness", "faithfulness"]


def load(d):
    with open(os.path.join(d, "per_question.jsonl")) as f:
        return {r["id"]: r for r in map(json.loads, f)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--metrics", nargs="*", default=DEFAULT)
    args = ap.parse_args()
    A, B = load(args.run_a), load(args.run_b)
    ids = [i for i in A if i in B]
    print(f"A = {args.run_a}\nB = {args.run_b}\n{len(ids)} shared questions\n")
    print(f"{'metric':14s} {'A':>6s} {'B':>6s} {'B-A':>7s}  {'95% CI':>17s}  {'W/L/T':>9s}  {'p':>6s}")
    for m in args.metrics:
        pairs = [(A[i].get(m), B[i].get(m)) for i in ids]
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
        if not pairs:
            continue
        a, b = zip(*pairs)
        r = paired_bootstrap(a, b)
        sig = "*" if r["ci_low"] > 0 or r["ci_high"] < 0 else " "
        print(f"{m:14s} {sum(a)/len(a):6.3f} {sum(b)/len(b):6.3f} {r['mean_diff']:+7.3f}{sig} "
              f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]  {r['wins']:>2d}/{r['losses']:<2d}/{r['ties']:<2d}  {r['p_value']:6.3f}")
    print("\n* = 95% CI excludes 0")
    changed = [i for i in ids if A[i].get("cause") != B[i].get("cause")]
    if changed:
        print("\nQuestions whose diagnosis changed:")
        for i in changed:
            print(f"  {i:8s} {A[i]['cause']:20s} -> {B[i]['cause']:20s} {A[i]['question'][:60]}")
