"""
Can you trust the LLM judge? Check it against your own labels.

1) export a sample of judged answers to label yourself:
     python scripts/judge_agreement.py export results/improved --n 20
   -> results/improved/human_labels.csv  (fill the 'human_score' column with 1 / 0.5 / 0)
2) compute agreement:
     python scripts/judge_agreement.py score results/improved
Cohen's kappa > 0.6 is usually considered substantial agreement. If it is low, fix the judge
prompt or use a stronger judge model before trusting correctness numbers.
"""
import argparse
import csv
import json
import os
import random

import _path  # noqa: F401

from ragkit.eval.stats import cohen_kappa

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["export", "score"])
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    path = os.path.join(args.run, "human_labels.csv")
    if args.mode == "export":
        rows = [json.loads(l) for l in open(os.path.join(args.run, "per_question.jsonl"))]
        rows = [r for r in rows if "correctness" in r]
        random.Random(0).shuffle(rows)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "question", "answer", "judge_score", "human_score"])
            for r in rows[: args.n]:
                w.writerow([r["id"], r["question"], r["answer"], r["correctness"], ""])
        print(f"label {path} (human_score = 1 / 0.5 / 0), then run: score {args.run}")
    else:
        rows = [r for r in csv.DictReader(open(path)) if r["human_score"].strip()]
        j = [float(r["judge_score"]) for r in rows]
        h = [float(r["human_score"]) for r in rows]
        agree = sum(a == b for a, b in zip(j, h)) / len(rows)
        print(f"{len(rows)} labelled | exact agreement {agree:.2f} | Cohen's kappa {cohen_kappa(j, h):.2f}")
        for r, a, b in zip(rows, j, h):
            if a != b:
                print(f"  disagree {r['id']}: judge={a} human={b}  {r['question'][:70]}")
