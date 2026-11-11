"""
Show what was retrieved vs what should have been retrieved for BEIR queries.

    # one query, two runs side by side
    python scripts/show_beir_query.py --dataset data/beir/nfcorpus \
        --runs results/beir_nfcorpus_hybrid results/beir_nfcorpus_hybrid_rerank --query PLAIN-3302
    # search queries by text
    python scripts/show_beir_query.py ... --find "fish"
    # the N queries where run 2 lost / gained the most nDCG@10 against run 1
    python scripts/show_beir_query.py ... --worst 5      (or --best 5)

Marks: [2] / [1] = graded relevant document (2 = strongest), [ ] = not relevant.
"""
import argparse
import csv
import json
import os


def load(args):
    titles = {}
    with open(os.path.join(args.dataset, "corpus.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            titles[d["_id"]] = d.get("title") or d.get("text", "")[:80]
    queries = {}
    with open(os.path.join(args.dataset, "queries.jsonl")) as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]
    qrels = {}
    with open(os.path.join(args.dataset, "qrels", f"{args.split}.tsv")) as f:
        for row in csv.reader(f, delimiter="\t"):
            if row and row[0] != "query-id" and int(float(row[2])) > 0:
                qrels.setdefault(row[0], {})[row[1]] = int(float(row[2]))
    runs = []
    for r in args.runs:
        with open(os.path.join(r, "per_question.jsonl")) as f:
            runs.append((os.path.basename(r.rstrip("/")), {x["id"]: x for x in map(json.loads, f)}))
    return titles, queries, qrels, runs


def show(qid, titles, queries, qrels, runs, width):
    rel = qrels.get(qid, {})
    print("=" * 110)
    print(f"{qid}: {queries.get(qid)!r}   ({len(rel)} relevant documents in the labels)")
    for name, rows in runs:
        r = rows.get(qid)
        if not r:
            continue
        print(f"\n  {name}:  nDCG@10={r['ndcg@10']:.3f}  MRR@10={r['mrr@10']:.3f}  "
              f"relevant in top 10: {sum(d in rel for d in r['top_docs'])}/10   cause: {r.get('cause')}")
        for i, d in enumerate(r["top_docs"], 1):
            mark = f"[{rel[d]}]" if d in rel else "[ ]"
            print(f"   {i:2d}. {mark} {d:10s} {titles.get(d, '?')[:width]}")
    found = set().union(*[set(rows[qid]["top_docs"]) for _, rows in runs if qid in rows])
    missed = sorted((d for d in rel if d not in found), key=lambda d: -rel[d])
    print(f"\n  SHOULD HAVE BEEN RETRIEVED (relevant, not in any top 10 above): {len(missed)}")
    for d in missed[:10]:
        print(f"       [{rel[d]}] {d:10s} {titles.get(d, '?')[:width]}")
    if len(missed) > 10:
        print(f"       ... and {len(missed) - 10} more")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--query", nargs="*", default=[])
    ap.add_argument("--find", default=None)
    ap.add_argument("--worst", type=int, default=0, help="queries where the LAST run lost most vs the first")
    ap.add_argument("--best", type=int, default=0, help="queries where the LAST run gained most vs the first")
    ap.add_argument("--width", type=int, default=85)
    a = ap.parse_args()
    titles, queries, qrels, runs = load(a)
    ids = list(a.query)
    if a.find:
        ids += [q for q in runs[0][1] if a.find.lower() in queries.get(q, "").lower()]
    if (a.worst or a.best) and len(runs) >= 2:
        first, last = runs[0][1], runs[-1][1]
        diff = sorted((last[q]["ndcg@10"] - first[q]["ndcg@10"], q) for q in first if q in last)
        ids += [q for _, q in diff[: a.worst]] + [q for _, q in diff[::-1][: a.best]]
    for q in ids:
        show(q, titles, queries, qrels, runs, a.width)
