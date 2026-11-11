"""
Retrieval evaluation on a BEIR medical dataset (NFCorpus, SciFact, TREC-COVID).

    python scripts/download_benchmarks.py beir nfcorpus scifact
    python scripts/run_beir.py --dataset data/beir/nfcorpus --config configs/beir/bm25.yaml
    python scripts/run_beir.py --dataset data/beir/nfcorpus --config configs/beir/hybrid_rerank.yaml
    python scripts/compare_runs.py results/beir_nfcorpus_bm25 results/beir_nfcorpus_hybrid_rerank \
        --metrics ndcg@10 recall@100 mrr@10

Reports nDCG@10 (the standard BEIR headline metric), Recall@100 and MRR@10 with bootstrap
95% CIs, per-query results for paired comparison, and a failure diagnosis
(RECALL_MISS / CANDIDATE_MISS / RANKING_MISS) for the queries with nDCG@10 = 0.
"""
import argparse
import json
import os
import time

import _path  # noqa: F401
from dotenv import load_dotenv

from ragkit.benchmarks.beir import doc_ranking, ir_metrics, load_beir
from ragkit.eval.diagnose import FIXES, diagnose_retrieval
from ragkit.eval.stats import bootstrap_ci
from ragkit.pipeline import RAGPipeline, load_config

METRICS = ["ndcg@10", "recall@100", "mrr@10", "hit@1", "hit@10"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="folder with corpus.jsonl, queries.jsonl, qrels/")
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None, help="only the first N queries (smoke test)")
    ap.add_argument("--split-long-docs", action="store_true", help="chunk documents longer than 2000 chars")
    ap.add_argument("--no-diagnose", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    load_dotenv()

    cfg = load_config(args.config)
    cfg["retrieval"]["fetch_k"] = max(100, cfg["retrieval"].get("fetch_k", 100))   # Recall@100 needs 100
    cfg["context"]["neighbours"] = 0
    name = os.path.basename(os.path.normpath(args.dataset))
    t0 = time.time()
    chunks, items = load_beir(args.dataset, args.split, args.split_long_docs, limit_queries=args.limit)
    print(f"{name}: {len(chunks)} chunks, {len(items)} {args.split} queries with relevance labels")
    pipe = RAGPipeline(cfg, chunks=chunks)
    print(f"index built in {time.time() - t0:.0f}s")

    rows = []
    for n, item in enumerate(items, 1):
        trace = pipe.retrieve(item.question)
        ranked = doc_ranking([c.chunk for c in trace.candidates])
        row = {"id": item.id, "question": item.question, "n_relevant": len(item.gold_docs),
               **ir_metrics(ranked, item.gold_docs), "top_docs": ranked[:10], "queries": trace.queries}
        if not args.no_diagnose and row["ndcg@10"] == 0:
            d = diagnose_retrieval(pipe, item, trace)
            row["cause"] = d["cause"]
            row["component_ranks"] = d["component_ranks"]
        else:
            row["cause"] = "OK" if row["ndcg@10"] > 0 else "NOT_DIAGNOSED"
        rows.append(row)
        if n % 50 == 0 or n == len(items):
            mean = sum(r["ndcg@10"] for r in rows) / len(rows)
            print(f"  {n}/{len(items)} queries  running nDCG@10 = {mean:.3f}")

    out = args.out or os.path.join("results", f"beir_{name}_{cfg['name']}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "per_question.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = {"dataset": name, "config": cfg["name"], "n_queries": len(rows), "n_chunks": len(chunks),
               "seconds": round(time.time() - t0, 1), "metrics": {}, "causes": {}}
    for m in METRICS:
        mean, lo, hi = bootstrap_ci([r[m] for r in rows])
        summary["metrics"][m] = {"mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    for r in rows:
        summary["causes"][r["cause"]] = summary["causes"].get(r["cause"], 0) + 1
    if pipe.llm is not None:
        summary["llm_usage"] = pipe.llm.usage()
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)

    lines = [f"# BEIR {name} · {cfg['name']}", "", f"{len(rows)} queries · {len(chunks)} chunks · "
             f"{summary['seconds']}s", "", "| metric | mean | 95% CI |", "|---|---|---|"]
    lines += [f"| {m} | {v['mean']:.4f} | {v['ci95'][0]:.3f} – {v['ci95'][1]:.3f} |" for m, v in summary["metrics"].items()]
    lines += ["", "## Queries with nDCG@10 = 0", "", "| cause | count | what to do |", "|---|---|---|"]
    lines += [f"| {c} | {k} | {FIXES.get(c, '')} |" for c, k in summary["causes"].items() if c != "OK"]
    open(os.path.join(out, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nresults in {out}/")
