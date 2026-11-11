"""
Answer accuracy on MIRAGE, with and without retrieval.

    python scripts/download_benchmarks.py mirage medrag-textbooks
    # 1) no retrieval baseline (the model's own knowledge)
    python scripts/run_mirage.py --config configs/mirage/cot.yaml --per-dataset 100
    # 2) RAG over the MedRAG Textbooks corpus
    python scripts/run_mirage.py --config configs/mirage/rag_bm25.yaml --per-dataset 100
    # 3) paired comparison on the same questions
    python scripts/compare_runs.py results/mirage_cot results/mirage_rag_bm25 --metrics correct

--per-dataset 100 samples 100 questions from each of the 5 datasets with a fixed seed, so every
config sees the SAME 500 questions. Results are written after every question; re-running the
same command resumes where it stopped (API errors / laptop sleep).
"""
import argparse
import json
import os
import time

import _path  # noqa: F401
from dotenv import load_dotenv

from ragkit.benchmarks.mirage import PROMPTS, format_question, load_medrag_corpus, load_mirage, parse_choice
from ragkit.eval.stats import bootstrap_ci
from ragkit.generate import format_sources
from ragkit.llm import get_llm
from ragkit.pipeline import RAGPipeline, load_config

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--benchmark", default="data/mirage/benchmark.json")
    ap.add_argument("--corpus", default="data/medrag/textbooks")
    ap.add_argument("--datasets", nargs="*", default=None, help="e.g. medqa pubmedqa (default: all)")
    ap.add_argument("--per-dataset", type=int, default=None, help="questions sampled per dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-docs", type=int, default=None, help="load only N corpus snippets (smoke test)")
    ap.add_argument("--retrieve-with-options", action="store_true",
                    help="also use the answer options as the search query (MIRAGE default: question only)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    load_dotenv()

    cfg = load_config(args.config)
    mode = cfg.get("mirage", {}).get("mode", "rag")
    out = args.out or os.path.join("results", f"mirage_{cfg['name']}")
    os.makedirs(out, exist_ok=True)
    items = load_mirage(args.benchmark, args.datasets, args.per_dataset, args.seed)
    print(f"{len(items)} questions from {sorted({i['dataset'] for i in items})} · mode={mode}")

    if mode == "cot":
        pipe, llm = None, get_llm(cfg["generation"]["llm"])
    else:
        t0 = time.time()
        chunks = load_medrag_corpus(args.corpus, args.max_docs)
        cfg["context"]["neighbours"] = 0
        pipe = RAGPipeline(cfg, chunks=chunks)
        llm = pipe.llm
        print(f"{len(chunks)} snippets indexed in {time.time() - t0:.0f}s")
    if llm is None:
        raise SystemExit("set generation.llm in the config (e.g. openai:gpt-4o-mini)")

    path = os.path.join(out, "per_question.jsonl")
    done = {}
    if os.path.exists(path):
        done = {r["id"]: r for r in map(json.loads, open(path))}
        print(f"resuming: {len(done)} already answered")
    with open(path, "a") as f:
        for n, item in enumerate(items, 1):
            if item["id"] in done:
                continue
            t0, calls0 = time.time(), llm.calls
            ctx = []
            if mode == "cot":
                system = PROMPTS["cot"]
            else:
                query = format_question(item) if args.retrieve_with_options else item["question"]
                ctx = pipe.retrieve(query).final
                system = PROMPTS[cfg.get("mirage", {}).get("prompt", "rag")].format(context=format_sources(ctx))
            raw = llm.complete(system, format_question(item), json_mode=True)
            pred = parse_choice(raw, item["options"])
            row = {"id": item["id"], "dataset": item["dataset"], "question": item["question"][:300],
                   "gold": item["answer"], "pred": pred, "correct": float(pred == item["answer"]),
                   "unparsed": pred is None, "context": [c.header or c.source for c in ctx],
                   "context_chars": sum(len(c.text) for c in ctx), "llm_calls": llm.calls - calls0,
                   "latency_s": round(time.time() - t0, 2), "raw": raw[:1500]}
            f.write(json.dumps(row) + "\n")
            f.flush()
            done[item["id"]] = row
            if n % 25 == 0 or n == len(items):
                acc = sum(r["correct"] for r in done.values()) / len(done)
                print(f"  {len(done)}/{len(items)}  accuracy so far {acc:.3f}")

    rows = [done[i["id"]] for i in items if i["id"] in done]
    summary = {"config": cfg["name"], "mode": mode, "llm": cfg["generation"]["llm"], "n": len(rows),
               "seed": args.seed, "per_dataset": args.per_dataset, "accuracy": {}, "usage": llm.usage()}
    groups = {"ALL": rows}
    for r in rows:
        groups.setdefault(r["dataset"], []).append(r)
    for g, rs in groups.items():
        m, lo, hi = bootstrap_ci([r["correct"] for r in rs])
        summary["accuracy"][g] = {"mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)], "n": len(rs)}
    summary["unparsed"] = sum(r["unparsed"] for r in rows)
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    lines = [f"# MIRAGE · {cfg['name']} ({mode}, {cfg['generation']['llm']})", "",
             "| dataset | n | accuracy | 95% CI |", "|---|---|---|---|"]
    lines += [f"| {g} | {v['n']} | {v['mean']:.3f} | {v['ci95'][0]:.3f} – {v['ci95'][1]:.3f} |"
              for g, v in summary["accuracy"].items()]
    lines += ["", f"unparsed answers: {summary['unparsed']} · LLM usage: {summary['usage']}"]
    open(os.path.join(out, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines) + f"\n\nresults in {out}/")
