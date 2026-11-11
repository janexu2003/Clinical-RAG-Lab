"""
Evaluate one configuration.

    python scripts/run_eval.py --config configs/offline_baseline.yaml --retrieval-only
    python scripts/run_eval.py --config configs/improved.yaml            # needs an LLM
    python scripts/run_eval.py --config configs/improved.yaml --limit 5  # quick smoke test

Writes results/<config name>/{per_question.jsonl, summary.json, report.md}
"""
import argparse
import os

import _path  # noqa: F401
from dotenv import load_dotenv

from ragkit.eval.dataset import load_eval_set
from ragkit.eval.runner import run_eval
from ragkit.pipeline import RAGPipeline, load_config

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval", default="eval_data/eval_set.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--categories", nargs="*", default=None)
    args = ap.parse_args()
    load_dotenv()

    cfg = load_config(args.config)
    if args.retrieval_only:
        cfg["generation"]["llm"] = "none"
        cfg["query"]["transforms"] = []          # LLM query transforms need an LLM
        cfg["corrective"]["enabled"] = False
        if cfg["rerank"].get("type") == "llm":
            cfg["rerank"]["type"] = "none"
    pipe = RAGPipeline(cfg)
    items = load_eval_set(args.eval, args.categories, args.limit)
    out = args.out or os.path.join("results", cfg["name"] + ("_retrieval" if args.retrieval_only else ""))
    summary = run_eval(pipe, items, out, retrieval_only=args.retrieval_only)
    print(open(os.path.join(out, "report.md")).read().split("## Questions that need attention")[0])
    print(f"Full report: {out}/report.md")
