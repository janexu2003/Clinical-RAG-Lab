"""
Ask one question and see EVERYTHING: queries, retrieved chunks with ranks per stage,
the answer, and the hallucination check.

    python scripts/ask.py --config configs/improved.yaml "How do doctors tell if someone has acne?"
    python scripts/ask.py --config configs/offline_improved.yaml --retrieval-only "lazy eye treatment"
"""
import argparse

import _path  # noqa: F401
from dotenv import load_dotenv

from ragkit.pipeline import RAGPipeline, load_config

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--config", default="configs/improved.yaml")
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--show", type=int, default=8, help="candidates to print")
    args = ap.parse_args()
    load_dotenv()
    cfg = load_config(args.config)
    if args.retrieval_only:
        cfg["generation"]["llm"] = "none"
    pipe = RAGPipeline(cfg)

    rt = pipe.retrieve(args.question) if pipe.llm is None else None
    at = None if rt else pipe.answer(args.question)
    rt = rt or at.retrieval
    print("QUERIES:")
    for q in rt.queries:
        print("  -", q[:150])
    if rt.notes:
        print("NOTES:", rt.notes)
    print(f"\nTOP CANDIDATES (ranks per stage):")
    for c in rt.candidates[: args.show]:
        ch = c.chunk
        print(f"  {c.ranks}  {ch.entry or ch.source} | {ch.section} | p{ch.page}-{ch.page_end}")
        print(f"      {ch.text[:160]}...")
    print(f"\nCONTEXT SENT TO THE LLM: {len(rt.final)} chunks")
    for i, ch in enumerate(rt.final, 1):
        print(f"  [{i}] {ch.cite_label()}")
    if at:
        print("\nANSWER:\n" + at.answer)
        if at.raw_answer != at.answer:
            print("\n(RAW ANSWER BEFORE REPAIR):\n" + at.raw_answer)
        h = at.hallucination
        print(f"\nHALLUCINATION CHECK: faithfulness={h.faithfulness} repaired={h.repaired}")
        for c in h.claims:
            print(f"  {c['verdict']:13s} {c['claim']}")
        for f in h.flags:
            print(f"  FLAG {f['type']}: {f.get('value') or f.get('quote') or f.get('sentence', '')[:100]}")
        print(f"\nLLM calls: {at.llm_calls}")
