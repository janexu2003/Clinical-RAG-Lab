"""
LOOK AT YOUR DATA. Prints random chunks plus statistics; most retrieval bugs are visible here.

    python scripts/inspect_chunks.py --config configs/offline_baseline.yaml --n 10
    python scripts/inspect_chunks.py --config configs/offline_improved.yaml --grep "lazy eye"
"""
import argparse
import random
import re
from collections import Counter

import _path  # noqa: F401

from ragkit.corpus import build_corpus
from ragkit.pipeline import load_config

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--grep", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    chunks = build_corpus(load_config(args.config)["corpus"])
    lens = sorted(len(c.text) for c in chunks)
    print(f"{len(chunks)} chunks | length min {lens[0]} / median {lens[len(lens)//2]} / max {lens[-1]}")
    tiny = [c for c in chunks if len(c.text) < 120]
    print(f"{len(tiny)} chunks shorter than 120 chars (often headers/footers = retrieval noise)")
    dup = Counter(c.text for c in chunks)
    print(f"{sum(v - 1 for v in dup.values() if v > 1)} exact duplicate chunks")
    if chunks[0].entry:
        print("sections:", Counter(c.section for c in chunks).most_common(8))
    pool = [c for c in chunks if re.search(args.grep, c.content, re.I)] if args.grep else chunks
    random.Random(args.seed).shuffle(pool)
    for c in pool[: args.n]:
        print("-" * 100)
        print(f"{c.entry or c.source} | {c.section} | p{c.page}-{c.page_end} | {len(c.text)} chars")
        print(c.content[:600])
