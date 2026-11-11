"""
BEIR-format retrieval benchmarks (Thakur et al., 2021).

A BEIR dataset folder looks like:
    corpus.jsonl        {"_id": "MED-10", "title": "...", "text": "..."}
    queries.jsonl       {"_id": "PLAIN-3", "text": "..."}
    qrels/test.tsv      query-id <tab> corpus-id <tab> score     (header line first)

Medical datasets that fit the "small corpus" scope of this project:
    nfcorpus     ~3.6k nutrition/medical documents, 323 test queries, graded relevance (1-2)
    scifact      ~5.2k scientific abstracts, 300 test claims
    trec-covid   ~171k COVID-19 papers, 50 queries with many judged documents (larger)

Each document becomes one chunk (long documents can be split; the doc id is kept in
chunk.meta["doc_id"] and rankings are de-duplicated back to documents before scoring).
Metrics follow BEIR / trec_eval conventions so numbers are comparable with published ones:
nDCG@10 with graded (linear) gains, Recall@100, MRR@10.
"""
from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from ragkit.corpus import pack_sentences
from ragkit.eval.dataset import EvalItem
from ragkit.types import Chunk

Qrels = Dict[str, Dict[str, int]]   # query id -> {doc id: graded relevance}


def _jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_qrels(path: str) -> Qrels:
    qrels: Qrels = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or row[0] in ("query-id", "query_id", "qid"):
                continue
            qid, did, score = row[0], row[1], int(float(row[2]))
            qrels[qid][did] = score
    return dict(qrels)


def load_beir(folder: str, split: str = "test", split_long_docs: bool = False, max_chars: int = 2000,
              title_as_header: bool = True, limit_queries: Optional[int] = None) -> Tuple[List[Chunk], List[EvalItem]]:
    """Return (chunks, eval items). Only queries that have at least one relevant doc are kept."""
    name = os.path.basename(os.path.normpath(folder))
    chunks: List[Chunk] = []
    for d in _jsonl(os.path.join(folder, "corpus.jsonl")):
        did, title, text = str(d["_id"]), (d.get("title") or "").strip(), (d.get("text") or "").strip()
        if not text and not title:
            continue
        pieces = pack_sentences(text, max_chars, 200) if (split_long_docs and len(text) > max_chars) else [text]
        for j, piece in enumerate(pieces):
            chunks.append(Chunk(id=f"{did}#{j}", text=piece or title, header=title if title_as_header else "",
                                source=name, entry=title[:120], section="", chunk_index=j,
                                meta={"doc_id": did}))

    qrels = load_qrels(os.path.join(folder, "qrels", f"{split}.tsv"))
    queries = {str(q["_id"]): q["text"] for q in _jsonl(os.path.join(folder, "queries.jsonl"))}
    items = []
    for qid in sorted(qrels):
        rel = {d: s for d, s in qrels[qid].items() if s > 0}
        if not rel or qid not in queries:
            continue
        items.append(EvalItem(id=qid, question=queries[qid], category=name, gold_docs=rel))
    if limit_queries:
        items = items[:limit_queries]
    return chunks, items


def doc_ranking(chunks: List[Chunk]) -> List[str]:
    """Ranked chunks -> ranked unique document ids (first occurrence wins)."""
    seen, out = set(), []
    for c in chunks:
        did = c.meta.get("doc_id", c.id)
        if did not in seen:
            seen.add(did)
            out.append(did)
    return out


def ir_metrics(ranked_docs: List[str], rel: Dict[str, int], k_ndcg: int = 10, k_recall: int = 100,
               k_mrr: int = 10) -> Dict[str, float]:
    """trec_eval-style metrics for one query. rel = {doc_id: grade > 0}."""
    rel = {d: g for d, g in rel.items() if g > 0}
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked_docs[:k_ndcg]))
    ideal = sorted(rel.values(), reverse=True)[:k_ndcg]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    first = next((i for i, d in enumerate(ranked_docs[:k_mrr]) if d in rel), None)
    found = sum(1 for d in ranked_docs[:k_recall] if d in rel)
    return {
        f"ndcg@{k_ndcg}": dcg / idcg if idcg else 0.0,
        f"recall@{k_recall}": found / len(rel) if rel else 0.0,
        f"mrr@{k_mrr}": 1.0 / (first + 1) if first is not None else 0.0,
        "hit@10": float(any(d in rel for d in ranked_docs[:10])),
        "hit@1": float(bool(ranked_docs) and ranked_docs[0] in rel),
    }
