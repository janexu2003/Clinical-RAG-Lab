"""
Failure diagnosis: WHY did a question fail? Each cause has a different fix.

For every question we follow the gold chunk through the pipeline:

  NOT_IN_CORPUS        no chunk overlaps the gold pages at all
                       -> data problem: add sources, or fix the PDF parsing / gold label
  RECALL_MISS          gold chunk not in the top `depth` of ANY retriever
                       -> vocabulary mismatch: query rewrite / multi-query / HyDE,
                          contextual headers, synonyms, a better or domain embedding model
  CANDIDATE_MISS       found by some retriever but outside fetch_k after fusion
                       -> raise fetch_k, add the retriever that found it (hybrid), tune weights
  RANKING_MISS         inside the candidate list (fetch_k) but not in the final top-k
                       -> reranker (cross-encoder / LLM), section boost, larger k
  RETRIEVED_OK         gold chunk is in the final context
                       -> if the answer is still wrong it is a GENERATION problem:
                          prompt, context order ("lost in the middle"), too much context,
                          neighbour expansion if the answer is split across chunks
  ANSWERED_OK          retrieval ok and the judge scored the answer >= 0.5
  OOS_CORRECT_ABSTAIN / OOS_HALLUCINATED   out-of-scope questions

The per-retriever ranks are printed too, which shows when retrievers complement each other
(e.g. BM25 rank 2 vs dense rank 140 for a rare drug name).
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

from ragkit.eval.dataset import EvalItem
from ragkit.pipeline import RAGPipeline
from ragkit.types import RetrievalTrace

FIXES = {
    "NOT_IN_CORPUS": "Add the missing source or fix parsing/labels; the retriever cannot find what is not indexed.",
    "RECALL_MISS": "Vocabulary mismatch: try query rewrite / multi_query / HyDE, contextual headers, or a domain embedding model.",
    "CANDIDATE_MISS": "One retriever finds it but fusion drops it: increase fetch_k, add that retriever to the hybrid, adjust weights.",
    "RANKING_MISS": "Retrieved but ranked too low: add a reranker (cross-encoder or LLM) or section boost, or raise k.",
    "RETRIEVED_OK": "Retrieval is fine. If the answer is wrong, fix generation: prompt, context order, context size.",
    "ANSWERED_OK": "Works.",
    "OOS_CORRECT_ABSTAIN": "Works: correctly declined.",
    "OOS_HALLUCINATED": "Answered a question the corpus cannot support: strengthen abstention (grounded prompt, claim check, corrective grading).",
    "OOS_NOT_EVALUATED": "Out-of-scope question; run with an LLM to check abstention.",
}


def diagnose_retrieval(pipe: RAGPipeline, item: EvalItem, trace: RetrievalTrace, depth: int = 200) -> Dict:
    if not item.in_scope:
        return {"id": item.id, "category": item.category, "question": item.question, "cause": "OOS_NOT_EVALUATED"}
    rel = item.is_relevant
    in_corpus = any(rel(c) for c in pipe.chunks)
    comp = {}
    for q in trace.queries:
        for name, ranks in pipe.retriever.component_ranks(q, depth).items():
            best = min((r for idx, r in ranks.items() if rel(pipe.chunks[idx])), default=None)
            if best is not None and (comp.get(name) is None or best < comp[name]):
                comp[name] = best
            comp.setdefault(name, None)
    cand_rank = next((i + 1 for i, c in enumerate(trace.candidates) if rel(c.chunk)), None)
    final_rank = next((i + 1 for i, c in enumerate(trace.final) if rel(c)), None)
    if not in_corpus:
        cause = "NOT_IN_CORPUS"
    elif final_rank is not None:
        cause = "RETRIEVED_OK"
    elif cand_rank is not None:
        cause = "RANKING_MISS"
    elif any(v is not None for v in comp.values()):
        cause = "CANDIDATE_MISS"
    else:
        cause = "RECALL_MISS"
    return {"id": item.id, "category": item.category, "question": item.question, "cause": cause,
            "component_ranks": comp, "candidate_rank": cand_rank, "final_rank": final_rank,
            "top_retrieved": [f"{c.meta.get('doc_id') or c.entry}|{c.section}|p{c.page}" for c in trace.final[:3]],
            "queries": trace.queries}


def refine_with_answer(diag: Dict, item: EvalItem, correctness: Optional[float], abstained: bool) -> Dict:
    if not item.in_scope:
        diag["cause"] = "OOS_CORRECT_ABSTAIN" if abstained else "OOS_HALLUCINATED"
    elif diag["cause"] == "RETRIEVED_OK" and correctness is not None and correctness >= 0.5:
        diag["cause"] = "ANSWERED_OK"
    return diag


def summarize_causes(diags: List[Dict]) -> Dict[str, int]:
    return dict(Counter(d["cause"] for d in diags).most_common())
