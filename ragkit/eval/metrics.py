"""
Metrics.

Retrieval (no LLM needed, run these first and often):
  hit@k      did ANY of the top-k chunks come from a gold page?
  recall@k   fraction of gold pages covered by the top-k
  mrr        1 / rank of the first relevant chunk (rewards putting it first)
  ndcg@k     rank-discounted gain; relevant chunks lower down count less
  gold_rank  rank of the first relevant chunk in the full candidate list (for diagnosis)

Generation (needs an LLM judge):
  correctness     judge compares answer to the reference answer: 1 / 0.5 / 0
  faithfulness    share of answer claims supported by the retrieved sources
  citation_rate   share of answer sentences that carry a [n] citation
  abstention      in-scope questions: should NOT abstain; out-of-scope: SHOULD abstain
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

from ragkit.hallucination import CITE_RE, sentences
from ragkit.llm import LLM
from ragkit.types import Chunk


def relevant(chunk: Chunk, gold_pages: Sequence[int]) -> bool:
    return bool(set(chunk.pages) & set(gold_pages))


def retrieval_metrics(chunks: List[Chunk], gold_pages: Sequence[int], ks=(1, 3, 5, 10)) -> Dict[str, float]:
    gold = set(gold_pages)
    rels = [relevant(c, gold) for c in chunks]
    out: Dict[str, float] = {}
    first = next((i for i, r in enumerate(rels) if r), None)
    out["mrr"] = 1.0 / (first + 1) if first is not None else 0.0
    for k in ks:
        top = chunks[:k]
        covered = set().union(*[set(c.pages) & gold for c in top]) if top else set()
        out[f"hit@{k}"] = float(any(rels[:k]))
        out[f"recall@{k}"] = len(covered) / len(gold) if gold else 0.0
        # gain only when a chunk covers a gold page not seen yet, so DCG <= ideal DCG
        seen, dcg = set(), 0.0
        for i, c in enumerate(top):
            new = (set(c.pages) & gold) - seen
            if new:
                dcg += 1.0 / math.log2(i + 2)
                seen |= new
        ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, max(1, len(gold)))))
        out[f"ndcg@{k}"] = dcg / ideal if ideal else 0.0
    return out


def gold_rank(chunks: List[Chunk], gold_pages: Sequence[int]) -> Optional[int]:
    for i, c in enumerate(chunks):
        if relevant(c, gold_pages):
            return i + 1
    return None


# --------------------------------------------------------------------------- generation
JUDGE_PROMPT = """You grade answers from a medical question-answering system.

Question: {q}
Reference answer: {ref}
System answer: {ans}

Score the system answer against the reference:
  1   = contains the key facts of the reference and nothing that contradicts it
  0.5 = partly correct: some key facts missing, or minor inaccuracies
  0   = wrong, contradicts the reference, or does not answer (e.g. says it doesn't know)
Extra correct detail beyond the reference is fine. Judge meaning, not wording.
JSON: {{"score": 1 | 0.5 | 0, "reason": "<one sentence>"}}"""


def judge_correctness(question: str, reference: str, answer: str, judge: LLM) -> Dict:
    out = judge.complete_json("You are a careful, strict grader. Reply with JSON only.",
                              JUDGE_PROMPT.format(q=question, ref=reference, ans=answer))
    try:
        score = float(out.get("score"))
    except (TypeError, ValueError):
        score = 0.0
    score = min([0.0, 0.5, 1.0], key=lambda x: abs(x - score))
    return {"score": score, "reason": out.get("reason", "")}


def citation_rate(answer: str) -> Optional[float]:
    sents = [s for s in sentences(answer) if len(s.split()) > 4]
    if not sents:
        return None
    return sum(bool(CITE_RE.search(s)) for s in sents) / len(sents)
