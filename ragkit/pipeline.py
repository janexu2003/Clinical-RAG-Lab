"""
Config-driven RAG pipeline. Each stage records what it did so failures can be diagnosed.

  question
    -> query transforms        (rewrite / multi_query / decompose / hyde)
    -> retrieval (fetch_k)     (bm25 / lsa / dense / hybrid)      ranks recorded per query
    -> fusion across queries   (RRF)
    -> rerank                  (cross_encoder / llm) + optional section boost
    -> corrective check        (optional: LLM grades context, retries with rewritten queries)
    -> context assembly (k)    (neighbours, ordering, budget)
    -> generation              (prompt style)
    -> hallucination check     (offline / llm_claims) -> optional repair
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional

import yaml

from ragkit import hallucination as H
from ragkit.context import NeighbourIndex, assemble
from ragkit.corpus import build_corpus
from ragkit.generate import ABSTAIN, format_sources, generate, is_abstention
from ragkit.llm import LLM, get_llm
from ragkit.query import transform
from ragkit.rerank import make_reranker, section_boost
from ragkit.retrievers import make_retriever, rrf
from ragkit.types import AnswerTrace, Candidate, Chunk, HallucinationReport, RetrievalTrace

DEFAULTS = {
    "name": "unnamed",
    "corpus": {"pdf": "data/Medical_book.pdf", "chunking": "section", "chunk_size": 1000,
               "overlap": 150, "contextual_header": True},
    "retrieval": {"retrievers": ["bm25"], "fetch_k": 30, "k": 5},
    "query": {"transforms": []},
    "rerank": {"type": "none", "section_boost": False},
    "corrective": {"enabled": False, "max_retries": 1},
    "context": {"neighbours": 0, "order": "relevance", "max_chars": 12000},
    "generation": {"llm": "none", "prompt": "grounded"},
    "hallucination": {"checks": ["offline"], "repair": False, "judge_llm": None},
}

GRADE_PROMPT = """Question: {q}

Sources:
{sources}

Do these sources contain the information needed to answer the question?
JSON: {{"verdict": "sufficient|partial|insufficient", "missing": "<what is missing, or empty>"}}"""


def _merge(base: Dict, over: Dict) -> Dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str, overrides: Optional[Dict] = None) -> Dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return _merge(_merge(DEFAULTS, cfg), overrides or {})


class RAGPipeline:
    def __init__(self, cfg: Dict, chunks: Optional[List[Chunk]] = None, llm: Optional[LLM] = None,
                 judge_llm: Optional[LLM] = None):
        self.cfg = _merge(DEFAULTS, cfg)
        self.chunks = chunks if chunks is not None else build_corpus(self.cfg["corpus"])
        self.llm = llm if llm is not None else get_llm(self.cfg["generation"]["llm"])
        self.judge = judge_llm or get_llm(self.cfg["hallucination"].get("judge_llm")) or self.llm
        self.retriever = make_retriever(self.cfg["retrieval"]).fit(self.chunks)
        self.reranker = make_reranker(self.cfg["rerank"], self.llm)
        self.nidx = NeighbourIndex(self.chunks)

    @classmethod
    def from_yaml(cls, path: str, **kw) -> "RAGPipeline":
        return cls(load_config(path), **kw)

    # ------------------------------------------------------------------ retrieval
    def _search(self, queries: List[str], fetch_k: int) -> List[Candidate]:
        rankings = [self.retriever.rank(q, fetch_k) for q in queries]
        fused = rrf(rankings) if len(rankings) > 1 else rankings[0]
        cands = []
        for r, (idx, score) in enumerate(fused[:fetch_k]):
            c = Candidate(self.chunks[idx], score, {"retrieval": r + 1})
            cands.append(c)
        return cands

    def _rank(self, question: str, cands: List[Candidate]) -> List[Candidate]:
        if self.reranker:
            n = self.cfg["rerank"].get("top_n") or len(cands)
            cands = self.reranker(question, cands[:n]) + cands[n:]
            for r, c in enumerate(cands):
                c.ranks["rerank"] = r + 1
        if self.cfg["rerank"].get("section_boost"):
            cands = section_boost(question, cands)
            for r, c in enumerate(cands):
                c.ranks["section_boost"] = r + 1
        return cands

    def retrieve(self, question: str) -> RetrievalTrace:
        rc, notes = self.cfg["retrieval"], []
        queries = transform(question, self.cfg["query"].get("transforms", []), self.llm,
                            self.cfg["query"].get("corpus_description"))
        cands = self._rank(question, self._search(queries, rc["fetch_k"]))
        final = assemble([c.chunk for c in cands], rc["k"], self.nidx, self.cfg["context"]["neighbours"],
                         self.cfg["context"]["order"], self.cfg["context"]["max_chars"])

        cor = self.cfg["corrective"]
        if cor.get("enabled") and self.llm is not None:
            for attempt in range(cor.get("max_retries", 1)):
                g = self.llm.complete_json("You judge retrieval quality. Reply with JSON only.",
                                           GRADE_PROMPT.format(q=question, sources=format_sources(final)))
                verdict = g.get("verdict", "partial")
                notes.append(f"grade[{attempt}]={verdict}")
                if verdict == "sufficient":
                    break
                extra = transform(question + (f" (missing: {g.get('missing')})" if g.get("missing") else ""),
                                  ["rewrite", "decompose"], self.llm,
                                  self.cfg["query"].get("corpus_description"))[1:]
                if not extra:
                    break
                queries += [q for q in extra if q not in queries]
                cands = self._rank(question, self._search(queries, rc["fetch_k"]))
                final = assemble([c.chunk for c in cands], rc["k"], self.nidx,
                                 self.cfg["context"]["neighbours"], self.cfg["context"]["order"],
                                 self.cfg["context"]["max_chars"])
        return RetrievalTrace(question, queries, cands, final, notes)

    # ------------------------------------------------------------------ answer
    def answer(self, question: str) -> AnswerTrace:
        if self.llm is None:
            raise RuntimeError("generation.llm is 'none' - set e.g. openai:gpt-4o-mini or ollama:llama3.1:8b")
        calls0 = self.llm.calls + (self.judge.calls if self.judge is not self.llm else 0)
        rt = self.retrieve(question)
        prompt = self.cfg["generation"]["prompt"]
        raw, extra = generate(question, rt.final, self.llm, prompt)
        hc = self.cfg["hallucination"]
        rep = H.check(raw, rt.final, self.judge, hc.get("checks", []), extra.get("quotes"))
        final = raw
        if hc.get("repair") and not is_abstention(raw):
            final, rep = H.repair(question, raw, rt.final, self.llm, rep, hc.get("checks", []))
        calls = self.llm.calls + (self.judge.calls if self.judge is not self.llm else 0) - calls0
        return AnswerTrace(question, rt, raw, final, is_abstention(final), rep, calls, extra)
