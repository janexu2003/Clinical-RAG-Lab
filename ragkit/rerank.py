"""
Rerankers: fix the ORDER when the right chunk is retrieved but not near the top.

    cross_encoder - reads (question, chunk) together; much more accurate than embeddings,
                    too slow for the whole corpus, perfect for the top 20-50 candidates
    llm           - asks the LLM to grade relevance 0-3 for each candidate (no extra model)
    section_boost - zero-cost rules: "How is X diagnosed?" prefers Diagnosis sections
"""
from __future__ import annotations

import re
from typing import List, Optional

from ragkit.llm import LLM
from ragkit.types import Candidate

# ---- section intent rules (hand-written; generic medical question words only) ----
SECTION_INTENTS = [
    (r"\b(diagnos\w*|tests? for|detect\w*)\b", {"Diagnosis", "Normal results", "Abnormal results"}),
    (r"\b(treat\w*|therap\w*|manag\w*|cure\w*|medications?|drugs?)\b", {"Treatment", "Alternative treatment", "Recommended dosage"}),
    (r"\b(caus\w*|symptoms?|signs?|risk factors?)\b", {"Causes and symptoms", "Causes", "Symptoms", "Description"}),
    (r"\b(prognos\w*|outlook|surviv\w*)\b", {"Prognosis"}),
    (r"\b(prevent\w*)\b", {"Prevention"}),
    (r"\b(prepar\w*)\b", {"Preparation"}),
    (r"\b(side effects?|adverse)\b", {"Side effects", "Risks", "Interactions"}),
    (r"\b(risks?|complications?)\b", {"Risks", "Precautions"}),
    (r"\b(aftercare)\b", {"Aftercare"}),
    (r"\b(used (for|to)|purpose)\b", {"Purpose"}),
]
DEFINITION_RE = re.compile(r"^\s*(what\s+(is|are)\b|define\b)", re.I)


def section_intent(question: str) -> set:
    hits = set()
    for pat, secs in SECTION_INTENTS:
        if re.search(pat, question, re.I):
            hits |= secs
    if not hits and DEFINITION_RE.search(question):
        hits = {"Definition", "Description"}
    return hits


def section_boost(question: str, cands: List[Candidate], weight: float = 0.5) -> List[Candidate]:
    intents = section_intent(question)
    if not intents:
        return cands
    scored = []
    for r, c in enumerate(cands):
        base = 1.0 / (r + 1)
        bonus = weight * base ** 0.5 if c.chunk.section in intents else 0.0
        scored.append((base + bonus, r, c))
    return [c for _, _, c in sorted(scored, key=lambda x: (-x[0], x[1]))]


class CrossEncoderReranker:
    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model)

    def __call__(self, question: str, cands: List[Candidate]) -> List[Candidate]:
        if not cands:
            return cands
        scores = self.model.predict([(question, c.chunk.content) for c in cands])
        for c, s in zip(cands, scores):
            c.score = float(s)
        return sorted(cands, key=lambda c: -c.score)


LLM_RERANK_PROMPT = """Question: {q}

Rate how useful each passage is for answering the question:
3 = directly answers it, 2 = partly answers it, 1 = same topic but does not answer, 0 = unrelated.

{passages}

JSON: {{"scores": {{"<passage number>": <0-3>, ...}}}}"""


class LLMReranker:
    def __init__(self, llm: LLM, max_chars: int = 600):
        self.llm, self.max_chars = llm, max_chars

    def __call__(self, question: str, cands: List[Candidate]) -> List[Candidate]:
        if not cands:
            return cands
        passages = "\n\n".join(f"[{i}] {c.chunk.content[: self.max_chars]}" for i, c in enumerate(cands, 1))
        out = self.llm.complete_json("You grade passage relevance. Reply with JSON only.",
                                     LLM_RERANK_PROMPT.format(q=question, passages=passages))
        scores = out.get("scores", {}) if isinstance(out.get("scores"), dict) else {}
        for i, c in enumerate(cands, 1):
            try:
                c.score = float(scores.get(str(i), 0)) - i * 1e-3   # ties keep retrieval order
            except (TypeError, ValueError):
                c.score = -i * 1e-3
        return sorted(cands, key=lambda c: -c.score)


def make_reranker(cfg: dict, llm: Optional[LLM]):
    kind = (cfg or {}).get("type", "none")
    if kind == "none":
        return None
    if kind == "cross_encoder":
        return CrossEncoderReranker(cfg.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    if kind == "llm":
        if llm is None:
            raise ValueError("llm reranker needs an llm")
        return LLMReranker(llm)
    raise ValueError(f"unknown reranker {kind}")
