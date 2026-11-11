"""
Query transformations: fix the QUESTION when the corpus uses different words.

    rewrite     - one clearer, encyclopedia-style query (lay -> clinical terms, expand abbreviations)
    multi_query - 3 paraphrases, results fused (helps when you don't know which wording matches)
    decompose   - split multi-part questions ("causes AND treatment") into sub-queries
    hyde        - "Hypothetical Document Embeddings": write a fake encyclopedia paragraph that
                  would answer the question, and search with THAT (works well with dense retrievers)

All of them keep the original question as query #1, so they can only add evidence.
"""
from __future__ import annotations

from typing import List, Optional

from ragkit.llm import LLM

DEFAULT_CORPUS = ("a medical encyclopedia whose entries are named by disease, test or drug and have sections "
                  "such as Definition, Causes and symptoms, Diagnosis, Treatment, Prognosis, Prevention")
SYSTEM = "You help search a medical document collection. Reply with JSON only."

PROMPTS = {
    "rewrite": """Rewrite the question as ONE search query for {corpus}.
Use clinical terms (e.g. "high blood pressure" -> "hypertension"), expand abbreviations,
keep the key disease / drug / test names.
Question: {q}
JSON: {{"queries": ["..."]}}""",
    "multi_query": """Write 3 different search queries that could retrieve the answer to this question
from {corpus}. Vary the wording (lay vs clinical terms, synonyms).
Question: {q}
JSON: {{"queries": ["...", "...", "..."]}}""",
    "decompose": """If the question asks several things, split it into one search query per part
(max 3). If it asks one thing, return it unchanged.
Question: {q}
JSON: {{"queries": ["..."]}}""",
    "hyde": """Write a short paragraph (3-4 sentences) in the style of the documents in {corpus} that
would answer the question. It does not need to be correct; it is only used as a search query.
Question: {q}
JSON: {{"queries": ["<the paragraph>"]}}""",
}


def transform(question: str, kinds: List[str], llm: Optional[LLM], corpus: Optional[str] = None) -> List[str]:
    queries = [question]
    if not llm:
        return queries
    for kind in kinds:
        if kind not in PROMPTS:
            raise ValueError(f"unknown query transform {kind}")
        out = llm.complete_json(SYSTEM, PROMPTS[kind].format(q=question, corpus=corpus or DEFAULT_CORPUS))
        for q in out.get("queries", [])[:3]:
            if isinstance(q, str) and q.strip() and q.strip() not in queries:
                queries.append(q.strip())
    return queries
