"""Evaluation items. Pages are 0-based PDF page indices (same as PyPDFLoader / pypdf)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EvalItem:
    id: str
    question: str
    category: str = "general"
    gold_pages: List[int] = field(default_factory=list)
    reference_answer: str = ""
    in_scope: bool = True
    key_phrase: Optional[str] = None
    source: str = ""
    gold_docs: Dict[str, int] = field(default_factory=dict)   # BEIR-style: doc id -> graded relevance

    def is_relevant(self, chunk) -> bool:
        if self.gold_docs:
            return chunk.meta.get("doc_id", chunk.id) in self.gold_docs
        return bool(set(chunk.pages) & set(self.gold_pages))


def load_eval_set(path: str, categories: Optional[List[str]] = None, limit: Optional[int] = None) -> List[EvalItem]:
    with open(path) as f:
        raw = json.load(f)
    items = [EvalItem(**{k: v for k, v in r.items() if k in EvalItem.__dataclass_fields__}) for r in raw]
    if categories:
        items = [i for i in items if i.category in categories]
    return items[:limit] if limit else items
