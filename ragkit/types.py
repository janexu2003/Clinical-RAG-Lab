"""Plain data classes used everywhere. No framework objects, so every step is easy to inspect."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Chunk:
    id: str
    text: str                      # body text shown to the LLM
    header: str = ""               # e.g. "Acne | Diagnosis" (contextual header, may be empty)
    source: str = ""
    entry: str = ""
    section: str = ""
    page: int = 0                  # 0-based PDF page where the chunk starts
    page_end: int = 0              # 0-based PDF page where it ends
    chunk_index: int = 0           # position inside its (entry, section)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def content(self) -> str:
        """What gets embedded / indexed: header + body."""
        return f"{self.header}\n{self.text}" if self.header else self.text

    @property
    def pages(self) -> List[int]:
        return list(range(self.page, max(self.page, self.page_end) + 1))

    def cite_label(self) -> str:
        where = " - ".join(x for x in (self.entry, self.section) if x)
        return f"{where or self.source}, PDF p.{self.page + 1}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Chunk":
        return cls(**d)


@dataclass
class Candidate:
    """A chunk on its way through the retrieval pipeline, with the rank it had at each stage."""
    chunk: Chunk
    score: float = 0.0
    ranks: Dict[str, Optional[int]] = field(default_factory=dict)   # stage/retriever -> 1-based rank


@dataclass
class RetrievalTrace:
    question: str
    queries: List[str]
    candidates: List[Candidate]          # after fusion / rerank, before cut-off
    final: List[Chunk]                   # what is sent to the LLM
    notes: List[str] = field(default_factory=list)


@dataclass
class HallucinationReport:
    faithfulness: Optional[float] = None         # supported claims / all claims (LLM check)
    claims: List[Dict[str, Any]] = field(default_factory=list)
    flags: List[Dict[str, Any]] = field(default_factory=list)   # cheap offline checks
    repaired: bool = False

    @property
    def unsupported_claims(self) -> List[Dict[str, Any]]:
        return [c for c in self.claims if c.get("verdict") != "supported"]


@dataclass
class AnswerTrace:
    question: str
    retrieval: RetrievalTrace
    raw_answer: str
    answer: str
    abstained: bool
    hallucination: HallucinationReport
    llm_calls: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)
