"""
Corpus loading and chunking.

Two chunking strategies so you can measure the difference yourself:

* "fixed":   fixed-size character windows per page (what most tutorials do; your v1)
* "section": clean the page furniture away, split the book into (entry, section) blocks,
             pack sentences into chunks inside a block, and optionally prepend a
             contextual header "Acne | Diagnosis" to each chunk.

Small-corpus tip: chunking is the cheapest thing to fix and usually the biggest win.
Always *look at* 20 random chunks before tuning anything else (scripts/inspect_chunks.py).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ragkit.types import Chunk

# --------------------------------------------------------------------------- #
# Page cleaning (Gale Encyclopedia of Medicine layout; harmless on other PDFs)
# --------------------------------------------------------------------------- #
FOOTER_TITLE_RE = re.compile(r"^GALE ENCYCLOPEDIA OF MEDICINE 2\s*\S*\s*$")
FOOTER_PRINT_RE = re.compile(r"^GEM\s*-\s*\d{4}\s+to\s+\d{4}.*Page\s+\d+\s*$")
SINGLE_LETTER_RE = re.compile(r"^[A-Z]$")
CROSS_REF_RE = re.compile(r"^[A-Z][^.:;]{0,60}\ssee(\s+[A-Z][^.]*)?$")

SECTION_HEADINGS = {
    "Definition", "Description", "Purpose", "Precautions", "Preparation", "Aftercare",
    "Risks", "Normal results", "Abnormal results", "Causes and symptoms", "Causes",
    "Symptoms", "Diagnosis", "Treatment", "Alternative treatment", "Prognosis",
    "Prevention", "Interactions", "Recommended dosage", "Side effects",
    "Special conditions", "Origins", "KEY TERMS", "Resources",
}
DROP_SECTIONS = {"Resources"}   # bibliographies and addresses: noise for retrieval


def load_pdf_pages(path: str) -> List[str]:
    from pypdf import PdfReader
    return [(p.extract_text() or "") for p in PdfReader(path).pages]


def split_page(page_text: str) -> Tuple[List[str], List[str]]:
    """Return (body_lines, caption_lines) with running footers removed."""
    body, captions, state = [], [], "body"
    for s in (l.strip() for l in (page_text or "").split("\n")):
        if FOOTER_TITLE_RE.match(s):
            state = "running_head"
            continue
        if FOOTER_PRINT_RE.match(s):
            state = "body"
            continue
        if not s or SINGLE_LETTER_RE.match(s):
            continue
        if state == "running_head":
            state = "caption"
            continue
        (captions if state == "caption" else body).append(s)
    return body, captions


def normalise(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)      # "diffi-\ncult" -> "difficult"
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


# --------------------------------------------------------------------------- #
# Sentence packing (transparent replacement for a recursive text splitter)
# --------------------------------------------------------------------------- #
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"“])")


def pack_sentences(text: str, size: int, overlap: int) -> List[str]:
    """Greedy-pack whole sentences into chunks of <= size chars, carrying ~overlap chars over."""
    sents: List[str] = []
    for s in SENT_RE.split(text):
        while len(s) > size:                       # very long "sentence": hard split on a space
            cut = s.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size
            sents.append(s[:cut].strip())
            s = s[cut:].strip()
        if s:
            sents.append(s)
    chunks, cur = [], []
    for s in sents:
        if cur and len(" ".join(cur + [s])) > size:
            chunks.append(" ".join(cur))
            carry, total = [], 0
            for prev in reversed(cur):             # sentence-level overlap
                if total + len(prev) > overlap:
                    break
                carry.insert(0, prev)
                total += len(prev) + 1
            cur = carry
        cur.append(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def char_windows(text: str, size: int, overlap: int) -> List[str]:
    """Fixed character windows that break on whitespace (the classic baseline)."""
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            sp = text.rfind(" ", start + size // 2, end)
            end = sp if sp > 0 else end
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [o for o in out if o]


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #
@dataclass
class Section:
    entry: str
    section: str
    lines: List[Tuple[str, int]] = field(default_factory=list)   # (line, page)


def parse_sections(pages: List[str]) -> List[Section]:
    stream: List[Tuple[str, int]] = []
    captions: Dict[int, List[str]] = {}
    for i, p in enumerate(pages):
        body, caps = split_page(p)
        stream += [(l, i) for l in body]
        if caps:
            captions[i] = caps

    sections: List[Section] = []
    cur: Optional[Section] = None
    entry: Optional[str] = None
    entry_on_page: Dict[int, str] = {}
    for i, (line, page) in enumerate(stream):
        nxt = stream[i + 1][0] if i + 1 < len(stream) else ""
        if nxt == "Definition" and line not in SECTION_HEADINGS and len(line) < 80:
            entry, cur = line, None
            continue
        if entry is None:
            continue
        entry_on_page[page] = entry
        if line in SECTION_HEADINGS:
            cur = Section(entry, line)
            sections.append(cur)
            continue
        if CROSS_REF_RE.match(line):
            continue
        if cur is None:
            cur = Section(entry, "Overview")
            sections.append(cur)
        cur.lines.append((line, page))

    for page, caps in captions.items():
        if page in entry_on_page:
            sections.append(Section(entry_on_page[page], "Figure caption", [(c, page) for c in caps]))
    return sections


# --------------------------------------------------------------------------- #
# Chunkers
# --------------------------------------------------------------------------- #
def _cid(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


def fixed_chunks(pages: List[str], source: str, size: int = 500, overlap: int = 20,
                 clean: bool = False) -> List[Chunk]:
    chunks = []
    for i, p in enumerate(pages):
        text = "\n".join(split_page(p)[0]) if clean else p
        for j, piece in enumerate(char_windows(text, size, overlap)):
            chunks.append(Chunk(id=_cid(source, i, j, piece), text=piece, source=source,
                                page=i, page_end=i, chunk_index=j))
    return chunks


def section_chunks(pages: List[str], source: str, size: int = 1000, overlap: int = 150,
                   contextual_header: bool = True, min_chars: int = 80,
                   title: str = "Gale Encyclopedia of Medicine") -> List[Chunk]:
    chunks = []
    for sec in parse_sections(pages):
        if sec.section in DROP_SECTIONS or not sec.lines:
            continue
        # build text while remembering the page of every character offset
        text, marks = "", []
        for line, page in sec.lines:
            marks.append((len(text), page))
            text += line + "\n"
        body = normalise(text)
        if len(body) < min_chars:
            continue
        ratio = len(text) / max(1, len(body))

        def page_at(off: int) -> int:
            p = marks[0][1]
            for o, pg in marks:
                if o <= off * ratio:
                    p = pg
                else:
                    break
            return p

        cursor = 0
        for j, piece in enumerate(pack_sentences(body, size, overlap)):
            start = max(0, body.find(piece[:60], cursor))
            cursor = start + 1
            chunks.append(Chunk(
                id=_cid(source, sec.entry, sec.section, j, piece),
                text=piece,
                header=f"{title} | {sec.entry} | {sec.section}" if contextual_header else "",
                source=source, entry=sec.entry, section=sec.section,
                page=page_at(start), page_end=page_at(start + len(piece)), chunk_index=j,
            ))
    return chunks


def build_corpus(cfg: Dict, cache_dir: str = ".cache") -> List[Chunk]:
    """cfg = {"pdf": path, "chunking": "fixed"|"section", ...}. Cached on disk by config hash."""
    key = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    path = os.path.join(cache_dir, f"chunks_{key}.jsonl")
    if os.path.exists(path):
        with open(path) as f:
            return [Chunk.from_dict(json.loads(l)) for l in f]
    pages = load_pdf_pages(cfg["pdf"])
    source = os.path.basename(cfg["pdf"])
    if cfg.get("chunking", "section") == "fixed":
        chunks = fixed_chunks(pages, source, cfg.get("chunk_size", 500), cfg.get("overlap", 20),
                              clean=cfg.get("clean", False))
    else:
        chunks = section_chunks(pages, source, cfg.get("chunk_size", 1000), cfg.get("overlap", 150),
                                contextual_header=cfg.get("contextual_header", True))
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "w") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict()) + "\n")
    return chunks
