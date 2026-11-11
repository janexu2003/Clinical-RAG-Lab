"""
Hallucination detection and repair.

A RAG hallucination = a statement in the answer that the retrieved sources do not support
(even if it happens to be medically true - in clinical use, unsupported is unsafe).

Three layers, from free to expensive:

  1. Offline checks (no LLM, milliseconds)
     - numbers / doses / percentages in the answer that appear in no source
     - citations: sentences without [n], or [n] pointing to a source that doesn't exist
     - lexical support: content words of a sentence vs. the sources it cites
     - quote check: "quote_then_answer" quotes must be exact substrings of the sources
  2. LLM claim verification (one call)
     - split the answer into atomic claims, label each supported / not_found / contradicted
     - faithfulness = supported / total  (same idea as RAGAS faithfulness)
  3. Repair
     - regenerate once with the list of problems; if claims are still unsupported,
       delete those sentences and tell the user something was removed
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from ragkit.generate import ABSTAIN, REGENERATE_PROMPT, format_sources, is_abstention
from ragkit.llm import LLM
from ragkit.retrievers import tokenize
from ragkit.types import Chunk, HallucinationReport

CITE_RE = re.compile(r"\[(\d+)\]")
NUM_RE = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)\s*(%|percent|mg|g|kg|ml|mcg|µg|units?|mmhg|hours?|days?|weeks?|months?|years?)?", re.I)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\-•*])|\n+")
NUMBER_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
                "seven": "7", "eight": "8", "nine": "9", "ten": "10", "twelve": "12"}


def sentences(answer: str) -> List[str]:
    return [s.strip() for s in SENT_SPLIT_RE.split(answer or "") if len(s.strip()) > 3]


def _digits(text: str) -> str:
    """'four to five months' -> '4 to 5 months' so words and digits compare equal."""
    for w, d in NUMBER_WORDS.items():
        text = re.sub(rf"\b{w}\b", d, text, flags=re.I)
    return text


def _unit(u: str) -> str:
    u = (u or "").lower()
    u = {"percent": "%", "units": "unit", "hours": "hour", "days": "day", "weeks": "week",
         "months": "month", "years": "year"}.get(u, u)
    return u


def _numbers(text: str) -> set:
    """Set of (number, unit) pairs; unit is '' when the number stands alone."""
    text = _digits(text)
    out = set()
    for m in NUM_RE.finditer(text):
        num = m.group(1).replace(",", "")
        out.add((num, _unit(m.group(2))))
    # ranges like "4 to 5 months" / "2-4 weeks": the unit applies to both numbers
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:to|-|–)\s*(\d+(?:\.\d+)?)\s*(%|percent|mg|hours?|days?|weeks?|months?|years?)", text, re.I):
        out.add((m.group(1), _unit(m.group(3))))
    return out


# --------------------------------------------------------------------------- #
# 1. Offline checks
# --------------------------------------------------------------------------- #
def numeric_check(answer: str, sources: List[Chunk]) -> List[Dict]:
    """Numbers (doses, durations, percentages) are the most dangerous thing to invent."""
    src = _numbers(" ".join(c.text for c in sources))
    src_plain = {n for n, _ in src}
    flags = []
    for s in sentences(answer):
        for num, unit in sorted(_numbers(CITE_RE.sub("", s))):
            # with a unit the exact (number, unit) pair must exist; "20 weeks" is not "20%"
            ok = (num, unit) in src if unit else num in src_plain
            if not ok:
                flags.append({"type": "number_not_in_sources", "value": f"{num} {unit}".strip(), "sentence": s})
    return flags


def citation_check(answer: str, n_sources: int) -> List[Dict]:
    flags = []
    if is_abstention(answer):
        return flags
    for s in sentences(answer):
        cites = [int(x) for x in CITE_RE.findall(s)]
        if not cites and not s.endswith(":") and len(s.split()) > 4 \
                and not re.search(r"\b(not covered|do not (cover|mention|address))\b", s, re.I):
            flags.append({"type": "uncited_sentence", "sentence": s})
        for n in cites:
            if n < 1 or n > n_sources:
                flags.append({"type": "invalid_citation", "value": n, "sentence": s})
    return flags


def lexical_support(answer: str, sources: List[Chunk], threshold: float = 0.5) -> List[Dict]:
    """Share of a sentence's content words found in the sources it cites (or all sources)."""
    flags = []
    for s in sentences(answer):
        cites = [int(x) for x in CITE_RE.findall(s) if 0 < int(x) <= len(sources)]
        pool = [sources[i - 1] for i in cites] or sources
        words = set(tokenize(CITE_RE.sub("", s)))
        if len(words) < 4:
            continue
        src_words = set(tokenize(" ".join(c.content for c in pool)))
        support = len(words & src_words) / len(words)
        if support < threshold:
            flags.append({"type": "low_lexical_support", "support": round(support, 2), "sentence": s})
    return flags


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[“”\"'’]", "", t)).strip().lower()


def quote_check(quotes: List[Dict], sources: List[Chunk]) -> List[Dict]:
    flags = []
    for q in quotes or []:
        try:
            n = int(q.get("source"))
        except (TypeError, ValueError):
            n = -1
        text = _norm(str(q.get("quote", "")))
        if not text:
            continue
        ok = 0 < n <= len(sources) and text in _norm(sources[n - 1].text)
        if not ok and any(text in _norm(c.text) for c in sources):
            flags.append({"type": "quote_wrong_source", "quote": q.get("quote"), "source": n})
        elif not ok:
            flags.append({"type": "fabricated_quote", "quote": q.get("quote"), "source": n})
    return flags


def offline_checks(answer: str, sources: List[Chunk], quotes: Optional[List[Dict]] = None) -> List[Dict]:
    if is_abstention(answer):
        return []
    return (numeric_check(answer, sources) + citation_check(answer, len(sources))
            + lexical_support(answer, sources) + quote_check(quotes or [], sources))


# --------------------------------------------------------------------------- #
# 2. LLM claim verification
# --------------------------------------------------------------------------- #
VERIFY_PROMPT = """You are checking a medical answer against its sources.

Sources:
{sources}

Answer:
{answer}

Split the answer into short atomic factual claims (ignore citations, greetings, and statements
that something is not covered). For each claim decide:
  "supported"    - stated or directly implied by at least one source
  "not_found"    - the sources do not say this (even if it is medically true)
  "contradicted" - a source says something different
JSON: {{"claims": [{{"claim": "...", "verdict": "supported|not_found|contradicted", "source": <n or null>}}]}}"""


def verify_claims(answer: str, sources: List[Chunk], llm: LLM) -> Tuple[Optional[float], List[Dict]]:
    if is_abstention(answer):
        return None, []
    out = llm.complete_json("You are a strict fact checker. Reply with JSON only.",
                            VERIFY_PROMPT.format(sources=format_sources(sources), answer=answer))
    claims = [c for c in out.get("claims", []) if isinstance(c, dict) and c.get("claim")]
    if not claims:
        return None, []
    for c in claims:
        c["verdict"] = str(c.get("verdict", "not_found")).lower().strip()
    supported = sum(c["verdict"] == "supported" for c in claims)
    return supported / len(claims), claims


# --------------------------------------------------------------------------- #
# 3. Check + repair
# --------------------------------------------------------------------------- #
def check(answer: str, sources: List[Chunk], llm: Optional[LLM], methods: List[str],
          quotes: Optional[List[Dict]] = None) -> HallucinationReport:
    rep = HallucinationReport()
    if any(m in methods for m in ("numbers", "citations", "lexical", "quotes", "offline")):
        flags = offline_checks(answer, sources, quotes)
        wanted = {"numbers": {"number_not_in_sources"},
                  "citations": {"uncited_sentence", "invalid_citation"},
                  "lexical": {"low_lexical_support"},
                  "quotes": {"fabricated_quote", "quote_wrong_source"}}
        keep = set().union(*[wanted.get(m, set()) for m in methods]) if "offline" not in methods else None
        rep.flags = [f for f in flags if keep is None or f["type"] in keep]
    if "llm_claims" in methods and llm is not None:
        rep.faithfulness, rep.claims = verify_claims(answer, sources, llm)
    return rep


def _problems(rep: HallucinationReport) -> List[str]:
    probs = [f"- not supported: {c['claim']}" for c in rep.unsupported_claims]
    probs += [f"- {f['type']}: {f.get('value') or f.get('quote') or f.get('sentence')}"
              for f in rep.flags if f["type"] in {"number_not_in_sources", "invalid_citation", "fabricated_quote"}]
    return probs


def repair(question: str, answer: str, sources: List[Chunk], llm: LLM, rep: HallucinationReport,
           methods: List[str]) -> Tuple[str, HallucinationReport]:
    problems = _problems(rep)
    if not problems:
        return answer, rep
    from ragkit.generate import PROMPTS
    system = PROMPTS["grounded"].format(context=format_sources(sources))
    new = llm.complete(system, f"Question: {question}\n\n" + REGENERATE_PROMPT.format(
        problems="\n".join(problems), answer=answer, abstain=ABSTAIN)).strip()
    new_rep = check(new, sources, llm, methods)
    still = new_rep.unsupported_claims
    if still:
        # last resort: drop sentences that contain an unsupported claim's key words
        bad_words = [set(tokenize(c["claim"])) for c in still]
        kept = [s for s in sentences(new)
                if not any(len(bw & set(tokenize(s))) >= max(2, int(0.6 * len(bw))) for bw in bad_words)]
        new = " ".join(kept) if kept else ABSTAIN
        if kept:
            new += "\n\n(Some statements were removed because the sources did not support them.)"
        new_rep = check(new, sources, llm, methods)
    new_rep.repaired = True
    return new, new_rep
