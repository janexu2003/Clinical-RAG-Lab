"""
Answer generation. Three prompt styles you can compare with the evaluation harness:

  v1_original        your original prompt ("three sentences max", "say you don't know")
  grounded           numbered sources, every sentence cites [n], partial answers allowed,
                     explicit abstention phrase
  quote_then_answer  the model first copies the exact sentences it relies on, THEN answers
                     from those quotes only. Quotes can be checked verbatim against the
                     sources with plain string matching - a cheap, strong hallucination guard.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from ragkit.llm import LLM
from ragkit.types import Chunk

ABSTAIN = "The provided documents do not cover this."
ABSTAIN_PATTERNS = re.compile(
    r"(do(es)? not cover this|don'?t know|do not know|not (have )?enough information|"
    r"cannot (be )?answer|no information (about|on)|not mentioned in the (provided )?(context|documents))",
    re.I)


def is_abstention(answer: str) -> bool:
    head = (answer or "").strip()[:200]
    return head.startswith(ABSTAIN) or bool(ABSTAIN_PATTERNS.search(head))


def format_sources(chunks: List[Chunk]) -> str:
    return "\n\n".join(f"[{i}] ({c.cite_label()})\n{c.text}" for i, c in enumerate(chunks, 1))


PROMPTS: Dict[str, str] = {
    "v1_original": (
        "You are an Medical assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise.\n\n{context}"
    ),
    "grounded": f"""You are a medical information assistant. Answer using ONLY the numbered sources below.

Rules:
1. Every sentence of your answer must end with the number(s) of the source(s) that support it, e.g. [2] or [1][3].
2. The answer may be spread over neighbouring sources; read all of them first.
3. If the sources answer only part of the question, answer that part and say clearly which part is not covered.
4. If the sources do not address the question at all, reply exactly: "{ABSTAIN}"
5. Never add facts, numbers, doses or drug names that are not in the sources, even if you know them.
6. Be concise and use plain language. Do not give personal medical advice.

Sources:
{{context}}""",
    "quote_then_answer": f"""You are a medical information assistant. Work in two steps.

Step 1 - Evidence: copy, word for word, the sentences from the sources that are needed to answer
the question (at most 6). Each quote must be an exact substring of one source.
Step 2 - Answer: answer the question using ONLY those quotes. End each sentence with the source
number(s) [n]. If no source contains relevant sentences, the answer must be exactly "{ABSTAIN}".
If only part of the question is covered, say which part is missing.

Return JSON: {{{{"quotes": [{{{{"source": <n>, "quote": "<exact text>"}}}}], "answer": "<answer with [n] citations>"}}}}

Sources:
{{context}}""",
}


def generate(question: str, chunks: List[Chunk], llm: LLM, prompt: str = "grounded") -> Tuple[str, Dict]:
    if prompt not in PROMPTS:
        raise ValueError(f"unknown prompt {prompt}")
    system = PROMPTS[prompt].format(context=format_sources(chunks))
    if prompt == "quote_then_answer":
        out = llm.complete_json(system, question)
        quotes = out.get("quotes", []) if isinstance(out.get("quotes"), list) else []
        return str(out.get("answer", "")).strip() or ABSTAIN, {"quotes": quotes}
    return llm.complete(system, question).strip(), {}


REGENERATE_PROMPT = """Your previous answer contained statements that the sources do not support:
{problems}

Rewrite the answer using ONLY information in the sources, with [n] citations on every sentence.
Remove anything you cannot support. If nothing remains, reply exactly: "{abstain}"

Previous answer:
{answer}"""
