"""
MIRAGE (Xiong et al., 2024): 7,663 multiple-choice medical questions from five datasets
(MMLU-Med, MedQA-US, MedMCQA, PubMedQA*, BioASQ-Y/N), used to benchmark medical RAG.

benchmark.json:   {"<dataset>": {"<qid>": {"question": str, "options": {"A": str, ...}, "answer": "A"}}}

Retrieval corpus: the MedRAG "Textbooks" corpus (18 medical textbooks, ~126k snippets),
jsonl files with fields like {"id", "title", "content"} (the loader also accepts
"contents" / "text"). Multiple choice means accuracy needs no LLM judge, and results can
be put next to the MedRAG paper's tables.

Following MIRAGE's "question-only retrieval" setting, the options are NOT used for retrieval
by default (they would leak the answer candidates into the search).
"""
from __future__ import annotations

import json
import os
import random
import re
from typing import Dict, List, Optional

from ragkit.types import Chunk


def load_mirage(path: str, datasets: Optional[List[str]] = None, per_dataset: Optional[int] = None,
                seed: int = 0) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        bench = json.load(f)
    items = []
    for ds in sorted(bench):
        if datasets and ds not in datasets:
            continue
        qids = sorted(bench[ds])
        if per_dataset and per_dataset < len(qids):
            qids = sorted(random.Random(seed).sample(qids, per_dataset))   # fixed sample = fair comparisons
        for qid in qids:
            q = bench[ds][qid]
            items.append({"id": f"{ds}-{qid}", "dataset": ds, "question": q["question"].strip(),
                          "options": {str(k): str(v) for k, v in q["options"].items()},
                          "answer": str(q["answer"]).strip()})
    return items


def load_medrag_corpus(folder: str, max_docs: Optional[int] = None) -> List[Chunk]:
    """Read every *.jsonl under folder (MedRAG corpora are already chunked into snippets)."""
    chunks: List[Chunk] = []
    files = sorted(os.path.join(r, f) for r, _, fs in os.walk(folder) for f in fs if f.endswith(".jsonl"))
    if not files:
        raise FileNotFoundError(f"no .jsonl files under {folder} (run scripts/download_benchmarks.py medrag)")
    for path in files:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                d = json.loads(line)
                text = d.get("content") or d.get("text") or d.get("contents") or ""
                title = d.get("title", "")
                if not text:
                    continue
                cid = str(d.get("id") or f"{os.path.basename(path)}:{i}")
                chunks.append(Chunk(id=cid, text=text, header=title, source=os.path.basename(path),
                                    entry=title, meta={"doc_id": cid}))
                if max_docs and len(chunks) >= max_docs:
                    return chunks
    return chunks


def format_question(item: Dict) -> str:
    opts = "\n".join(f"{k}. {v}" for k, v in sorted(item["options"].items()))
    return f"{item['question']}\n\nOptions:\n{opts}"


JSON_SPEC = ('Respond in JSON: {"step_by_step_thinking": "<brief reasoning>", '
             '"answer_choice": "<one option letter>"}')

PROMPTS = {   # "cot" is used as-is; the rag prompts are filled with .format(context=...)
    # no retrieval: the model's own knowledge (MedRAG's "CoT" baseline)
    "cot": "You are a helpful medical expert. Answer the multiple-choice question. "
           "Think step by step, then choose exactly one option. " + JSON_SPEC,
    # retrieval-augmented: documents first, the model may still use its own knowledge
    "rag": "You are a helpful medical expert. Here are documents that may be relevant:\n\n{context}\n\n"
           "Answer the multiple-choice question. Use the documents where they are relevant, and your own "
           "medical knowledge otherwise. Think step by step, then choose exactly one option. "
           + JSON_SPEC.replace("{", "{{").replace("}", "}}"),     # escaped: this prompt is .format()-ed
    # retrieval-augmented, must say whether the documents were used (measures reliance on retrieval)
    "rag_cite": "You are a helpful medical expert. Here are numbered documents that may be relevant:\n\n"
                "{context}\n\nAnswer the multiple-choice question. Use the documents where relevant and your "
                "own knowledge otherwise. Respond in JSON: {{\"step_by_step_thinking\": \"<brief reasoning, "
                "citing [n] for any document you use>\", \"documents_used\": [<numbers>], "
                "\"answer_choice\": \"<one option letter>\"}}",
}


def parse_choice(text: str, options: Dict[str, str]) -> Optional[str]:
    letters = set(options)
    try:
        d = json.loads(re.search(r"\{.*\}", text or "", re.S).group(0))
        c = str(d.get("answer_choice", "")).strip().strip("().").upper()[:1]
        if c in letters:
            return c
    except (AttributeError, json.JSONDecodeError, TypeError):
        pass
    m = re.search(r"answer(?:_choice)?\"?\s*(?:is|:)?\s*\"?\(?([A-J])\b", text or "", re.I)
    if m and m.group(1).upper() in letters:
        return m.group(1).upper()
    found = [l for l in re.findall(r"\b([A-J])\b", text or "") if l in letters]
    return found[-1] if found else None
