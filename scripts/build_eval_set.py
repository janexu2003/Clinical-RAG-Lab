"""
Grow the eval set with LLM-written questions (then REVIEW them by hand).

For a stratified sample of chunks (spread over entries and section types), the LLM writes
one question a student/clinician would ask, in their own words, plus a reference answer.
Gold = the chunk's pages. Output is a CSV for review + a JSON eval set.

    python scripts/build_eval_set.py --config configs/offline_improved.yaml --n 60
    # open eval_data/generated_review.csv, set keep=0 for bad rows, then:
    python scripts/build_eval_set.py --from-review eval_data/generated_review.csv

Why review: generated questions are often answerable from several places (wrong gold),
too literal (copy the chunk's wording, so retrieval looks too easy), or ambiguous.
"""
import argparse
import csv
import json
import random
from collections import defaultdict

import _path  # noqa: F401
from dotenv import load_dotenv

from ragkit.corpus import build_corpus
from ragkit.llm import get_llm
from ragkit.pipeline import load_config

PROMPT = """Excerpt from a medical encyclopedia ({entry} - {section}):

{text}

Write ONE question that this excerpt answers, as a {persona} would ask it.
- Use your own words; do NOT copy phrases or the section heading from the excerpt.
- Name the disease/test/drug (or describe it in lay terms if the persona is a patient).
- Also write a 1-3 sentence reference answer using only the excerpt.
JSON: {{"question": "...", "answer": "..."}}"""
PERSONAS = ["medical student", "nurse", "patient with no medical training"]
SECTIONS = {"Definition", "Causes and symptoms", "Diagnosis", "Treatment", "Prognosis", "Prevention",
            "Description", "Purpose", "Risks", "Preparation"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/offline_improved.yaml")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--llm", default="openai:gpt-4o-mini")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from-review", default=None)
    ap.add_argument("--out", default="eval_data/generated.json")
    args = ap.parse_args()

    if args.from_review:
        items = []
        with open(args.from_review) as f:
            for r in csv.DictReader(f):
                if r.get("keep", "1").strip() in ("1", "y", "yes", "true"):
                    items.append({"id": r["id"], "question": r["question"], "category": r["category"],
                                  "gold_pages": json.loads(r["gold_pages"]), "reference_answer": r["reference_answer"],
                                  "in_scope": True, "source": "llm-generated, human-reviewed"})
        json.dump(items, open(args.out, "w"), indent=1)
        print(f"wrote {len(items)} reviewed questions to {args.out}")
        raise SystemExit

    load_dotenv()
    llm = get_llm(args.llm)
    chunks = [c for c in build_corpus(load_config(args.config)["corpus"])
              if c.section in SECTIONS and len(c.text) > 300]
    by_entry = defaultdict(list)
    for c in chunks:
        by_entry[c.entry].append(c)
    rng = random.Random(args.seed)
    entries = rng.sample(sorted(by_entry), min(args.n, len(by_entry)))   # one chunk per entry: spread out
    rows = []
    for i, e in enumerate(entries):
        c = rng.choice(by_entry[e])
        persona = PERSONAS[i % len(PERSONAS)]
        out = llm.complete_json("You write evaluation questions. Reply with JSON only.",
                                PROMPT.format(entry=c.entry, section=c.section, text=c.text, persona=persona))
        if not out.get("question"):
            continue
        rows.append({"id": f"gen-{i+1:03d}", "keep": 1, "question": out["question"],
                     "category": "lay_wording" if persona.startswith("patient") else "generated",
                     "gold_pages": json.dumps(c.pages), "reference_answer": out.get("answer", ""),
                     "entry": c.entry, "section": c.section, "source_text": c.text[:500]})
    path = "eval_data/generated_review.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}. Review them (set keep=0), then run with --from-review.")
