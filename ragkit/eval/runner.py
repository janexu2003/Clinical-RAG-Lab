"""Run one configuration over an eval set and write per-question results + a summary report."""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional

from ragkit.eval.dataset import EvalItem
from ragkit.eval.diagnose import FIXES, diagnose_retrieval, refine_with_answer, summarize_causes
from ragkit.eval.metrics import citation_rate, judge_correctness, relevant, retrieval_metrics
from ragkit.eval.stats import bootstrap_ci
from ragkit.llm import LLM, get_llm
from ragkit.pipeline import RAGPipeline

RETRIEVAL_KEYS = ["hit@1", "hit@3", "hit@5", "recall@5", "mrr", "ndcg@5", "context_hit", "context_chars"]
ANSWER_KEYS = ["correctness", "faithfulness", "citation_rate", "n_flags", "llm_calls", "latency_s"]


def run_eval(pipe: RAGPipeline, items: List[EvalItem], out_dir: str, retrieval_only: bool = False,
             judge: Optional[LLM] = None, verbose: bool = True) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    judge = judge or get_llm(pipe.cfg.get("eval", {}).get("judge_llm")) or pipe.judge or pipe.llm
    rows = []
    for n, item in enumerate(items, 1):
        t0 = time.time()
        row: Dict = {"id": item.id, "category": item.category, "question": item.question,
                     "in_scope": item.in_scope}
        if retrieval_only or pipe.llm is None:
            trace = pipe.retrieve(item.question)
            answer_trace = None
        else:
            answer_trace = pipe.answer(item.question)
            trace = answer_trace.retrieval
        if item.in_scope:
            # ranking quality: the ranked candidate list (before neighbour expansion)
            row.update(retrieval_metrics([c.chunk for c in trace.candidates], item.gold_pages))
            # what the LLM actually sees: is the gold page anywhere in the final context?
            row["context_hit"] = float(any(relevant(c, item.gold_pages) for c in trace.final))
        row["context_chars"] = sum(len(c.text) for c in trace.final)
        diag = diagnose_retrieval(pipe, item, trace)

        if answer_trace is not None:
            a = answer_trace
            row.update({"answer": a.answer, "raw_answer": a.raw_answer, "abstained": a.abstained,
                        "faithfulness": a.hallucination.faithfulness,
                        "unsupported_claims": [c["claim"] for c in a.hallucination.unsupported_claims],
                        "flags": a.hallucination.flags, "n_flags": len(a.hallucination.flags),
                        "repaired": a.hallucination.repaired, "citation_rate": citation_rate(a.answer),
                        "llm_calls": a.llm_calls})
            if item.in_scope and judge is not None:
                j = judge_correctness(item.question, item.reference_answer, a.answer, judge)
                row["correctness"], row["judge_reason"] = j["score"], j["reason"]
            diag = refine_with_answer(diag, item, row.get("correctness"), a.abstained)
        row["latency_s"] = round(time.time() - t0, 2)
        row["diagnosis"] = diag
        row["cause"] = diag["cause"]
        rows.append(row)
        if verbose:
            extra = f" correct={row.get('correctness')}" if "correctness" in row else ""
            print(f"[{n}/{len(items)}] {item.id:8s} {row['cause']:20s} hit@5={row.get('hit@5', '-')}{extra}")

    summary = summarize(rows, pipe.cfg)
    with open(os.path.join(out_dir, "per_question.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(pipe.cfg, f, indent=1)
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write(render_report(summary, rows))
    return summary


def _ci(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    m, lo, hi = bootstrap_ci(vals)
    return {"mean": round(m, 3), "ci95": [round(lo, 3), round(hi, 3)], "n": len(vals)}


def summarize(rows: List[Dict], cfg: Dict) -> Dict:
    ins = [r for r in rows if r["in_scope"]]
    oos = [r for r in rows if not r["in_scope"]]
    s: Dict = {"name": cfg.get("name"), "n_questions": len(rows), "n_in_scope": len(ins), "overall": {}}
    for k in RETRIEVAL_KEYS + ANSWER_KEYS:
        pool = rows if k in ("context_chars", "llm_calls", "latency_s", "n_flags", "citation_rate", "faithfulness") else ins
        v = _ci([r.get(k) for r in pool])
        if v:
            s["overall"][k] = v
    if any("abstained" in r for r in rows):
        s["overall"]["false_abstention_rate"] = _ci([float(r["abstained"]) for r in ins if "abstained" in r])
        if oos:
            s["overall"]["oos_correct_abstention"] = _ci([float(r["abstained"]) for r in oos if "abstained" in r])
    by_cat = defaultdict(list)
    for r in ins:
        by_cat[r["category"]].append(r)
    s["by_category"] = {c: {k: round(sum(r.get(k, 0) or 0 for r in rs) / len(rs), 3)
                            for k in ["hit@5", "mrr", "context_hit"] + (["correctness"] if any("correctness" in r for r in rs) else [])}
                        | {"n": len(rs)} for c, rs in by_cat.items()}
    s["causes"] = summarize_causes([r["diagnosis"] for r in rows])
    return s


def render_report(s: Dict, rows: List[Dict]) -> str:
    L = [f"# Evaluation report: {s['name']}", "",
         f"{s['n_questions']} questions ({s['n_in_scope']} in scope). Means with bootstrap 95% CI.", "",
         "| metric | mean | 95% CI | n |", "|---|---|---|---|"]
    for k, v in s["overall"].items():
        if v:
            L.append(f"| {k} | {v['mean']} | {v['ci95'][0]} – {v['ci95'][1]} | {v['n']} |")
    L += ["", "## By question category", "", "| category | n | " + " | ".join(
        k for k in next(iter(s["by_category"].values()), {}) if k != "n") + " |"]
    keys = [k for k in next(iter(s["by_category"].values()), {}) if k != "n"]
    L.append("|---|---|" + "---|" * len(keys))
    for c, v in s["by_category"].items():
        L.append(f"| {c} | {v['n']} | " + " | ".join(str(v[k]) for k in keys) + " |")
    L += ["", "## Failure causes", "", "| cause | count | what to do |", "|---|---|---|"]
    for c, n in s["causes"].items():
        L.append(f"| {c} | {n} | {FIXES.get(c, '')} |")
    L += ["", "## Questions that need attention", ""]
    for r in rows:
        if r["cause"] in {"ANSWERED_OK", "OOS_CORRECT_ABSTAIN", "OOS_NOT_EVALUATED"} or (r["cause"] == "RETRIEVED_OK" and "answer" not in r):
            continue
        d = r["diagnosis"]
        L.append(f"**{r['id']}** ({r['category']}) {r['question']}  ")
        L.append(f"cause: `{r['cause']}` · per-retriever rank of gold: {d.get('component_ranks')} · "
                 f"candidate rank: {d.get('candidate_rank')} · final rank: {d.get('final_rank')}  ")
        L.append(f"top retrieved: {d.get('top_retrieved')}  ")
        if "answer" in r:
            L.append(f"answer: {r['answer'][:300]}  ")
            if r.get("unsupported_claims"):
                L.append(f"unsupported claims: {r['unsupported_claims']}  ")
            if r.get("judge_reason"):
                L.append(f"judge: {r.get('correctness')} – {r['judge_reason']}  ")
        L.append("")
    return "\n".join(L)
