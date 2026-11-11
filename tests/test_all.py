"""Offline tests: pytest -q"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragkit import hallucination as H
from ragkit.context import NeighbourIndex, assemble
from ragkit.corpus import char_windows, pack_sentences, section_chunks, split_page
from ragkit.eval.dataset import EvalItem
from ragkit.eval.metrics import retrieval_metrics
from ragkit.eval.runner import run_eval
from ragkit.eval.stats import bootstrap_ci, cohen_kappa, paired_bootstrap
from ragkit.generate import ABSTAIN, is_abstention
from ragkit.llm import StubLLM, parse_json
from ragkit.pipeline import RAGPipeline
from ragkit.retrievers import BM25Retriever, HybridRetriever, rrf
from ragkit.types import Chunk

PAGE_A = """Acne
Definition
Acne is a common skin disease characterized by pimples on the face.
Diagnosis
Acne has a characteristic appearance and is not diffi-
cult to diagnose.
A
GALE ENCYCLOPEDIA OF MEDICINE 2 24
Acne
A photo of acne. (Reproduced by permission.)
GEM - 0001 to 0432 - A  10/22/03 1:41 PM  Page 24"""
PAGE_B = """The doctor takes a complete medical history.
Treatment
Oral antibiotics are taken daily for two to four months.
GALE ENCYCLOPEDIA OF MEDICINE 2 25
Acne
GEM - 0001 to 0432 - A  10/22/03 1:41 PM  Page 25"""


def C(i, text, entry="Acne", section="Treatment", page=0, idx=0):
    return Chunk(id=str(i), text=text, header=f"Book | {entry} | {section}", source="b.pdf",
                 entry=entry, section=section, page=page, page_end=page, chunk_index=idx)


TOY = [
    C(0, "Acne is a common skin disease characterized by pimples.", section="Definition", page=1),
    C(1, "Acne has a characteristic appearance and is not difficult to diagnose.", section="Diagnosis", page=2),
    C(2, "Oral antibiotics are taken daily for two to four months.", section="Treatment", page=3, idx=0),
    C(3, "Isotretinoin is the treatment of choice for severe acne.", section="Treatment", page=3, idx=1),
    C(4, "Anemia is a lack of red blood cells or hemoglobin.", entry="Anemia", section="Definition", page=9),
    C(5, "Bunions are caused by poorly fitting footwear.", entry="Bunion", section="Definition", page=20),
]


# ------------------------------------------------------------------ corpus
def test_footer_and_caption_removed():
    body, caps = split_page(PAGE_A)
    assert not any("GALE" in l or "GEM -" in l for l in body)
    assert caps == ["A photo of acne. (Reproduced by permission.)"]


def test_section_chunks_headers_pages_hyphenation():
    chunks = section_chunks([PAGE_A, PAGE_B], "b.pdf", min_chars=10)
    diag = next(c for c in chunks if c.section == "Diagnosis")
    assert diag.header.endswith("Acne | Diagnosis")
    assert "not difficult to diagnose" in diag.text and "medical history" in diag.text
    assert (diag.page, diag.page_end) == (0, 1)
    assert any(c.section == "Figure caption" for c in chunks)


def test_splitters():
    text = "One sentence here. " * 40
    parts = pack_sentences(text, 200, 40)
    assert all(len(p) <= 200 for p in parts) and len(parts) > 3
    assert parts[1].startswith("One sentence")                    # sentence boundaries kept
    assert all(len(w) <= 100 for w in char_windows("word " * 100, 100, 10))


# ------------------------------------------------------------------ retrieval
def test_bm25_and_rrf():
    r = BM25Retriever().fit(TOY)
    assert r.rank("how long oral antibiotics acne", 1)[0][0] == 2
    fused = rrf([[(1, 1.0), (2, 0.5)], [(2, 1.0), (1, 0.5)], [(2, 1.0)]])
    assert fused[0][0] == 2


def test_component_ranks():
    h = HybridRetriever([BM25Retriever()]).fit(TOY)
    ranks = h.component_ranks("bunion footwear")
    assert ranks["bm25"][5] == 1


def test_neighbours_keep_reading_order():
    nidx = NeighbourIndex(TOY)
    out = assemble([TOY[3]], k=1, nidx=nidx, window=1)
    assert [c.id for c in out] == ["2", "3"]


# ------------------------------------------------------------------ hallucination
def test_numeric_check_is_unit_aware():
    src = [TOY[2], C(9, "About 20% of patients need a second course.")]
    assert H.numeric_check("Antibiotics are taken for two to four months [1].", src) == []
    flags = H.numeric_check("It is taken for 20 weeks at 0.5 mg [1].", src)
    assert {f["value"] for f in flags} == {"20 week", "0.5 mg"}


def test_citation_and_quote_checks():
    flags = H.citation_check("Acne is common [3]. This sentence has no citation at all.", 2)
    assert {f["type"] for f in flags} == {"invalid_citation", "uncited_sentence"}
    q = H.quote_check([{"source": 1, "quote": "taken daily for two to four months"},
                       {"source": 1, "quote": "cures acne forever"}], [TOY[2]])
    assert [f["type"] for f in q] == ["fabricated_quote"]
    assert H.offline_checks(ABSTAIN, TOY) == []


def test_llm_claim_verification_and_repair():
    def fn(system, user):
        if "fact checker" in system:
            bad = "0.5 mg" in user.split("Answer:")[-1]
            return json.dumps({"claims": [
                {"claim": "Antibiotics are taken for two to four months", "verdict": "supported", "source": 1}] + (
                [{"claim": "Isotretinoin dose is 0.5 mg per kg", "verdict": "not_found", "source": None}] if bad else [])})
        return "Oral antibiotics are taken daily for two to four months [1]."   # regenerated answer
    llm = StubLLM(fn)
    bad = "Oral antibiotics are taken for two to four months [1]. Isotretinoin dose is 0.5 mg per kg [1]."
    rep = H.check(bad, [TOY[2]], llm, ["offline", "llm_claims"])
    assert rep.faithfulness == 0.5 and len(rep.unsupported_claims) == 1
    fixed, rep2 = H.repair("q", bad, [TOY[2]], llm, rep, ["offline", "llm_claims"])
    assert "0.5" not in fixed and rep2.faithfulness == 1.0 and rep2.repaired


# ------------------------------------------------------------------ pipeline + evaluation
def smart_stub(system, user):
    """Answers from the sources it is given; abstains when nothing relevant is there."""
    if "fact checker" in system:
        return json.dumps({"claims": [{"claim": "x", "verdict": "supported", "source": 1}]})
    if "careful, strict grader" in system:
        return json.dumps({"score": 0 if ABSTAIN in user else 1, "reason": "stub"})
    if "judge retrieval quality" in system:
        return json.dumps({"verdict": "sufficient"})
    if "two to four months" in system and "antibiotic" in user.lower():
        return "Oral antibiotics are taken daily for two to four months [1]."
    if "poorly fitting footwear" in system and "toe" in user.lower():
        return "Bunions are caused by poorly fitting footwear [1]."
    return ABSTAIN


def test_pipeline_answer_and_eval(tmp_path):
    cfg = {"name": "toy", "retrieval": {"retrievers": ["bm25"], "fetch_k": 5, "k": 2},
           "rerank": {"section_boost": True}, "corrective": {"enabled": True},
           "hallucination": {"checks": ["offline", "llm_claims"], "repair": True}}
    pipe = RAGPipeline(cfg, chunks=TOY, llm=StubLLM(smart_stub))
    at = pipe.answer("How long are oral antibiotics taken for acne?")
    assert "two to four months" in at.answer and not at.abstained and at.hallucination.faithfulness == 1.0

    items = [
        EvalItem("q1", "How long are oral antibiotics taken for acne?", "numeric", [3], "two to four months"),
        EvalItem("q2", "What causes a bump on the big toe?", "lay_wording", [20], "poorly fitting footwear"),
        EvalItem("q3", "What is the normal hemoglobin level?", "textbook", [99], "n/a"),
        EvalItem("q4", "What are the symptoms of COVID-19?", "out_of_scope", [], "NOT_IN_DOCUMENTS", in_scope=False),
    ]
    summary = run_eval(pipe, items, str(tmp_path), verbose=False)
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(tmp_path / "per_question.jsonl")}
    assert rows["q1"]["cause"] == "ANSWERED_OK" and rows["q1"]["correctness"] == 1.0
    assert rows["q3"]["cause"] == "NOT_IN_CORPUS"
    assert rows["q4"]["cause"] == "OOS_CORRECT_ABSTAIN"
    assert summary["overall"]["oos_correct_abstention"]["mean"] == 1.0
    assert (tmp_path / "report.md").exists()


def test_diagnosis_ranking_miss():
    cfg = {"retrieval": {"retrievers": ["bm25"], "fetch_k": 6, "k": 1}}
    pipe = RAGPipeline(cfg, chunks=TOY, llm=None)
    item = EvalItem("q", "acne treatment", "x", [3], "")
    from ragkit.eval.diagnose import diagnose_retrieval
    tr = pipe.retrieve(item.question)
    d = diagnose_retrieval(pipe, item, tr)
    assert d["cause"] in {"RETRIEVED_OK", "RANKING_MISS"}
    if d["cause"] == "RANKING_MISS":
        assert d["candidate_rank"] > 1


# ------------------------------------------------------------------ metrics + stats
def test_retrieval_metrics_bounds():
    m = retrieval_metrics([TOY[2], TOY[3], TOY[0]], [3])     # two chunks from the same gold page
    assert m["hit@1"] == 1 and m["mrr"] == 1 and 0 <= m["ndcg@3"] <= 1


def test_stats():
    m, lo, hi = bootstrap_ci([0, 1] * 10)
    assert lo < m < hi
    r = paired_bootstrap([0] * 10, [1] * 10)
    assert r["ci_low"] > 0 and r["wins"] == 10
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0


def test_misc():
    assert parse_json('Sure! ```json\n{"a": 1}\n```') == {"a": 1}
    assert is_abstention("I don't know the answer.") and is_abstention(ABSTAIN)
    assert not is_abstention("Acne is diagnosed clinically [1].")
