"""Benchmark loaders and metrics, tested on tiny fixtures in tests/fixtures (offline)."""
import io
import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragkit.benchmarks.beir import doc_ranking, ir_metrics, load_beir
from ragkit.benchmarks.mirage import PROMPTS, format_question, load_medrag_corpus, load_mirage, parse_choice
from ragkit.eval.diagnose import diagnose_retrieval
from ragkit.generate import format_sources
from ragkit.pipeline import RAGPipeline
from ragkit.types import Chunk

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def test_load_beir_filters_unlabelled_and_zero_grades():
    chunks, items = load_beir(os.path.join(FIX, "beir_tiny"))
    assert len(chunks) == 8 and chunks[0].meta["doc_id"] == "D1" and chunks[0].header
    assert [i.id for i in items] == ["Q1", "Q2", "Q3"]              # Q4 has no qrels
    assert items[2].gold_docs == {"D1": 2, "D7": 1}                 # grade-0 judgement dropped


def test_ir_metrics_known_values():
    rel = {"a": 2, "b": 1}
    assert ir_metrics(["a", "b", "c"], rel)["ndcg@10"] == 1.0
    m = ir_metrics(["c", "a"], rel)
    assert m["mrr@10"] == 0.5 and m["recall@100"] == 0.5 and m["hit@1"] == 0.0
    assert ir_metrics([], rel)["ndcg@10"] == 0.0


def test_doc_ranking_dedupes_split_documents():
    cs = [Chunk(id=f"{d}#{j}", text="x", meta={"doc_id": d}) for d, j in [("A", 0), ("B", 0), ("A", 1), ("C", 0)]]
    assert doc_ranking(cs) == ["A", "B", "C"]


def test_bm25_pipeline_and_doc_level_diagnosis():
    chunks, items = load_beir(os.path.join(FIX, "beir_tiny"))
    pipe = RAGPipeline({"retrieval": {"retrievers": ["bm25"], "fetch_k": 100, "k": 10}}, chunks=chunks)
    tr = pipe.retrieve(items[0].question)                            # salt / blood pressure
    ranked = doc_ranking([c.chunk for c in tr.candidates])
    assert ranked[0] == "D5" and ir_metrics(ranked, items[0].gold_docs)["ndcg@10"] > 0.9
    d = diagnose_retrieval(pipe, items[0], tr)
    assert d["cause"] == "RETRIEVED_OK" and d["component_ranks"]["bm25"] == 1


def test_mirage_loading_sampling_and_prompts():
    path = os.path.join(FIX, "mirage_tiny.json")
    all_items = load_mirage(path)
    assert len(all_items) == 5 and all_items[0]["answer"] in all_items[0]["options"]
    a = load_mirage(path, per_dataset=2, seed=1)
    b = load_mirage(path, per_dataset=2, seed=1)
    assert [i["id"] for i in a] == [i["id"] for i in b] and len(a) == 4     # same sample every run
    assert load_mirage(path, datasets=["pubmedqa"])[0]["dataset"] == "pubmedqa"
    q = format_question(all_items[0])
    assert "Options:" in q and "A. Isotretinoin" in q
    ctx = load_medrag_corpus(os.path.join(FIX, "textbooks_tiny"))
    assert len(ctx) == 5 and ctx[0].header == "Dermatology"
    for name in ("rag", "rag_cite"):
        s = PROMPTS[name].format(context=format_sources(ctx[:2]))      # must not raise on JSON braces
        assert "[1]" in s and "answer_choice" in s


def test_parse_choice_variants():
    opts = {"A": "x", "B": "y", "C": "z"}
    assert parse_choice('{"step_by_step_thinking": "...", "answer_choice": "B"}', opts) == "B"
    assert parse_choice('{"answer_choice": "(C)."}', opts) == "C"
    assert parse_choice("Reasoning... The answer is A.", opts) == "A"
    assert parse_choice('{"answer_choice": "E"}', opts) is None


def test_download_beir_unzips_into_place(tmp_path, monkeypatch):
    sys.path.insert(0, os.path.join(os.path.dirname(FIX), "..", "scripts"))
    import download_benchmarks as dl
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("nfcorpus/corpus.jsonl", json.dumps({"_id": "1", "title": "t", "text": "x"}) + "\n")
        z.writestr("nfcorpus/queries.jsonl", json.dumps({"_id": "q", "text": "x"}) + "\n")
        z.writestr("nfcorpus/qrels/test.tsv", "query-id\tcorpus-id\tscore\nq\t1\t1\n")
    monkeypatch.setattr(dl, "DATA", str(tmp_path))
    monkeypatch.setattr(dl, "fetch", lambda url: buf.getvalue())
    dl.beir(["nfcorpus"])
    chunks, items = load_beir(str(tmp_path / "beir" / "nfcorpus"))
    assert len(chunks) == 1 and items[0].gold_docs == {"1": 1}
