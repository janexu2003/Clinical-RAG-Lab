"""
Download the benchmark data (run this on your own machine; it needs internet access).

    python scripts/download_benchmarks.py beir nfcorpus scifact      # small BEIR medical sets
    python scripts/download_benchmarks.py beir trec-covid            # larger (~171k docs)
    python scripts/download_benchmarks.py mirage                     # MIRAGE benchmark.json
    python scripts/download_benchmarks.py medrag-textbooks           # MedRAG Textbooks corpus (needs: pip install huggingface_hub)

Everything goes under data/ (git-ignored):
    data/beir/<name>/{corpus.jsonl, queries.jsonl, qrels/test.tsv}
    data/mirage/benchmark.json
    data/medrag/textbooks/**/*.jsonl

Licences: check each dataset's terms before redistributing; these are for research use.
"""
import argparse
import io
import os
import shutil
import sys
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
BEIR_HF = "BeIR/{name}"          # fallback mirror on the Hugging Face Hub
MIRAGE_URL = "https://raw.githubusercontent.com/Teddy-XiongGZ/MIRAGE/main/benchmark.json"
MEDRAG_TEXTBOOKS_HF = "MedRAG/textbooks"


def fetch(url: str) -> bytes:
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "clinical-rag-lab"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read()


def beir(names):
    for name in names:
        dest = os.path.join(DATA, "beir", name)
        if os.path.exists(os.path.join(dest, "corpus.jsonl")):
            print(f"{name}: already present")
            continue
        try:
            z = zipfile.ZipFile(io.BytesIO(fetch(BEIR_URL.format(name=name))))
            tmp = os.path.join(DATA, "beir", "_tmp")
            z.extractall(tmp)
            inner = os.path.join(tmp, name) if os.path.isdir(os.path.join(tmp, name)) else tmp
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.move(inner, dest)
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception as e:   # noqa: BLE001
            print(f"  direct download failed ({e}); trying the Hugging Face mirror")
            beir_from_hf(name, dest)
        n_docs = sum(1 for _ in open(os.path.join(dest, "corpus.jsonl")))
        print(f"{name}: {n_docs} documents -> {dest}")


def beir_from_hf(name, dest):
    """Fallback: BeIR/<name> (corpus + queries) and BeIR/<name>-qrels on the Hub."""
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("pip install datasets   (needed for the Hugging Face fallback)")
    import csv
    import json
    os.makedirs(os.path.join(dest, "qrels"), exist_ok=True)
    for cfg in ("corpus", "queries"):
        ds = load_dataset(BEIR_HF.format(name=name), cfg)[cfg]
        with open(os.path.join(dest, f"{cfg}.jsonl"), "w") as f:
            for r in ds:
                f.write(json.dumps({"_id": r["_id"], "title": r.get("title", ""), "text": r["text"]}) + "\n")
    q = load_dataset(BEIR_HF.format(name=name) + "-qrels")
    split = "test" if "test" in q else list(q)[0]
    with open(os.path.join(dest, "qrels", "test.tsv"), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["query-id", "corpus-id", "score"])
        for r in q[split]:
            w.writerow([r["query-id"], r["corpus-id"], r["score"]])


def mirage():
    dest = os.path.join(DATA, "mirage", "benchmark.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(fetch(MIRAGE_URL))
    import json
    b = json.load(open(dest))
    print("MIRAGE:", {k: len(v) for k, v in b.items()}, "->", dest)


def medrag_textbooks():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("pip install huggingface_hub")
    dest = os.path.join(DATA, "medrag", "textbooks")
    snapshot_download(repo_id=MEDRAG_TEXTBOOKS_HF, repo_type="dataset", local_dir=dest,
                      allow_patterns=["*.jsonl", "chunk/*", "*.md"])
    files = [os.path.join(r, f) for r, _, fs in os.walk(dest) for f in fs if f.endswith(".jsonl")]
    n = sum(sum(1 for _ in open(p)) for p in files)
    print(f"MedRAG Textbooks: {len(files)} files, {n} snippets -> {dest}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["beir", "mirage", "medrag-textbooks"])
    ap.add_argument("names", nargs="*", default=["nfcorpus", "scifact"])
    a = ap.parse_args()
    if a.what == "beir":
        beir(a.names)
    elif a.what == "mirage":
        mirage()
    else:
        medrag_textbooks()
