"""
Retrievers for a SMALL corpus: everything lives in memory, no vector database needed.
(3-4k chunks x 384 dims = ~6 MB. A numpy dot product over that takes a millisecond.)

    bm25   - keyword matching; great for drug names, rare disease names, abbreviations
    tfidf  - keyword matching with cosine; mostly a sanity baseline
    lsa    - TF-IDF + truncated SVD: a cheap "semantic" retriever that needs no download
    dense  - sentence-transformers embeddings (MiniLM, bge, MedCPT, ...)
    hybrid - several of the above fused with Reciprocal Rank Fusion

Every retriever returns a full ranking on request, which is what the failure
diagnosis needs ("the right chunk was at rank 37 for dense but rank 2 for BM25").
"""
from __future__ import annotations

import hashlib
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ragkit.types import Chunk

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP = set(
    "the a an of and or to in is are was were be been by for on with as at from that this it its "
    "what how which who why when does do can should could would will may might into about".split())


def tokenize(text: str) -> List[str]:
    toks = [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP]
    # light suffix stripping so "diagnosed"~"diagnosis", "treated"~"treatment"
    return [re.sub(r"(osis|osed|ose|ment|ing|ed|es|s)$", "", t) if len(t) > 4 else t for t in toks]


Ranking = List[Tuple[int, float]]   # (chunk position, score), best first


class Retriever:
    name = "base"

    def fit(self, chunks: List[Chunk]) -> "Retriever":
        self.chunks = chunks
        return self

    def rank(self, query: str, k: Optional[int] = None) -> Ranking:
        raise NotImplementedError


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b

    def fit(self, chunks):
        from rank_bm25 import BM25Okapi
        super().fit(chunks)
        self.bm25 = BM25Okapi([tokenize(c.content) for c in chunks], k1=self.k1, b=self.b)
        return self

    def rank(self, query, k=None):
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)[: k or len(scores)]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]


class TfidfRetriever(Retriever):
    name = "tfidf"

    def fit(self, chunks):
        from sklearn.feature_extraction.text import TfidfVectorizer
        super().fit(chunks)
        self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                   ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self.X = self.vec.fit_transform([c.content for c in chunks])
        return self

    def rank(self, query, k=None):
        q = self.vec.transform([query])
        scores = (self.X @ q.T).toarray().ravel()
        order = np.argsort(-scores)[: k or len(scores)]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]


class LSARetriever(TfidfRetriever):
    """Latent semantic analysis: offline, download-free stand-in for an embedding model."""
    name = "lsa"

    def __init__(self, dims: int = 256, seed: int = 0):
        self.dims, self.seed = dims, seed

    def fit(self, chunks):
        from sklearn.decomposition import TruncatedSVD
        super().fit(chunks)
        self.svd = TruncatedSVD(n_components=min(self.dims, self.X.shape[1] - 1), random_state=self.seed)
        Z = self.svd.fit_transform(self.X)
        self.Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
        return self

    def rank(self, query, k=None):
        z = self.svd.transform(self.vec.transform([query]))[0]
        z = z / (np.linalg.norm(z) + 1e-9)
        scores = self.Z @ z
        order = np.argsort(-scores)[: k or len(scores)]
        return [(int(i), float(scores[i])) for i in order]


class DenseRetriever(Retriever):
    """
    sentence-transformers model, embeddings cached on disk.
    Models worth comparing on a medical corpus:
      sentence-transformers/all-MiniLM-L6-v2   (fast general baseline, 384-d)
      BAAI/bge-small-en-v1.5                   (strong small general model; query_prefix below)
      ncbi/MedCPT-Query-Encoder + ncbi/MedCPT-Article-Encoder (biomedical, asymmetric)
    """
    name = "dense"

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 query_prefix: str = "", doc_model: Optional[str] = None, cache_dir: str = ".cache"):
        self.model_name, self.doc_model_name = model, doc_model or model
        self.query_prefix, self.cache_dir = query_prefix, cache_dir

    def fit(self, chunks):
        from sentence_transformers import SentenceTransformer
        super().fit(chunks)
        self.q_model = SentenceTransformer(self.model_name)
        self.d_model = self.q_model if self.doc_model_name == self.model_name else SentenceTransformer(self.doc_model_name)
        key = hashlib.sha1(("".join(c.id for c in chunks) + self.doc_model_name).encode()).hexdigest()[:12]
        path = os.path.join(self.cache_dir, f"emb_{key}.npy")
        if os.path.exists(path):
            self.E = np.load(path)
        else:
            self.E = self.d_model.encode([c.content for c in chunks], batch_size=64,
                                         normalize_embeddings=True, show_progress_bar=True)
            os.makedirs(self.cache_dir, exist_ok=True)
            np.save(path, self.E)
        return self

    def rank(self, query, k=None):
        q = self.q_model.encode([self.query_prefix + query], normalize_embeddings=True)[0]
        scores = self.E @ q
        order = np.argsort(-scores)[: k or len(scores)]
        return [(int(i), float(scores[i])) for i in order]


def rrf(rankings: Sequence[Ranking], k: int = 60, weights: Optional[Sequence[float]] = None) -> Ranking:
    """Reciprocal Rank Fusion: score = sum_i w_i / (k + rank_i). Robust, no score calibration needed."""
    weights = weights or [1.0] * len(rankings)
    fused: Dict[int, float] = defaultdict(float)
    for w, ranking in zip(weights, rankings):
        for r, (idx, _) in enumerate(ranking):
            fused[idx] += w / (k + r + 1)
    return sorted(fused.items(), key=lambda x: -x[1])


class HybridRetriever(Retriever):
    name = "hybrid"

    def __init__(self, retrievers: List[Retriever], weights: Optional[List[float]] = None, depth: int = 100):
        self.retrievers, self.weights, self.depth = retrievers, weights, depth

    def fit(self, chunks):
        super().fit(chunks)
        for r in self.retrievers:
            r.fit(chunks)
        return self

    def rank(self, query, k=None):
        return rrf([r.rank(query, self.depth) for r in self.retrievers], weights=self.weights)[: k or None]

    def component_ranks(self, query: str, depth: int = 200) -> Dict[str, Dict[int, int]]:
        """For diagnosis: chunk position -> 1-based rank, per component retriever."""
        return {r.name: {idx: i + 1 for i, (idx, _) in enumerate(r.rank(query, depth))}
                for r in self.retrievers}


def make_retriever(cfg: Dict) -> Retriever:
    """cfg example: {"retrievers": ["bm25", "dense"], "dense_model": "...", "weights": [1, 1]}"""
    def one(name: str) -> Retriever:
        if name == "bm25":
            return BM25Retriever()
        if name == "tfidf":
            return TfidfRetriever()
        if name == "lsa":
            return LSARetriever(cfg.get("lsa_dims", 256))
        if name == "dense":
            return DenseRetriever(cfg.get("dense_model", "sentence-transformers/all-MiniLM-L6-v2"),
                                  cfg.get("query_prefix", ""), cfg.get("doc_model"))
        raise ValueError(f"unknown retriever {name}")

    names = cfg.get("retrievers", ["bm25"])
    parts = [one(n) for n in names]
    depth = cfg.get("depth", max(100, cfg.get("fetch_k", 30)))
    return HybridRetriever(parts, cfg.get("weights"), depth=depth)
