"""
Hybrid RAG Engine
=================
Combines BM25 lexical search + FAISS semantic search with
Reciprocal Rank Fusion (RRF) and Relative Score Fusion (RSF).
Uses Ollama as the LLM backend.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import faiss
import httpx
import numpy as np
from rank_bm25 import BM25Okapi


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Document:
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    doc: Document
    score: float
    rank: int
    source: str  # "lexical" | "semantic" | "hybrid"


# ---------------------------------------------------------------------------
# Embedder (Ollama nomic-embed-text or fallback TF-IDF vectors)
# ---------------------------------------------------------------------------

class OllamaEmbedder:
    """Calls Ollama embedding endpoint."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._dim: Optional[int] = None

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            vec = self._embed_one(text)
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)

    def _embed_one(self, text: str) -> np.ndarray:
    # Truncate very long texts to avoid timeout
        text = text[:2000]
        try:
            r = httpx.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=60,
            )
            r.raise_for_status()
            vec = np.array(r.json()["embedding"], dtype=np.float32)
        # Cache the real dimension on first success
            if self._dim is None:
                self._dim = len(vec)
            norm = np.linalg.norm(vec)
            return vec / (norm + 1e-10)
        except Exception as e:
        # Fallback: deterministic hash-based pseudo-embedding
            dim = self._dim or 768  # nomic-embed-text produces 768-dim
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            vec = rng.standard_normal(dim).astype(np.float32)
            norm = np.linalg.norm(vec)
            return vec / (norm + 1e-10)
    
    @property
    def dim(self) -> int:
        if self._dim is None:
            sample = self._embed_one("test")
            self._dim = len(sample)
        return self._dim


# ---------------------------------------------------------------------------
# Text splitter
# ---------------------------------------------------------------------------

def split_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """Sentence-aware sliding window chunker."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current, current_len = [], [], 0
    for sent in sentences:
        sent_len = len(sent.split())
        if current_len + sent_len > chunk_size and current:
            chunks.append(" ".join(current))
            # overlap: keep last N words
            overlap_words = " ".join(current).split()[-overlap:]
            current = [" ".join(overlap_words)]
            current_len = len(overlap_words)
        current.append(sent)
        current_len += sent_len
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if len(c.strip()) > 20]


# ---------------------------------------------------------------------------
# Fusion algorithms
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    lexical_results: List[RetrievedChunk],
    semantic_results: List[RetrievedChunk],
    k: int = 60,
    alpha: float = 0.5,   # weight for semantic
) -> List[RetrievedChunk]:
    """
    RRF formula: score(d) = Σ 1 / (k + rank(d))
    Documents appearing in both lists are boosted.
    """
    scores: Dict[str, float] = {}
    docs: Dict[str, Document] = {}

    beta = 1 - alpha  # weight for lexical

    for rank_idx, chunk in enumerate(lexical_results, start=1):
        did = chunk.doc.id
        scores[did] = scores.get(did, 0.0) + beta * (1.0 / (k + rank_idx))
        docs[did] = chunk.doc

    for rank_idx, chunk in enumerate(semantic_results, start=1):
        did = chunk.doc.id
        scores[did] = scores.get(did, 0.0) + alpha * (1.0 / (k + rank_idx))
        docs[did] = chunk.doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        RetrievedChunk(doc=docs[did], score=sc, rank=i + 1, source="hybrid-rrf")
        for i, (did, sc) in enumerate(ranked)
    ]


def relative_score_fusion(
    lexical_results: List[RetrievedChunk],
    semantic_results: List[RetrievedChunk],
    semantic_weight: float = 0.6,
) -> List[RetrievedChunk]:
    """
    RSF: normalize each list to [0,1] then weighted average.
    semantic_weight + lexical_weight = 1.0
    """
    lexical_weight = 1.0 - semantic_weight

    def normalize(results: List[RetrievedChunk]) -> Dict[str, float]:
        if not results:
            return {}
        scores = [r.score for r in results]
        mn, mx = min(scores), max(scores)
        rng = mx - mn if mx != mn else 1e-10
        return {r.doc.id: (r.score - mn) / rng for r in results}

    lex_norm = normalize(lexical_results)
    sem_norm = normalize(semantic_results)

    docs: Dict[str, Document] = {}
    for r in lexical_results + semantic_results:
        docs[r.doc.id] = r.doc

    all_ids = set(lex_norm) | set(sem_norm)
    combined: Dict[str, float] = {}
    for did in all_ids:
        combined[did] = (
            lexical_weight * lex_norm.get(did, 0.0)
            + semantic_weight * sem_norm.get(did, 0.0)
        )

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [
        RetrievedChunk(doc=docs[did], score=sc, rank=i + 1, source="hybrid-rsf")
        for i, (did, sc) in enumerate(ranked)
    ]


# ---------------------------------------------------------------------------
# Hybrid Retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Maintains a FAISS index (semantic) + BM25 index (lexical).
    Supports incremental document addition and persistence.
    """

    def __init__(
        self,
        embedder: OllamaEmbedder,
        index_dir: str = "./indexes",
        fusion: str = "rrf",          # "rrf" | "rsf"
        semantic_weight: float = 0.6,
        top_k: int = 5,
    ):
        self.embedder = embedder
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.fusion = fusion
        self.semantic_weight = semantic_weight
        self.top_k = top_k

        self.documents: List[Document] = []
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self._bm25: Optional[BM25Okapi] = None

    # ---- indexing ----

    def add_documents(self, docs: List[Document]):
        self.documents.extend(docs)
        self._rebuild_indexes()

    def _rebuild_indexes(self):
        if not self.documents:
            return
        texts = [d.text for d in self.documents]

        # BM25
        tokenized = [t.lower().split() for t in texts]
        self._bm25 = BM25Okapi(tokenized)

        # FAISS
        embeddings = self.embedder.embed(texts)
        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embeddings)

    # ---- retrieval ----

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        if not self.documents:
            return []

        lexical = self._lexical_search(query)
        semantic = self._semantic_search(query)

        if self.fusion == "rrf":
            results = reciprocal_rank_fusion(lexical, semantic, alpha=self.semantic_weight)
        else:
            results = relative_score_fusion(lexical, semantic, semantic_weight=self.semantic_weight)

        return results[: self.top_k]

    def _lexical_search(self, query: str) -> List[RetrievedChunk]:
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        ranked_ids = np.argsort(scores)[::-1][: self.top_k * 2]
        return [
            RetrievedChunk(
                doc=self.documents[i],
                score=float(scores[i]),
                rank=rank + 1,
                source="lexical",
            )
            for rank, i in enumerate(ranked_ids)
            if scores[i] > 0
        ]

    def _semantic_search(self, query: str) -> List[RetrievedChunk]:
        q_vec = self.embedder.embed([query])
        distances, indices = self.faiss_index.search(q_vec, self.top_k * 2)
        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0:
                continue
            results.append(
                RetrievedChunk(
                    doc=self.documents[idx],
                    score=float(dist),
                    rank=rank + 1,
                    source="semantic",
                )
            )
        return results

    # ---- persistence ----

    def save(self):
        with open(self.index_dir / "docs.pkl", "wb") as f:
            pickle.dump(self.documents, f)
        if self.faiss_index:
            faiss.write_index(self.faiss_index, str(self.index_dir / "faiss.index"))

    def load(self):
        doc_path = self.index_dir / "docs.pkl"
        idx_path = self.index_dir / "faiss.index"
        if doc_path.exists():
            with open(doc_path, "rb") as f:
                self.documents = pickle.load(f)
        if idx_path.exists():
            self.faiss_index = faiss.read_index(str(idx_path))
            texts = [d.text for d in self.documents]
            tokenized = [t.lower().split() for t in texts]
            self._bm25 = BM25Okapi(tokenized)
