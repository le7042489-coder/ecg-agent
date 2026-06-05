"""Hybrid (sparse + optional dense) retriever over a versioned guideline corpus.

Key properties:
  * **Version-aware**: ``only_current=True`` (default) never serves superseded
    recommendations. With ``only_current=False`` a superseded hit is annotated
    with a forward-resolution note pointing at the current recommendation.
  * **Metadata filtering**: society / topic / lang / status.
  * **Graceful degradation**: dense retrieval is used only when precomputed
    corpus vectors (built offline) AND sentence-transformers are available;
    otherwise it falls back to pure-Python BM25 with zero third-party deps.

Storage backend for v1 is intentionally simple (in-memory records + a numpy
vector matrix). The corpus is small, so brute-force scoring is microseconds; a
LanceDB/Qdrant backend can be swapped in behind ``search()`` once it grows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .schema import GuidelineRecord, load_corpus, load_corpus_version
from .tokenize import tokenize


@dataclass
class SearchResult:
    record: GuidelineRecord
    score: float
    retrieval: str                 # "hybrid" | "bm25" | "dense"
    change_note: Optional[str] = None

    def to_dict(self) -> dict:
        r = self.record
        return {
            "id": r.id,
            "society": r.society,
            "doc_title": r.doc_title,
            "version": r.version,
            "status": r.status,
            "topic": r.topic,
            "section": r.section,
            "class": r.recommendation_class,
            "LOE": r.level_of_evidence,
            "citation": r.citation,
            "text": r.text,
            "score": round(self.score, 6),
            "retrieval": self.retrieval,
            "change_note": self.change_note,
        }


class _SimpleBM25:
    """Dependency-free BM25 (Okapi). Fine for a few thousand documents."""

    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(corpus_tokens)
        self.doc_len = [len(d) for d in corpus_tokens]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        self.tf: List[dict] = []
        df: dict = {}
        for d in corpus_tokens:
            counts: dict = {}
            for t in d:
                counts[t] = counts.get(t, 0) + 1
            self.tf.append(counts)
            for t in counts:
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def scores(self, query_tokens: List[str]) -> List[float]:
        out = [0.0] * self.N
        if not self.avgdl:
            return out
        for i, counts in enumerate(self.tf):
            dl = self.doc_len[i] or 1
            s = 0.0
            for t in query_tokens:
                f = counts.get(t)
                if not f:
                    continue
                idf = self.idf.get(t, 0.0)
                s += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out[i] = s
        return out


def _rrf(rankings: List[List[int]], k: int = 60) -> dict:
    """Reciprocal Rank Fusion over several best-first ranked index lists."""
    fused: dict = {}
    for ranks in rankings:
        for rank, idx in enumerate(ranks):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return fused


class GuidelineRetriever:
    def __init__(
        self,
        records: List[GuidelineRecord],
        vectors=None,
        embed_model_name: Optional[str] = None,
        corpus_version: str = "unknown",
        enable_dense: bool = True,
    ):
        self.records = records
        self.corpus_version = corpus_version
        self._by_id = {r.id: r for r in records}
        self._tokens = [tokenize(f"{r.text} {r.section}", r.lang) for r in records]
        self._bm25 = _SimpleBM25(self._tokens)
        self._vectors = vectors                      # numpy [N, d] (L2-normalized) or None
        self._embed_model_name = embed_model_name
        self._enable_dense = bool(enable_dense and vectors is not None)
        self._embedder = None

    # ---- constructors -------------------------------------------------------

    @classmethod
    def from_corpus(cls, jsonl_path: str | Path, enable_dense: bool = False) -> "GuidelineRetriever":
        """BM25-only retriever straight from a corpus JSONL (no build step)."""
        records = load_corpus(jsonl_path)
        return cls(
            records,
            vectors=None,
            corpus_version=load_corpus_version(jsonl_path),
            enable_dense=enable_dense,
        )

    @classmethod
    def from_index(cls, index_dir: str | Path, enable_dense: bool = True) -> "GuidelineRetriever":
        """Load a prebuilt index dir (records.jsonl + embeddings.npy + index_manifest.json)."""
        import json

        index_dir = Path(index_dir)
        records = load_corpus(index_dir / "records.jsonl")
        meta_path = index_dir / "index_manifest.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        vectors, model_name = None, meta.get("embed_model")
        emb_path = index_dir / "embeddings.npy"
        if enable_dense and emb_path.is_file():
            try:
                import numpy as np

                vectors = np.load(emb_path)
            except Exception:
                vectors = None
        return cls(
            records,
            vectors=vectors,
            embed_model_name=model_name,
            corpus_version=meta.get("corpus_version", "unknown"),
            enable_dense=enable_dense,
        )

    # ---- internals ----------------------------------------------------------

    def _passes(self, r: GuidelineRecord, society, topic, only_current, lang) -> bool:
        if only_current and r.status != "current":
            return False
        if society and r.society.lower() != str(society).lower():
            return False
        if topic:
            # topic may be a single string or a collection of allowed topics
            if isinstance(topic, str):
                if r.topic != topic:
                    return False
            elif r.topic not in topic:
                return False
        if lang and r.lang != lang:
            return False
        return True

    def _encode_query(self, query: str):
        import numpy as np

        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self._embed_model_name or "BAAI/bge-small-zh-v1.5")
        v = self._embedder.encode([query], normalize_embeddings=True)
        return np.asarray(v, dtype="float32")[0]

    # ---- public API ---------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 5,
        society: Optional[str] = None,
        topic: Optional[str] = None,
        only_current: bool = True,
        lang: Optional[str] = None,
        mode: str = "auto",          # "auto" | "hybrid" | "bm25" | "dense"
        min_score: float = 0.0,      # BM25-only floor: require raw score > this (>0 = ≥1 token matched)
    ) -> List[SearchResult]:
        cand = [i for i, r in enumerate(self.records)
                if self._passes(r, society, topic, only_current, lang)]
        if not cand:
            return []

        sparse = self._bm25.scores(tokenize(query, lang or "zh"))
        sparse_rank = sorted(cand, key=lambda i: sparse[i], reverse=True)

        want_dense = self._enable_dense and mode in ("auto", "hybrid", "dense")
        dense_rank: Optional[List[int]] = None
        if want_dense:
            try:
                import numpy as np

                qv = self._encode_query(query)
                sims = self._vectors[cand] @ qv
                dense_rank = [cand[j] for j in np.argsort(-sims)]
            except Exception:
                dense_rank = None  # degrade silently to BM25 (no model / no numpy)

        # Always keep BM25 as the floor: only drop it when the caller explicitly
        # asked for dense-only AND dense actually produced a ranking. This means
        # mode="dense" on a BM25-only (on-device) index degrades to BM25 instead
        # of returning arbitrary zero-scored hits.
        rankings: List[List[int]] = []
        if not (mode == "dense" and dense_rank is not None):
            rankings.append(sparse_rank)
        if dense_rank is not None:
            rankings.append(dense_rank)

        if len(rankings) == 1:
            order = rankings[0]
            fused = {idx: float(len(order) - rank) for rank, idx in enumerate(order)}
            retrieval = "dense" if order is dense_rank else "bm25"
        else:
            fused = _rrf(rankings)
            retrieval = "hybrid"

        # Min-score floor. When BM25 is the only evidence (no dense ranking), a
        # candidate must actually match ≥1 query token (raw score > min_score);
        # otherwise an all-zero query (e.g. EN classifier labels vs a CN corpus)
        # would fall back to arbitrary file-order records. Dense, when it ran,
        # supplies a real similarity signal, so we don't floor it here.
        eligible = cand if dense_rank is not None else [i for i in cand if sparse[i] > min_score]
        top = sorted(eligible, key=lambda i: fused.get(i, 0.0), reverse=True)[:k]
        results: List[SearchResult] = []
        for i in top:
            r = self.records[i]
            note = None
            if r.status == "superseded" and r.superseded_by:
                cur = self._by_id.get(r.superseded_by[0])
                if cur:
                    note = (f"该条已被取代，现行版本：{cur.citation} "
                            f"（推荐类别 {cur.recommendation_class} / 证据级别 {cur.level_of_evidence}）")
            results.append(SearchResult(record=r, score=fused.get(i, 0.0),
                                        retrieval=retrieval, change_note=note))
        return results
