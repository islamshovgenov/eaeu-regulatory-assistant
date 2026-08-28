"""Hybrid retrieval: dense vectors + BM25, fused with Reciprocal Rank Fusion.

Why both:

* **Dense search** handles paraphrase ("когда можно не проводить исследование
  сравнительной биодоступности" → «биовейвер»).
* **BM25** handles the literal regulatory tokens that embeddings blur:
  «пункт 42», «AUC(0-t)», «90% доверительный интервал», «высоковариабельный»,
  «референтный препарат».

Fusion uses weighted RRF (``1 / (k + rank)``), which needs no score calibration
between the two very differently-scaled rankers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.config import Settings, get_settings
from app.db.models import Chunk
from app.db.repository import Repository
from app.rag.indexes import BM25Index, VectorIndex

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RetrievedChunk:
    """A chunk with its provenance through the retrieval pipeline."""

    chunk: Chunk
    score: float = 0.0
    vector_rank: int | None = None
    vector_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None
    matched_queries: list[str] = field(default_factory=list)

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    def retrieval_methods(self) -> list[str]:
        methods: list[str] = []
        if self.vector_rank is not None:
            methods.append("vector")
        if self.bm25_rank is not None:
            methods.append("bm25")
        if self.rerank_score is not None:
            methods.append("rerank")
        return methods


class HybridSearcher:
    """Runs both rankers and fuses their results."""

    def __init__(
        self,
        settings: Settings | None = None,
        repository: Repository | None = None,
        vector_index: VectorIndex | None = None,
        bm25_index: BM25Index | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.repository = repository or Repository()
        self._vector_index = vector_index
        self._bm25_index = bm25_index
        self._bm25_loaded = bm25_index is not None
        self._chunk_cache: dict[str, Chunk] = {}

    # -- lazy resources -----------------------------------------------------
    @property
    def vector_index(self) -> VectorIndex | None:
        if self._vector_index is None:
            try:
                self._vector_index = VectorIndex(self.settings)
            except Exception as exc:  # noqa: BLE001
                logger.error("Vector index unavailable: %s", exc)
                return None
        return self._vector_index

    @property
    def bm25_index(self) -> BM25Index | None:
        if not self._bm25_loaded:
            self._bm25_index = BM25Index.load()
            self._bm25_loaded = True
            if self._bm25_index is None:
                logger.warning("BM25 index not found — keyword search disabled")
        return self._bm25_index

    def _chunk_by_id(self, chunk_id: str) -> Chunk | None:
        if chunk_id not in self._chunk_cache:
            chunk = self.repository.get_chunk(chunk_id)
            if chunk is None:
                return None
            self._chunk_cache[chunk_id] = chunk
        return self._chunk_cache[chunk_id]

    # -- search -------------------------------------------------------------
    def search(
        self,
        query: str,
        vector_top_k: int | None = None,
        bm25_top_k: int | None = None,
        final_top_k: int | None = None,
        query_filter: object | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve and fuse results for a single query string."""
        vector_top_k = vector_top_k or self.settings.vector_top_k
        bm25_top_k = bm25_top_k or self.settings.bm25_top_k
        final_top_k = final_top_k or self.settings.final_top_k

        results: dict[str, RetrievedChunk] = {}

        # -- dense ----------------------------------------------------------
        index = self.vector_index
        if index is not None:
            try:
                for rank, (chunk, score) in enumerate(
                    index.search(query, vector_top_k, query_filter), start=1
                ):
                    entry = results.setdefault(chunk.chunk_id, RetrievedChunk(chunk))
                    entry.vector_rank = rank
                    entry.vector_score = score
                    self._chunk_cache[chunk.chunk_id] = chunk
            except Exception as exc:  # noqa: BLE001
                logger.error("Vector search failed: %s", exc)

        # -- lexical --------------------------------------------------------
        bm25 = self.bm25_index
        if bm25 is not None:
            for rank, (chunk_id, score) in enumerate(
                bm25.search(query, bm25_top_k), start=1
            ):
                chunk = self._chunk_by_id(chunk_id)
                if chunk is None:
                    continue
                entry = results.setdefault(chunk_id, RetrievedChunk(chunk))
                entry.bm25_rank = rank
                entry.bm25_score = score

        fused = self._fuse(list(results.values()))
        for item in fused:
            item.matched_queries = [query]
        return fused[:final_top_k]

    def multi_search(
        self,
        queries: list[str],
        final_top_k: int | None = None,
        query_filter: object | None = None,
    ) -> list[RetrievedChunk]:
        """Run several query formulations and fuse their fused results.

        Used for INN-specific search: the same question is asked with the
        Russian INN, the Latin INN and an Expert-Committee-oriented phrasing.
        """
        final_top_k = final_top_k or self.settings.final_top_k
        merged: dict[str, RetrievedChunk] = {}

        for query in queries:
            for rank, item in enumerate(self.search(query, query_filter=query_filter), start=1):
                existing = merged.get(item.chunk_id)
                contribution = 1.0 / (self.settings.rrf_k + rank)
                if existing is None:
                    item.score = contribution
                    merged[item.chunk_id] = item
                else:
                    existing.score += contribution
                    existing.matched_queries.extend(item.matched_queries)
                    if item.vector_rank is not None and (
                        existing.vector_rank is None or item.vector_rank < existing.vector_rank
                    ):
                        existing.vector_rank = item.vector_rank
                        existing.vector_score = item.vector_score
                    if item.bm25_rank is not None and (
                        existing.bm25_rank is None or item.bm25_rank < existing.bm25_rank
                    ):
                        existing.bm25_rank = item.bm25_rank
                        existing.bm25_score = item.bm25_score

        ordered = sorted(merged.values(), key=lambda r: r.score, reverse=True)
        for item in ordered:
            item.matched_queries = sorted(set(item.matched_queries))
        return ordered[:final_top_k]

    # -- fusion -------------------------------------------------------------
    def _fuse(self, items: list[RetrievedChunk]) -> list[RetrievedChunk]:
        k = self.settings.rrf_k
        for item in items:
            score = 0.0
            if item.vector_rank is not None:
                score += self.settings.rrf_vector_weight / (k + item.vector_rank)
            if item.bm25_rank is not None:
                score += self.settings.rrf_bm25_weight / (k + item.bm25_rank)
            item.score = score
        return sorted(items, key=lambda r: r.score, reverse=True)

    # -- health -------------------------------------------------------------
    def is_ready(self) -> bool:
        index = self.vector_index
        return bool(index and index.count() > 0) or self.bm25_index is not None

    def stats(self) -> dict[str, int]:
        index = self.vector_index
        bm25 = self.bm25_index
        return {
            "vectors": index.count() if index else 0,
            "bm25_chunks": len(bm25.chunk_ids) if bm25 else 0,
            "documents": self.repository.count_documents(),
            "chunks": self.repository.count_chunks(),
        }
