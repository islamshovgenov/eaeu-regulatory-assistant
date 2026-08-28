"""Optional cross-encoder reranking of fused retrieval results.

Reranking is **off by default** (``RERANK_ENABLED=false``): the multilingual
cross-encoders good enough for Russian legal text (e.g.
``BAAI/bge-reranker-v2-m3``) weigh ~2.2 GB and add seconds per query on CPU,
which is a poor trade for a laptop demo.  When it is enabled the model is
loaded lazily and any failure degrades gracefully to the RRF order.

A cheap, always-available :class:`LexicalOverlapReranker` provides a small
precision gain without extra downloads and is used as the fallback.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod

from app.config import Settings, get_settings
from app.rag.hybrid_search import RetrievedChunk
from app.rag.indexes import tokenize

logger = logging.getLogger(__name__)


class Reranker(ABC):
    name = "abstract"

    @abstractmethod
    def rerank(
        self, query: str, items: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        ...


class NoOpReranker(Reranker):
    """Keeps the fusion order (used when reranking is disabled)."""

    name = "none"

    def rerank(
        self, query: str, items: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        return items[:top_k]


class LexicalOverlapReranker(Reranker):
    """Zero-dependency reranker: IDF-free token overlap + structural priors.

    Rewards chunks that literally contain the query's rare tokens and chunks
    that are numbered provisions (пункты) rather than headings, then blends that
    with the fusion score.  Not a substitute for a cross-encoder, but strictly
    better than nothing and instant.
    """

    name = "lexical-overlap"

    def rerank(
        self, query: str, items: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return items[:top_k]

        for item in items:
            chunk_tokens = set(tokenize(item.chunk.text))
            overlap = len(query_tokens & chunk_tokens) / len(query_tokens)
            structural_bonus = 0.1 if item.chunk.metadata.paragraph else 0.0
            length_penalty = 0.05 if item.chunk.char_count < 200 else 0.0
            item.rerank_score = overlap + structural_bonus - length_penalty
            # Fusion score is ~1e-2; scale it so both terms are comparable.
            item.score = 0.6 * item.rerank_score + 0.4 * math.tanh(item.score * 50)

        return sorted(items, key=lambda r: r.score, reverse=True)[:top_k]


class CrossEncoderReranker(Reranker):
    """sentence-transformers ``CrossEncoder`` reranker (optional, heavy)."""

    def __init__(self, model_name: str, device: str = "") -> None:
        from sentence_transformers import CrossEncoder

        logger.info("Loading cross-encoder reranker %s", model_name)
        self._model = CrossEncoder(model_name, device=device or None, max_length=512)
        self.name = model_name

    def rerank(
        self, query: str, items: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not items:
            return []
        pairs = [(query, item.chunk.text) for item in items]
        scores = self._model.predict(pairs, show_progress_bar=False)
        for item, score in zip(items, scores, strict=True):
            item.rerank_score = float(score)
            item.score = float(score)
        return sorted(items, key=lambda r: r.score, reverse=True)[:top_k]


_cached: Reranker | None = None


def get_reranker(settings: Settings | None = None) -> Reranker:
    """Return the configured reranker, falling back safely."""
    global _cached
    settings = settings or get_settings()
    if not settings.rerank_enabled:
        return LexicalOverlapReranker()
    if _cached is not None:
        return _cached
    try:
        _cached = CrossEncoderReranker(settings.rerank_model, settings.embedding_device)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Cross-encoder unavailable (%s) — falling back to lexical reranking", exc
        )
        _cached = LexicalOverlapReranker()
    return _cached
