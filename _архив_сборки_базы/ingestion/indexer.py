"""Index construction: Qdrant vector index + BM25 keyword index.

The vector payload carries the **full chunk metadata**, not just text, so the
retriever can filter by authority/status/INN and build citations without a
second lookup.  The BM25 index is persisted alongside it and rebuilt from the
same chunk list, guaranteeing the two views never diverge.
"""

from __future__ import annotations

import logging
import pickle
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from app.config import BM25_INDEX_FILE, Settings, get_settings
from app.db.models import Chunk
from ingestion.embeddings import EmbeddingProvider, get_embedding_provider
from ingestion.normalizer import normalize_for_search

logger = logging.getLogger(__name__)

#: Tokeniser tuned for regulatory Russian: keeps numbers with separators
#: ("90%", "0-t", "AUC(0-t)" -> "auc", "0", "t"), keeps Latin acronyms.
_TOKEN_RE = re.compile(r"[а-яa-z]+|\d+(?:[.,]\d+)?")

#: Very common Russian function words — removed to keep BM25 discriminative.
_STOPWORDS = frozenset(
    """
    и в во не что он на я с со как а то все она так его но да ты к у же вы за
    бы по только ее мне было вот от меня еще нет о из ему теперь когда даже ну
    вдруг ли если уже или ни быть был него до вас нибудь опять уж вам ведь там
    потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была
    сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому
    этого какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были
    куда зачем всех никогда можно при наконец два об другой хоть после над больше
    тот через эти нас про всего них какая много разве три эту моя впрочем хорошо
    свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно
    всю между также при этом является том числе
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Tokens for BM25 — lower-cased, stop-word filtered, ё→е normalised."""
    normalised = normalize_for_search(text)
    return [
        token
        for token in _TOKEN_RE.findall(normalised)
        if token not in _STOPWORDS and len(token) > 1
    ]


# --------------------------------------------------------------------------- #
# BM25
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class BM25Index:
    """Persisted BM25 index over chunk texts."""

    chunk_ids: list[str]
    model: Any  # rank_bm25.BM25Okapi

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "BM25Index":
        from rank_bm25 import BM25Okapi

        corpus = [tokenize(chunk.text) for chunk in chunks]
        # BM25Okapi cannot handle a completely empty corpus.
        if not corpus:
            corpus = [["-"]]
            ids = ["-"]
        else:
            ids = [chunk.chunk_id for chunk in chunks]
        logger.info("Building BM25 index over %d chunks", len(ids))
        return cls(chunk_ids=ids, model=BM25Okapi(corpus))

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.model.get_scores(tokens)
        order = np.argsort(scores)[::-1][:top_k]
        return [
            (self.chunk_ids[i], float(scores[i]))
            for i in order
            if scores[i] > 0
        ]

    def save(self, path: Path = BM25_INDEX_FILE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump({"chunk_ids": self.chunk_ids, "model": self.model}, handle)
        logger.info("BM25 index saved -> %s", path)

    @classmethod
    def load(cls, path: Path = BM25_INDEX_FILE) -> "BM25Index | None":
        if not path.exists():
            return None
        try:
            with open(path, "rb") as handle:
                data = pickle.load(handle)
            return cls(chunk_ids=data["chunk_ids"], model=data["model"])
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load BM25 index %s: %s", path, exc)
            return None


# --------------------------------------------------------------------------- #
# Qdrant
# --------------------------------------------------------------------------- #


def _point_id(chunk_id: str) -> str:
    """Qdrant needs a UUID/int id; derive a stable UUID5 from the chunk id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class VectorIndex:
    """Thin wrapper over a Qdrant collection holding the chunk payloads."""

    def __init__(
        self,
        settings: Settings | None = None,
        provider: EmbeddingProvider | None = None,
    ) -> None:
        from qdrant_client import QdrantClient

        self.settings = settings or get_settings()
        self.provider = provider or get_embedding_provider(self.settings)
        self.collection = self.settings.qdrant_collection
        self.client = QdrantClient(**self.settings.qdrant_location())

    # -- lifecycle ----------------------------------------------------------
    def recreate(self) -> None:
        from qdrant_client import models

        logger.info(
            "Creating Qdrant collection %s (dim=%d, cosine)",
            self.collection,
            self.provider.dimension,
        )
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=self.provider.dimension, distance=models.Distance.COSINE
            ),
        )

    def exists(self) -> bool:
        try:
            return self.client.collection_exists(self.collection)
        except Exception:  # noqa: BLE001
            return False

    def count(self) -> int:
        if not self.exists():
            return 0
        return int(self.client.count(self.collection, exact=True).count)

    # -- writing ------------------------------------------------------------
    def upsert_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> int:
        from qdrant_client import models

        total = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = self.provider.embed_documents([c.text for c in batch])
            points = [
                models.PointStruct(
                    id=_point_id(chunk.chunk_id),
                    vector=vector.tolist(),
                    payload=chunk.to_payload(),
                )
                for chunk, vector in zip(batch, vectors, strict=True)
            ]
            self.client.upsert(collection_name=self.collection, points=points)
            total += len(points)
            logger.info("  indexed %d/%d chunks", total, len(chunks))
        return total

    def prune_missing(self, valid_chunk_ids: set[str]) -> int:
        """Delete vectors whose chunk no longer exists in the database.

        Guards against a collection that survived an interrupted run: Qdrant
        keeps points from the previous chunking, which would then be retrieved
        and cited even though the underlying chunk is gone.
        """
        if not self.exists():
            return 0
        stale: list[str] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=2000,
                offset=offset,
                with_payload=["chunk_id"],
                with_vectors=False,
            )
            for point in points:
                chunk_id = (point.payload or {}).get("chunk_id")
                if chunk_id and chunk_id not in valid_chunk_ids:
                    stale.append(str(point.id))
            if offset is None:
                break
        if stale:
            self.client.delete(collection_name=self.collection, points_selector=stale)
            logger.info("Removed %d stale vectors left by an earlier run", len(stale))
        return len(stale)

    # -- reading ------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int,
        query_filter: Any | None = None,
    ) -> list[tuple[Chunk, float]]:
        if not self.exists():
            return []
        vector = self.provider.embed_query(query)
        hits = self.client.query_points(
            collection_name=self.collection,
            query=vector.tolist(),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        ).points
        results: list[tuple[Chunk, float]] = []
        for hit in hits:
            if not hit.payload:
                continue
            try:
                results.append((Chunk.from_payload(hit.payload), float(hit.score)))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Malformed payload skipped: %s", exc)
        return results

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:  # noqa: BLE001  - local mode may already be closed
            pass


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def build_indexes(
    chunks: Iterable[Chunk],
    settings: Settings | None = None,
    recreate: bool = True,
) -> dict[str, int]:
    """Build both indexes from the same chunk list."""
    chunk_list = list(chunks)
    settings = settings or get_settings()

    bm25 = BM25Index.build(chunk_list)
    bm25.save()

    vector_index = VectorIndex(settings)
    try:
        if recreate or not vector_index.exists():
            vector_index.recreate()
        indexed = vector_index.upsert_chunks(chunk_list)
        pruned = vector_index.prune_missing({c.chunk_id for c in chunk_list})
        total_vectors = vector_index.count()
    finally:
        vector_index.close()

    return {
        "chunks": len(chunk_list),
        "vectors": total_vectors,
        "indexed": indexed,
        "pruned_stale_vectors": pruned,
        "bm25": len(bm25.chunk_ids),
    }
