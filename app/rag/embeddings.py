"""Pluggable embedding providers.

Two interchangeable back-ends are supported, selected with
``EMBEDDING_PROVIDER``:

``local``
    ``sentence-transformers`` running on the machine — no API key, no cost.
    Default model: **intfloat/multilingual-e5-base** (see README for the
    rationale: strong Russian retrieval quality at 768 dimensions and ~280 MB,
    trained with the asymmetric ``query:``/``passage:`` prefixes that this
    corpus benefits from).

``openai``
    ``text-embedding-3-large`` — requires ``OPENAI_API_KEY``.

Adding another provider means implementing :class:`EmbeddingProvider` and
registering it in :func:`get_embedding_provider`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

#: e5-family models require these prefixes; other models must not get them.
_E5_QUERY_PREFIX = "query: "
_E5_PASSAGE_PREFIX = "passage: "


class EmbeddingProvider(ABC):
    """Common interface for every embedding back-end."""

    name: str = "abstract"
    dimension: int = 0

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed corpus passages. Returns ``(len(texts), dimension)`` float32."""

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single search query. Returns ``(dimension,)`` float32."""

    def describe(self) -> str:
        return f"{self.name} (dim={self.dimension})"


# --------------------------------------------------------------------------- #
# Local sentence-transformers
# --------------------------------------------------------------------------- #


class LocalEmbeddingProvider(EmbeddingProvider):
    """sentence-transformers back-end (default, fully offline after download)."""

    def __init__(self, model_name: str, device: str = "", batch_size: int = 16) -> None:
        from sentence_transformers import SentenceTransformer

        resolved_device = device or _auto_device()
        logger.info("Loading embedding model %s on %s", model_name, resolved_device)
        self._model = SentenceTransformer(model_name, device=resolved_device)
        self.name = model_name
        self.batch_size = batch_size
        self.dimension = int(self._model.get_sentence_embedding_dimension())
        self._uses_e5_prefixes = "e5" in model_name.lower()

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        prepared = [f"{prefix}{t}" if self._uses_e5_prefixes else t for t in texts]
        vectors = self._model.encode(
            prepared,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(prepared) > 256,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return self._encode(texts, _E5_PASSAGE_PREFIX)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([text], _E5_QUERY_PREFIX)[0]


def _auto_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:  # pragma: no cover
        pass
    return "cpu"


# --------------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------------- #


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embeddings back-end."""

    _DIMENSIONS = {
        "text-embedding-3-large": 3072,
        "text-embedding-3-small": 1536,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model_name: str, api_key: str, batch_size: int = 64) -> None:
        from openai import OpenAI

        if not api_key:
            raise RuntimeError(
                "EMBEDDING_PROVIDER=openai требует OPENAI_API_KEY в .env"
            )
        self._client = OpenAI(api_key=api_key)
        self.name = model_name
        self.batch_size = batch_size
        self.dimension = self._DIMENSIONS.get(model_name, 3072)

    def _call(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self._client.embeddings.create(model=self.name, input=batch)
            vectors.extend(item.embedding for item in response.data)
        array = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        return array / np.clip(norms, 1e-12, None)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return self._call(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._call([text])[0]


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=4)
def _build(provider: str, model: str, device: str, batch: int, key: str) -> EmbeddingProvider:
    if provider == "openai":
        return OpenAIEmbeddingProvider(model, key, batch)
    return LocalEmbeddingProvider(model, device, batch)


def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    """Return the configured provider (cached per configuration)."""
    settings = settings or get_settings()
    if settings.embedding_provider == "openai":
        return _build(
            "openai",
            settings.openai_embedding_model,
            "",
            64,
            settings.openai_api_key,
        )
    return _build(
        "local",
        settings.embedding_model,
        settings.embedding_device,
        settings.embedding_batch_size,
        "",
    )
