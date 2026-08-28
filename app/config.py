"""Central configuration for the EAEU Regulatory Assistant.

All settings are read from environment variables / ``.env``.  Secrets (API keys)
are *never* hard-coded and *never* logged.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REGISTRY_DIR = DATA_DIR / "registry"
EVAL_DIR = DATA_DIR / "eval"
DATABASE_DIR = PROJECT_ROOT / "database"
LOGS_DIR = PROJECT_ROOT / "logs"

INN_VOCABULARY_JSON = PROCESSED_DIR / "inn_vocabulary.json"
CORPUS_STATS_JSON = PROCESSED_DIR / "corpus_statistics.json"
DOCUMENTS_CSV = REGISTRY_DIR / "documents.csv"
BM25_INDEX_FILE = DATABASE_DIR / "bm25_index.pkl"
SQLITE_FILE = DATABASE_DIR / "eaeu.sqlite3"
QDRANT_LOCAL_PATH = DATABASE_DIR / "qdrant"

# Используются только кодом пересборки базы из `_архив_сборки_базы/`.
CHUNKS_JSONL = PROCESSED_DIR / "chunks.jsonl"
SOURCES_YAML = REGISTRY_DIR / "sources.yaml"
DISCOVERED_JSON = REGISTRY_DIR / "discovered.json"

# ``load_dotenv`` is a no-op when the file is missing.
load_dotenv(PROJECT_ROOT / ".env")


def ensure_directories() -> None:
    """Create every directory the application writes to."""
    for path in (
        DATA_DIR,
        RAW_DIR,
        RAW_DIR / "eaeu",
        RAW_DIR / "ich",
        RAW_DIR / "ema",
        RAW_DIR / "who",
        PROCESSED_DIR,
        REGISTRY_DIR,
        EVAL_DIR,
        EVAL_DIR / "cases",
        DATABASE_DIR,
        LOGS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #

EmbeddingProvider = Literal["local", "openai"]
LLMProvider = Literal["anthropic", "openai", "none"]


class Settings(BaseSettings):
    """Runtime configuration, populated from the environment."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -- LLM ----------------------------------------------------------------
    llm_provider: LLMProvider = "anthropic"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    #: Override the API endpoint. Lets the OpenAI client talk to any
    #: OpenAI-compatible service (a regional provider, a gateway, or a local
    #: server such as Ollama/LM Studio) without touching the code.
    openai_base_url: str = ""
    anthropic_base_url: str = ""
    model_name: str = ""
    llm_temperature: float = 0.0
    llm_max_tokens: int = 4096
    llm_timeout_seconds: int = 180

    # -- embeddings ---------------------------------------------------------
    embedding_provider: EmbeddingProvider = "local"
    embedding_model: str = "intfloat/multilingual-e5-base"
    openai_embedding_model: str = "text-embedding-3-large"
    embedding_batch_size: int = 16
    embedding_device: str = ""  # "" -> auto ("cuda" when available, else "cpu")

    # -- vector store -------------------------------------------------------
    qdrant_mode: Literal["local", "server"] = "local"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "eaeu_regulatory_chunks"

    # -- retrieval ----------------------------------------------------------
    vector_top_k: int = 40
    bm25_top_k: int = 40
    final_top_k: int = 12
    rrf_k: int = 60
    rrf_vector_weight: float = 1.0
    rrf_bm25_weight: float = 1.0
    min_retrieval_score: float = 0.0

    # -- reranking ----------------------------------------------------------
    rerank_enabled: bool = False
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_candidates: int = 40

    # ----------------------------------------------------------------------
    # Настройки только для пересборки базы знаний
    # ----------------------------------------------------------------------
    # При работе приложения не используются: база уже собрана, а код сборки
    # вынесен в `_архив_сборки_базы/`.  Оставлены здесь, чтобы архивный код
    # заработал сразу, если его вернут в проект.  Значения — те, с которыми
    # фактически собран текущий корпус.

    #: OCR сканов (Tesseract): путь к бинарю, разрешение, языки.
    tesseract_cmd: str = ""
    ocr_dpi: int = 200
    ocr_languages: str = "rus+eng"

    #: Нарезка документов на фрагменты.
    chunk_target_chars: int = 1600
    chunk_max_chars: int = 2600
    chunk_overlap_chars: int = 200

    #: Скачивание документов с портала ЕАЭС.
    http_user_agent: str = (
        "EAEU-Regulatory-Assistant/1.0 (academic research project; "
        "contact: student@example.edu)"
    )
    http_timeout_seconds: int = 90
    http_retries: int = 3
    http_rate_limit_seconds: float = 1.0
    download_max_bytes: int = 120 * 1024 * 1024

    # -- behaviour ----------------------------------------------------------
    supplementary_sources_enabled: bool = True
    log_level: str = "INFO"

    @field_validator("model_name")
    @classmethod
    def _default_model(cls, value: str) -> str:
        return value.strip()

    # -- helpers ------------------------------------------------------------
    def resolved_model_name(self) -> str:
        """Return the model id, falling back to a sensible per-provider default."""
        if self.model_name:
            return self.model_name
        if self.llm_provider == "anthropic":
            return "claude-sonnet-4-5-20250929"
        if self.llm_provider == "openai":
            return "gpt-4o"
        return ""

    def has_llm_credentials(self) -> bool:
        if self.llm_provider == "anthropic":
            return bool(self.anthropic_api_key)
        if self.llm_provider == "openai":
            return bool(self.openai_api_key)
        return False

    def qdrant_location(self) -> dict[str, object]:
        """Keyword arguments for :class:`qdrant_client.QdrantClient`."""
        if self.qdrant_mode == "server":
            kwargs: dict[str, object] = {"url": self.qdrant_url}
            if self.qdrant_api_key:
                kwargs["api_key"] = self.qdrant_api_key
            return kwargs
        QDRANT_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        return {"path": str(QDRANT_LOCAL_PATH)}


_settings: Settings | None = None


def get_settings(refresh: bool = False) -> Settings:
    """Return the process-wide :class:`Settings` singleton."""
    global _settings
    if _settings is None or refresh:
        _settings = Settings()
    return _settings


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

_SECRET_ENV_KEYS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "QDRANT_API_KEY",
)


class _SecretRedactingFilter(logging.Filter):
    """Replace any occurrence of a known secret in a log record with ``***``."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D102
        secrets = [os.environ.get(key, "") for key in _SECRET_ENV_KEYS]
        secrets = [s for s in secrets if len(s) >= 8]
        if not secrets:
            return True
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - defensive
            return True
        redacted = message
        for secret in secrets:
            redacted = redacted.replace(secret, "***REDACTED***")
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


_logging_configured = False


def configure_logging(name: str = "app", level: str | None = None) -> logging.Logger:
    """Configure root logging once (console + rotating file) and return a logger."""
    global _logging_configured
    settings = get_settings()
    resolved_level = (level or settings.log_level).upper()

    if not _logging_configured:
        ensure_directories()
        root = logging.getLogger()
        root.setLevel(resolved_level)
        root.handlers.clear()

        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
        )
        redactor = _SecretRedactingFilter()

        console = logging.StreamHandler(stream=sys.stderr)
        console.setFormatter(fmt)
        console.addFilter(redactor)
        root.addHandler(console)

        file_handler = logging.handlers.RotatingFileHandler(
            LOGS_DIR / "eaeu_assistant.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(fmt)
        file_handler.addFilter(redactor)
        root.addHandler(file_handler)

        # Third-party noise.
        for noisy in ("urllib3", "httpx", "httpcore", "sentence_transformers"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        _logging_configured = True

    return logging.getLogger(name)
