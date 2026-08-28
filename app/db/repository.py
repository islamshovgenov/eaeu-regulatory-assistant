"""SQLite persistence layer.

Holds the document registry, chunk metadata, ingestion state and (optionally)
chat sessions.  Deliberately built on the standard-library ``sqlite3`` driver:
the schema is small, fully owned by this module, and adding an ORM would only
add a dependency without adding safety.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

from app.config import SQLITE_FILE, configure_logging
from app.db.models import (
    Chunk,
    ChunkMetadata,
    DocumentRecord,
    DocumentStatus,
    DocumentType,
    DownloadStatus,
    SourceAuthority,
    SourceTier,
    VersionStatus,
)

logger = configure_logging(__name__)

SCHEMA_VERSION = 1

_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    document_id      TEXT PRIMARY KEY,
    title            TEXT NOT NULL,
    short_title      TEXT,
    authority        TEXT,
    document_type    TEXT,
    document_number  TEXT,
    adoption_date    TEXT,
    effective_date   TEXT,
    expiration_date  TEXT,
    status           TEXT,
    version_status   TEXT,
    jurisdiction     TEXT,
    language         TEXT,
    source_url       TEXT,
    page_url         TEXT,
    source_authority TEXT,
    source_tier      TEXT,
    parent_document  TEXT,
    amends_document  TEXT,
    amended_by       TEXT,
    version          TEXT,
    topic            TEXT,
    download_date    TEXT,
    download_status  TEXT,
    sha256           TEXT,
    size_bytes       INTEGER,
    content_type     TEXT,
    local_file       TEXT,
    n_pages          INTEGER,
    n_chunks         INTEGER DEFAULT 0,
    notes            TEXT
);

CREATE INDEX IF NOT EXISTS idx_documents_authority ON documents(source_authority);
CREATE INDEX IF NOT EXISTS idx_documents_status    ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_type      ON documents(document_type);
CREATE INDEX IF NOT EXISTS idx_documents_sha       ON documents(sha256);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    document_id  TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    ordinal      INTEGER,
    appendix     TEXT,
    section      TEXT,
    chapter      TEXT,
    paragraph    TEXT,
    subparagraph TEXT,
    heading      TEXT,
    page_start   INTEGER,
    page_end     INTEGER,
    char_count   INTEGER,
    token_estimate INTEGER,
    text         TEXT NOT NULL,
    payload      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    stage        TEXT,
    status       TEXT,
    details      TEXT
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id   TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    mode         TEXT,
    payload      TEXT
);
"""

_DOC_COLUMNS = (
    "document_id", "title", "short_title", "authority", "document_type",
    "document_number", "adoption_date", "effective_date", "expiration_date",
    "status", "version_status", "jurisdiction", "language", "source_url",
    "page_url", "source_authority", "source_tier", "parent_document",
    "amends_document", "amended_by", "version", "topic", "download_date",
    "download_status", "sha256", "size_bytes", "content_type", "local_file",
    "n_pages", "n_chunks", "notes",
)


# --------------------------------------------------------------------------- #
# (de)serialisation helpers
# --------------------------------------------------------------------------- #


def _dump_date(value: date | datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _load_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _load_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _document_to_row(doc: DocumentRecord) -> tuple[Any, ...]:
    return (
        doc.document_id,
        doc.title,
        doc.short_title,
        doc.authority,
        str(doc.document_type),
        doc.document_number,
        _dump_date(doc.adoption_date),
        _dump_date(doc.effective_date),
        _dump_date(doc.expiration_date),
        str(doc.status),
        str(doc.version_status),
        doc.jurisdiction,
        doc.language,
        doc.source_url,
        doc.page_url,
        str(doc.source_authority),
        str(doc.source_tier),
        doc.parent_document,
        doc.amends_document,
        json.dumps(doc.amended_by, ensure_ascii=False),
        doc.version,
        json.dumps(doc.topic, ensure_ascii=False),
        _dump_date(doc.download_date),
        str(doc.download_status),
        doc.sha256,
        doc.size_bytes,
        doc.content_type,
        doc.local_file,
        doc.n_pages,
        doc.n_chunks,
        doc.notes,
    )


def _row_to_document(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        document_id=row["document_id"],
        title=row["title"],
        short_title=row["short_title"] or "",
        authority=row["authority"] or "",
        document_type=DocumentType(row["document_type"] or "unknown"),
        document_number=row["document_number"] or "",
        adoption_date=_load_date(row["adoption_date"]),
        effective_date=_load_date(row["effective_date"]),
        expiration_date=_load_date(row["expiration_date"]),
        status=DocumentStatus(row["status"] or "unknown"),
        version_status=VersionStatus(
            row["version_status"] or "requires_expert_validation"
        ),
        jurisdiction=row["jurisdiction"] or "",
        language=row["language"] or "ru",
        source_url=row["source_url"] or "",
        page_url=row["page_url"] or "",
        source_authority=SourceAuthority(row["source_authority"] or "OTHER"),
        source_tier=SourceTier(row["source_tier"] or "TIER_1"),
        parent_document=row["parent_document"] or "",
        amends_document=row["amends_document"] or "",
        amended_by=json.loads(row["amended_by"] or "[]"),
        version=row["version"] or "1",
        topic=json.loads(row["topic"] or "[]"),
        download_date=_load_datetime(row["download_date"]),
        download_status=DownloadStatus(row["download_status"] or "ok"),
        sha256=row["sha256"] or "",
        size_bytes=row["size_bytes"] or 0,
        content_type=row["content_type"] or "",
        local_file=row["local_file"] or "",
        n_pages=row["n_pages"],
        n_chunks=row["n_chunks"] or 0,
        notes=row["notes"] or "",
    )


# --------------------------------------------------------------------------- #
# Repository
# --------------------------------------------------------------------------- #


class Repository:
    """Thin, explicit data-access object over the SQLite file."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path else SQLITE_FILE
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialise()

    # -- connection ---------------------------------------------------------
    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialise(self) -> None:
        """Create the schema if it does not exist (idempotent 'migration')."""
        with self.connect() as conn:
            conn.executescript(_SCHEMA)
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) VALUES (?, ?)",
                ("schema_version", str(SCHEMA_VERSION)),
            )

    # -- documents ----------------------------------------------------------
    def upsert_document(self, doc: DocumentRecord) -> None:
        placeholders = ", ".join("?" for _ in _DOC_COLUMNS)
        columns = ", ".join(_DOC_COLUMNS)
        with self.connect() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO documents ({columns}) VALUES ({placeholders})",
                _document_to_row(doc),
            )

    def upsert_documents(self, docs: Iterable[DocumentRecord]) -> int:
        rows = [_document_to_row(d) for d in docs]
        if not rows:
            return 0
        placeholders = ", ".join("?" for _ in _DOC_COLUMNS)
        columns = ", ".join(_DOC_COLUMNS)
        with self.connect() as conn:
            conn.executemany(
                f"INSERT OR REPLACE INTO documents ({columns}) VALUES ({placeholders})",
                rows,
            )
        return len(rows)

    def get_document(self, document_id: str) -> DocumentRecord | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        return _row_to_document(row) if row else None

    def get_document_by_sha256(self, sha256: str) -> DocumentRecord | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE sha256 = ? LIMIT 1", (sha256,)
            ).fetchone()
        return _row_to_document(row) if row else None

    def list_documents(
        self,
        source_authority: str | None = None,
        document_type: str | None = None,
        search: str | None = None,
    ) -> list[DocumentRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if source_authority:
            clauses.append("source_authority = ?")
            params.append(source_authority)
        if document_type:
            clauses.append("document_type = ?")
            params.append(document_type)
        if search:
            clauses.append("(title LIKE ? OR short_title LIKE ? OR document_number LIKE ?)")
            like = f"%{search}%"
            params.extend([like, like, like])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM documents {where} ORDER BY source_tier, document_number",
                params,
            ).fetchall()
        return [_row_to_document(r) for r in rows]

    def delete_documents_not_in(self, keep_ids: Iterable[str]) -> list[str]:
        """Remove registry entries that the current discovery no longer yields.

        Used after a *full* download run so that documents dropped from the
        official source (or harvested by mistake from a listing page) do not
        linger in the index.  The downloaded files themselves are left on disk,
        so provenance of what was once fetched is preserved.
        """
        keep = set(keep_ids)
        with self.connect() as conn:
            rows = conn.execute("SELECT document_id FROM documents").fetchall()
            stale = [r["document_id"] for r in rows if r["document_id"] not in keep]
            if stale:
                placeholders = ",".join("?" for _ in stale)
                conn.execute(
                    f"DELETE FROM chunks WHERE document_id IN ({placeholders})", stale
                )
                conn.execute(
                    f"DELETE FROM documents WHERE document_id IN ({placeholders})", stale
                )
        if stale:
            logger.info("Pruned %d stale documents from the registry", len(stale))
        return stale

    def document_numbers(self) -> set[str]:
        """All act numbers present in the corpus (used to spot invented ones)."""
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT document_number FROM documents "
                "WHERE document_number IS NOT NULL AND document_number != ''"
            ).fetchall()
        return {r["document_number"] for r in rows}

    def set_document_chunk_count(self, document_id: str, n_chunks: int) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE documents SET n_chunks = ? WHERE document_id = ?",
                (n_chunks, document_id),
            )

    # -- chunks -------------------------------------------------------------
    def replace_chunks(self, document_id: str, chunks: list[Chunk]) -> int:
        rows = [
            (
                c.chunk_id,
                c.document_id,
                ordinal,
                c.metadata.appendix,
                c.metadata.section,
                c.metadata.chapter,
                c.metadata.paragraph,
                c.metadata.subparagraph,
                c.metadata.heading,
                c.metadata.page_start,
                c.metadata.page_end,
                c.char_count,
                c.token_estimate,
                c.text,
                json.dumps(c.to_payload(), ensure_ascii=False),
            )
            for ordinal, c in enumerate(chunks)
        ]
        with self.connect() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            if rows:
                conn.executemany(
                    """INSERT OR REPLACE INTO chunks (
                        chunk_id, document_id, ordinal, appendix, section, chapter,
                        paragraph, subparagraph, heading, page_start, page_end,
                        char_count, token_estimate, text, payload)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    rows,
                )
            conn.execute(
                "UPDATE documents SET n_chunks = ? WHERE document_id = ?",
                (len(rows), document_id),
            )
        return len(rows)

    def iter_chunks(self) -> Iterator[Chunk]:
        with self.connect() as conn:
            for row in conn.execute("SELECT payload FROM chunks ORDER BY chunk_id"):
                yield Chunk.from_payload(json.loads(row["payload"]))

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT payload FROM chunks WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()
        return Chunk.from_payload(json.loads(row["payload"])) if row else None

    def count_chunks(self) -> int:
        with self.connect() as conn:
            return int(conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"])

    def count_documents(self) -> int:
        with self.connect() as conn:
            return int(
                conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
            )

    # -- statistics ---------------------------------------------------------
    def counts_by(self, column: str) -> dict[str, int]:
        allowed = {
            "source_authority",
            "document_type",
            "status",
            "version_status",
            "source_tier",
            "download_status",
        }
        if column not in allowed:
            raise ValueError(f"Unsupported grouping column: {column}")
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT {column} AS k, COUNT(*) AS n FROM documents GROUP BY {column}"
            ).fetchall()
        return {(r["k"] or "unknown"): r["n"] for r in rows}

    def counts_by_year(self) -> dict[str, int]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT substr(adoption_date, 1, 4) AS y, COUNT(*) AS n "
                "FROM documents GROUP BY y ORDER BY y"
            ).fetchall()
        return {(r["y"] or "unknown"): r["n"] for r in rows}

    def total_tokens(self) -> int:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(token_estimate), 0) AS t FROM chunks"
            ).fetchone()
        return int(row["t"])

    def last_update(self) -> datetime | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT MAX(download_date) AS d FROM documents"
            ).fetchone()
        return _load_datetime(row["d"]) if row and row["d"] else None

    # -- ingestion runs -----------------------------------------------------
    def start_run(self, stage: str) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO ingestion_runs (started_at, stage, status) VALUES (?,?,?)",
                (datetime.now().isoformat(timespec="seconds"), stage, "running"),
            )
            return int(cur.lastrowid or 0)

    def finish_run(self, run_id: int, status: str, details: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE ingestion_runs SET finished_at = ?, status = ?, details = ? "
                "WHERE run_id = ?",
                (
                    datetime.now().isoformat(timespec="seconds"),
                    status,
                    json.dumps(details, ensure_ascii=False, default=str),
                    run_id,
                ),
            )

    def last_run(self) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM ingestion_runs ORDER BY run_id DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None


__all__ = ["Repository", "ChunkMetadata"]
