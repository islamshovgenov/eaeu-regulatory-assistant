"""Corpus statistics.

    python scripts/corpus_stats.py

Prints a summary and writes ``data/processed/corpus_statistics.json``.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import CORPUS_STATS_JSON, configure_logging, ensure_directories
from app.db.repository import Repository

logger = configure_logging("scripts.corpus_stats")


def collect(repository: Repository | None = None) -> dict[str, object]:
    repository = repository or Repository()
    documents = repository.list_documents()

    topics: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    total_bytes = 0
    for document in documents:
        topics.update(document.topic)
        languages[document.language] += 1
        total_bytes += document.size_bytes

    chunk_count = repository.count_chunks()
    token_total = repository.total_tokens()
    last_update = repository.last_update()

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "documents": len(documents),
        "chunks": chunk_count,
        "tokens_estimated": token_total,
        "avg_chunks_per_document": (
            round(chunk_count / len(documents), 2) if documents else 0
        ),
        "avg_tokens_per_chunk": (
            round(token_total / chunk_count, 1) if chunk_count else 0
        ),
        "raw_bytes": total_bytes,
        "documents_by_authority": repository.counts_by("source_authority"),
        "documents_by_type": repository.counts_by("document_type"),
        "documents_by_status": repository.counts_by("status"),
        "documents_by_version_status": repository.counts_by("version_status"),
        "documents_by_tier": repository.counts_by("source_tier"),
        "documents_by_download_status": repository.counts_by("download_status"),
        "documents_by_year": repository.counts_by_year(),
        "documents_by_language": dict(languages),
        "documents_by_topic": dict(topics.most_common()),
        "last_update": last_update.isoformat() if last_update else None,
        "last_ingestion_run": repository.last_run(),
    }


def main() -> int:
    ensure_directories()
    stats = collect()
    CORPUS_STATS_JSON.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    print("=== CORPUS STATISTICS ===")
    print(f"  Документов:            {stats['documents']}")
    print(f"  Чанков:                {stats['chunks']}")
    print(f"  Токенов (оценка):      {stats['tokens_estimated']:,}".replace(",", " "))
    print(f"  Чанков на документ:    {stats['avg_chunks_per_document']}")
    print(f"  Токенов на чанк:       {stats['avg_tokens_per_chunk']}")
    for title, key in (
        ("По органу", "documents_by_authority"),
        ("По типу", "documents_by_type"),
        ("По статусу", "documents_by_status"),
        ("По статусу редакции", "documents_by_version_status"),
        ("По уровню источника", "documents_by_tier"),
        ("По году", "documents_by_year"),
        ("По тематике", "documents_by_topic"),
    ):
        print(f"\n  {title}:")
        for name, count in sorted(
            stats[key].items(), key=lambda kv: kv[1], reverse=True  # type: ignore[union-attr]
        ):
            print(f"    {name:<38} {count}")
    print(f"\n  Сохранено: {CORPUS_STATS_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
