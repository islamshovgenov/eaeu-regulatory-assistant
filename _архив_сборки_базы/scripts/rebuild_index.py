"""Rebuild the search indexes from the chunks already stored in SQLite.

    python scripts/rebuild_index.py

Use this after changing the embedding model or when the indexes drift out of
sync with the database. Documents are not re-downloaded and not re-parsed.
Add ``--rechunk`` to re-run parsing and chunking as well.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import INN_VOCABULARY_JSON, configure_logging, get_settings
from app.db.repository import Repository
from app.regulatory.inn import load_vocabulary
from ingestion.indexer import build_indexes
from ingestion.run_ingestion import stage_build_inn_vocabulary, stage_parse_and_chunk

logger = configure_logging("scripts.rebuild_index")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Пересборка поисковых индексов")
    parser.add_argument(
        "--rechunk", action="store_true", help="заново распарсить и разбить документы"
    )
    parser.add_argument("--ocr", action="store_true", help="разрешить OCR при --rechunk")
    args = parser.parse_args(argv)

    settings = get_settings()
    repository = Repository()
    run_id = repository.start_run("rebuild_index")

    try:
        if args.rechunk:
            stage_build_inn_vocabulary(repository)
            chunks = stage_parse_and_chunk(repository, args.ocr, None)
        else:
            load_vocabulary(INN_VOCABULARY_JSON)
            chunks = list(repository.iter_chunks())

        if not chunks:
            print("В базе нет чанков. Запустите: python ingestion/run_ingestion.py")
            repository.finish_run(run_id, "failed", {"error": "no chunks"})
            return 1

        print(
            f"Пересборка индексов: {len(chunks)} чанков, "
            f"эмбеддинги = {settings.embedding_provider}"
        )
        stats = build_indexes(chunks, settings, recreate=True)
        repository.finish_run(run_id, "ok", stats)
    except Exception as exc:  # noqa: BLE001
        repository.finish_run(run_id, "failed", {"error": str(exc)})
        logger.exception("Rebuild failed")
        return 1

    print("=== ГОТОВО ===")
    for key, value in stats.items():
        print(f"  {key:<12} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
