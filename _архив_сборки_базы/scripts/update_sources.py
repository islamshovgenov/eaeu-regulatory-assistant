"""Incremental update of the knowledge base.

    python scripts/update_sources.py

Steps:

1. re-crawl the registered sources;
2. compare against the current registry;
3. download new / changed documents (unchanged files are left untouched);
4. update metadata;
5. parse and chunk the changed documents;
6. rebuild the indexes.

Provenance is preserved: previously downloaded files are never deleted, and a
changed document is stored as a *new* file (its name carries the content hash),
so the earlier revision remains auditable on disk and in ``documents.csv``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import INN_VOCABULARY_JSON, configure_logging, ensure_directories
from app.db.models import DownloadStatus
from app.db.repository import Repository
from app.regulatory.inn import load_vocabulary
from ingestion.chunker import ChunkingConfig, build_chunks
from ingestion.discover_sources import discover
from ingestion.downloader import DocumentDownloader, write_documents_csv
from ingestion.indexer import build_indexes
from ingestion.parser import parse_file
from ingestion.run_ingestion import stage_build_inn_vocabulary

logger = configure_logging("scripts.update_sources")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Инкрементальное обновление базы")
    parser.add_argument("--ocr", action="store_true", help="разрешить OCR")
    parser.add_argument(
        "--no-reindex", action="store_true", help="не пересобирать индексы"
    )
    args = parser.parse_args(argv)

    ensure_directories()
    repository = Repository()
    run_id = repository.start_run("update_sources")
    started = datetime.now()

    try:
        known_before = {d.document_id: d.sha256 for d in repository.list_documents()}

        print("1/6 Обход зарегистрированных источников…")
        discovered = discover()
        print(f"      найдено кандидатов: {len(discovered)}")

        print("2/6 Загрузка новых и изменившихся документов…")
        downloader = DocumentDownloader(repository=repository)
        outcomes = downloader.download_all(discovered, prune_registry=True)

        changed = [o.record for o in outcomes if o.changed]
        new_documents = [
            o.record for o in outcomes if o.record.document_id not in known_before
        ]
        print(
            f"      изменено/загружено: {len(changed)}, новых: {len(new_documents)}, "
            f"без изменений: {sum(1 for o in outcomes if o.reason == 'unchanged')}"
        )

        print("3/6 Обновление словаря МНН…")
        stage_build_inn_vocabulary(repository)
        load_vocabulary(INN_VOCABULARY_JSON)

        print("4/6 Парсинг и чанкинг изменившихся документов…")
        config = ChunkingConfig.from_settings()
        reprocessed = 0
        for record in {r.document_id: r for r in changed + new_documents}.values():
            if record.download_status != DownloadStatus.OK:
                continue
            path = Path(record.local_file)
            if not path.exists():
                continue
            try:
                parsed = parse_file(path, allow_ocr=args.ocr)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Parse failed for %s: %s", record.document_id, exc)
                continue
            record.n_pages = parsed.n_pages
            chunks = build_chunks(record, parsed, config)
            repository.replace_chunks(record.document_id, chunks)
            repository.upsert_document(record)
            reprocessed += 1
        print(f"      переобработано документов: {reprocessed}")

        print("5/6 Обновление реестра documents.csv…")
        write_documents_csv(repository.list_documents())

        stats: dict[str, object] = {
            "discovered": len(discovered),
            "changed": len(changed),
            "new": len(new_documents),
            "reprocessed": reprocessed,
        }

        if args.no_reindex:
            print("6/6 Пересборка индексов пропущена (--no-reindex).")
        elif reprocessed or not known_before:
            print("6/6 Пересборка индексов…")
            stats.update(build_indexes(list(repository.iter_chunks())))
        else:
            print("6/6 Изменений нет — индексы не пересобирались.")

        stats["duration_seconds"] = round((datetime.now() - started).total_seconds(), 1)
        repository.finish_run(run_id, "ok", stats)
    except Exception as exc:  # noqa: BLE001
        repository.finish_run(run_id, "failed", {"error": str(exc)})
        logger.exception("Update failed")
        return 1

    print("\n=== ОБНОВЛЕНИЕ ЗАВЕРШЕНО ===")
    for key, value in stats.items():
        print(f"  {key:<20} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
