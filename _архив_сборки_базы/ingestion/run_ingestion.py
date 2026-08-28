"""End-to-end ingestion pipeline.

    discover -> download -> parse -> chunk -> embed -> index

Every stage is idempotent and can be run in isolation::

    python ingestion/run_ingestion.py                 # full pipeline
    python ingestion/run_ingestion.py --skip-discover # reuse discovered.json
    python ingestion/run_ingestion.py --only-index    # re-chunk + re-index
    python ingestion/run_ingestion.py --limit 20      # quick smoke run
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import (
    CHUNKS_JSONL,
    INN_VOCABULARY_JSON,
    configure_logging,
    ensure_directories,
    get_settings,
)
from app.db.models import Chunk, DocumentRecord, DocumentType, DownloadStatus
from app.db.repository import Repository
from app.regulatory.inn import (
    extract_inns_from_title,
    known_inns,
    register_inns,
    save_vocabulary,
)
from ingestion.chunker import ChunkingConfig, build_chunks, build_metadata_only_chunk
from ingestion.discover_sources import discover, load_discovered
from ingestion.downloader import DocumentDownloader, write_documents_csv
from ingestion.indexer import build_indexes
from ingestion.normalizer import detect_script, is_russian_text
from ingestion.parser import UnsupportedFormat, parse_file

logger = configure_logging("ingestion")


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #


def stage_discover(force: bool) -> int:
    existing = load_discovered()
    if existing and not force:
        logger.info("Reusing %d discovered files (use --rediscover to refresh)", len(existing))
        return len(existing)
    found = discover()
    return len(found)


def stage_download(limit: int | None) -> list[DocumentRecord]:
    items = load_discovered()
    if limit:
        items = items[:limit]
    if not items:
        raise RuntimeError(
            "Список источников пуст. Запустите: python -m ingestion.discover_sources"
        )
    downloader = DocumentDownloader()
    # Pruning is only safe for a complete run: a --limit subset must not be
    # mistaken for "everything the sources still publish".
    outcomes = downloader.download_all(items, prune_registry=limit is None)
    return [o.record for o in outcomes]


def stage_build_inn_vocabulary(repository: Repository) -> int:
    """Harvest INNs from Expert Committee recommendation titles.

    Those documents are titled after the substance they concern
    ("… с МНН «эстриол» …"), which gives an authoritative, corpus-derived INN
    vocabulary — far better than a hard-coded list — for INN-specific search.
    """
    names: list[str] = []
    for document in repository.list_documents():
        if document.document_type == DocumentType.EXPERT_COMMITTEE_RECOMMENDATION:
            names.extend(extract_inns_from_title(document.title))
    added = register_inns(names)
    save_vocabulary(INN_VOCABULARY_JSON)
    logger.info(
        "INN vocabulary: +%d new names, %d total -> %s",
        added,
        len(known_inns()),
        INN_VOCABULARY_JSON,
    )
    return len(known_inns())


def stage_parse_and_chunk(
    repository: Repository, allow_ocr: bool, limit: int | None
) -> list[Chunk]:
    documents = repository.list_documents()
    documents = [
        d
        for d in documents
        if d.download_status == DownloadStatus.OK and d.local_file and Path(d.local_file).exists()
    ]
    if limit:
        documents = documents[:limit]

    config = ChunkingConfig.from_settings()
    all_chunks: list[Chunk] = []
    failures: list[tuple[str, str]] = []
    skipped_language = 0
    scanned = 0

    for index, document in enumerate(documents, start=1):
        logger.info("[%d/%d] parsing %s", index, len(documents), document.document_id)
        try:
            parsed = parse_file(document.local_file, allow_ocr=allow_ocr)
        except (UnsupportedFormat, FileNotFoundError) as exc:
            logger.warning("  skipped: %s", exc)
            failures.append((document.document_id, str(exc)))
            continue
        except Exception as exc:  # noqa: BLE001 - a corrupt file must not stop ingestion
            logger.error("  parse error: %s", exc)
            failures.append((document.document_id, str(exc)))
            continue

        if parsed.warnings:
            for warning in parsed.warnings:
                logger.warning("  %s", warning)
            document.notes = (document.notes + " " + " ".join(parsed.warnings)).strip()

        document.n_pages = parsed.n_pages

        # -- language gate --------------------------------------------------
        # EAEU acts are published in five official languages under similar file
        # names; only the Russian text belongs in this corpus.
        text_available = len(parsed.normalized_text.strip()) >= 200
        if text_available and document.language == "ru" and not is_russian_text(
            parsed.normalized_text
        ):
            script = detect_script(parsed.normalized_text)
            logger.warning("  not Russian (script=%s) — excluded from index", script)
            document.language = script
            document.n_chunks = 0
            document.notes = (
                document.notes
                + f" Исключён из индекса: язык документа не русский (script={script})."
            ).strip()
            repository.replace_chunks(document.document_id, [])
            repository.upsert_document(document)
            skipped_language += 1
            continue

        if not text_available:
            reason = (
                "отсканированный документ без текстового слоя"
                if (document.local_file or "").lower().endswith(".pdf")
                else "пустой или нечитаемый текстовый слой"
            )
            logger.warning("  no text layer — indexing metadata card only (%s)", reason)
            document.notes = (
                document.notes + f" Текст не извлечён ({reason}); "
                "проиндексирована только карточка документа."
            ).strip()
            card = [build_metadata_only_chunk(document, reason)]
            document.n_chunks = len(card)
            repository.replace_chunks(document.document_id, card)
            repository.upsert_document(document)
            all_chunks.extend(card)
            scanned += 1
            continue

        chunks = build_chunks(document, parsed, config)
        # Set the count on the record before upserting: replace_chunks() writes
        # it too, and a stale in-memory value would overwrite it back to zero.
        document.n_chunks = len(chunks)
        repository.replace_chunks(document.document_id, chunks)
        repository.upsert_document(document)
        all_chunks.extend(chunks)

    if failures:
        logger.warning("Parsing failed for %d documents", len(failures))
    logger.info(
        "Parse stage: %d chunks, %d documents excluded by language, "
        "%d documents without a text layer, %d parse failures",
        len(all_chunks),
        skipped_language,
        scanned,
        len(failures),
    )
    _write_chunks_jsonl(all_chunks)
    return all_chunks


def _write_chunks_jsonl(chunks: list[Chunk]) -> None:
    CHUNKS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_JSONL, "w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_payload(), ensure_ascii=False) + "\n")
    logger.info("Wrote %d chunks -> %s", len(chunks), CHUNKS_JSONL)


def stage_index(chunks: list[Chunk]) -> dict[str, int]:
    if not chunks:
        logger.warning("No chunks to index")
        return {"chunks": 0, "vectors": 0, "bm25": 0}
    return build_indexes(chunks)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Полный конвейер загрузки нормативной базы ЕАЭС"
    )
    parser.add_argument("--rediscover", action="store_true", help="перезапустить обход источников")
    parser.add_argument("--skip-discover", action="store_true", help="не обходить источники")
    parser.add_argument("--skip-download", action="store_true", help="не скачивать файлы")
    parser.add_argument("--only-index", action="store_true", help="только парсинг+чанкинг+индекс")
    parser.add_argument("--ocr", action="store_true", help="разрешить OCR для сканов")
    parser.add_argument("--limit", type=int, default=None, help="ограничить число документов")
    args = parser.parse_args(argv)

    ensure_directories()
    settings = get_settings()
    repository = Repository()
    run_id = repository.start_run("full_ingestion")
    started = datetime.now()
    summary: dict[str, object] = {}

    try:
        if not (args.skip_discover or args.only_index):
            summary["discovered"] = stage_discover(force=args.rediscover)

        if not (args.skip_download or args.only_index):
            records = stage_download(args.limit)
            summary["downloaded"] = sum(
                1 for r in records if r.download_status == DownloadStatus.OK
            )
            summary["manual_required"] = sum(
                1 for r in records if r.download_status == DownloadStatus.MANUAL_REQUIRED
            )
            summary["download_failed"] = sum(
                1 for r in records if r.download_status == DownloadStatus.FAILED
            )

        summary["inn_vocabulary"] = stage_build_inn_vocabulary(repository)

        chunks = stage_parse_and_chunk(repository, args.ocr, args.limit)
        summary["chunks"] = len(chunks)

        index_stats = stage_index(chunks)
        summary.update(index_stats)

        # Refresh the CSV registry with the final chunk counts.
        write_documents_csv(repository.list_documents())

        summary["embedding_provider"] = settings.embedding_provider
        summary["embedding_model"] = (
            settings.embedding_model
            if settings.embedding_provider == "local"
            else settings.openai_embedding_model
        )
        summary["duration_seconds"] = round(
            (datetime.now() - started).total_seconds(), 1
        )
        repository.finish_run(run_id, "ok", summary)
    except Exception as exc:  # noqa: BLE001
        repository.finish_run(run_id, "failed", {"error": str(exc)})
        logger.exception("Ingestion failed")
        return 1

    print("\n=== INGESTION SUMMARY ===")
    for key, value in summary.items():
        print(f"  {key:<22} {value}")
    print(
        f"\n  Документов в базе: {repository.count_documents()}"
        f"\n  Чанков в базе:     {repository.count_chunks()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
