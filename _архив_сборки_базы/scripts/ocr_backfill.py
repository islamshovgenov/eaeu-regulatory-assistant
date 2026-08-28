"""OCR backfill for documents indexed as a metadata card only.

A large share of the Expert Committee recommendations is published as scanned
PDFs with no text layer.  Ingestion indexes those as a single registry card, so
the assistant can name the document but never quote it — which is honest, but
means the most INN-specific part of the corpus contributes nothing to answers.

This script re-parses **only** those documents with OCR and updates the indexes
in place:

    parse (OCR, parallel) -> chunk -> SQLite -> vectors (upsert) -> BM25 (rebuild)

Nothing else in the corpus is re-embedded, so a run costs minutes rather than
the hours a full ``run_ingestion.py`` would take.

Usage::

    python scripts/ocr_backfill.py --limit 20      # pilot on 20 documents
    python scripts/ocr_backfill.py                 # everything still scanned
    python scripts/ocr_backfill.py --workers 6

Requires ``pytesseract``, ``pillow`` and a Tesseract binary with the ``rus``
language data; set ``TESSERACT_CMD`` in ``.env`` if it is not on PATH.
The Streamlit app must be closed first — Qdrant's local mode allows a single
process to open the storage folder.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - direct execution
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import CHUNKS_JSONL, configure_logging, ensure_directories, get_settings
from app.db.models import Chunk, DocumentRecord
from app.db.repository import Repository
from ingestion.chunker import ChunkingConfig, build_chunks
from ingestion.downloader import write_documents_csv
from ingestion.indexer import BM25Index, VectorIndex
from ingestion.normalizer import detect_script, is_russian_text
from ingestion.parser import parse_file

logger = configure_logging("ingestion.ocr")

#: Below this, OCR is considered to have failed (a blank or unreadable scan).
MIN_USABLE_CHARS = 200


@dataclass(slots=True)
class OcrOutcome:
    """Result of OCR for one document, sent back from a worker process."""

    document_id: str
    text_blocks: list[tuple[str, int | None]]
    raw_text: str
    n_pages: int
    error: str = ""


def _ocr_document(args: tuple[str, str]) -> OcrOutcome:
    """Worker: OCR one file.  Runs in a separate process — no shared state."""
    document_id, local_file = args
    try:
        parsed = parse_file(local_file, allow_ocr=True)
    except Exception as exc:  # noqa: BLE001 - one bad scan must not stop the run
        return OcrOutcome(document_id, [], "", 0, error=f"{type(exc).__name__}: {exc}")
    return OcrOutcome(
        document_id=document_id,
        text_blocks=[(block.text, block.page) for block in parsed.blocks],
        raw_text=parsed.normalized_text,
        n_pages=parsed.n_pages,
    )


def _scanned_documents(repository: Repository) -> list[DocumentRecord]:
    """Documents whose only chunk is the "text not extracted" registry card."""
    card_ids = {
        chunk.document_id for chunk in repository.iter_chunks() if chunk.is_metadata_card
    }
    documents = [d for d in repository.list_documents() if d.document_id in card_ids]
    return [
        d for d in documents if d.local_file and Path(d.local_file).exists()
    ]


def _rebuild_parsed(outcome: OcrOutcome):
    """Rebuild a ParsedDocument-shaped object from what crossed the process boundary."""
    from ingestion.parser import ParsedDocument, TextBlock

    blocks = [TextBlock(text=text, page=page) for text, page in outcome.text_blocks]
    return ParsedDocument(
        path=Path("."),
        blocks=blocks,
        n_pages=outcome.n_pages,
        raw_text=outcome.raw_text,
        # Blocks arrive already normalised by parse_file in the worker.
        normalized_text=outcome.raw_text,
        parser_used="pymupdf+ocr",
        warnings=["Текст получен OCR — возможны ошибки распознавания."],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=None, help="сколько документов обработать")
    parser.add_argument("--workers", type=int, default=6, help="параллельных процессов OCR")
    parser.add_argument(
        "--dry-run", action="store_true", help="только показать, что будет обработано"
    )
    args = parser.parse_args(argv)

    ensure_directories()
    settings = get_settings()
    repository = Repository()

    documents = _scanned_documents(repository)
    if args.limit:
        documents = documents[: args.limit]
    if not documents:
        print("Документов без текстового слоя не найдено — дозаполнять нечего.")
        return 0

    total_pages = sum(d.n_pages or 1 for d in documents)
    print(
        f"К обработке: {len(documents)} документов (~{total_pages} страниц), "
        f"OCR: {settings.ocr_languages} @ {settings.ocr_dpi} dpi, "
        f"процессов: {args.workers}"
    )
    if args.dry_run:
        for document in documents[:20]:
            print(f"  {document.document_id}  {document.title[:90]}")
        return 0

    # Fail before spending an hour on OCR: local Qdrant allows one process, so
    # a running Streamlit app would only surface at the very end, after SQLite
    # had already been updated and the two stores had diverged.
    try:
        probe = VectorIndex(settings)
        probe.count()
        probe.close()
    except Exception as exc:  # noqa: BLE001
        print(
            "Векторное хранилище недоступно — закройте запущенное приложение "
            f"Streamlit и повторите.\nПричина: {exc}"
        )
        return 1

    config = ChunkingConfig.from_settings()
    new_chunks: list[Chunk] = []
    recovered = 0
    still_scanned = 0
    not_russian = 0
    failed = 0

    payload = [(d.document_id, d.local_file) for d in documents]
    by_id = {d.document_id: d for d in documents}

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, outcome in enumerate(pool.map(_ocr_document, payload), start=1):
            document = by_id[outcome.document_id]
            logger.info(
                "[%d/%d] %s", index, len(payload), document.document_id
            )
            if outcome.error:
                logger.error("  OCR failed: %s", outcome.error)
                failed += 1
                continue
            if len(outcome.raw_text.strip()) < MIN_USABLE_CHARS:
                logger.warning("  OCR gave no usable text — card kept")
                still_scanned += 1
                continue
            if document.language == "ru" and not is_russian_text(outcome.raw_text):
                script = detect_script(outcome.raw_text)
                logger.warning("  not Russian (script=%s) — excluded", script)
                document.language = script
                document.n_chunks = 0
                repository.replace_chunks(document.document_id, [])
                repository.upsert_document(document)
                not_russian += 1
                continue

            chunks = build_chunks(document, _rebuild_parsed(outcome), config)
            if not chunks:
                still_scanned += 1
                continue
            document.n_pages = outcome.n_pages or document.n_pages
            document.n_chunks = len(chunks)
            document.notes = (
                document.notes.replace(
                    "Текст не извлечён", "Текст восстановлен OCR; ранее не извлечён"
                )
                + " Текст получен OCR — возможны ошибки распознавания."
            ).strip()
            repository.replace_chunks(document.document_id, chunks)
            repository.upsert_document(document)
            new_chunks.extend(chunks)
            recovered += 1
            logger.info("  recovered %d chunks", len(chunks))

    print(
        f"\nOCR завершён: восстановлено {recovered}, "
        f"без текста {still_scanned}, не русский {not_russian}, ошибок {failed}"
    )
    if not new_chunks:
        print("Новых чанков нет — индексы не тронуты.")
        return 0

    # -- reindex ------------------------------------------------------------
    all_chunks = list(repository.iter_chunks())
    print(f"Пересборка индексов: {len(new_chunks)} новых из {len(all_chunks)} чанков")

    CHUNKS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_JSONL, "w", encoding="utf-8") as handle:
        for chunk in all_chunks:
            handle.write(json.dumps(chunk.to_payload(), ensure_ascii=False) + "\n")

    # BM25 is cheap and must cover the whole corpus, so it is rebuilt entirely;
    # the vector index only embeds what actually changed.
    BM25Index.build(all_chunks).save()

    vector_index = VectorIndex(settings)
    try:
        indexed = vector_index.upsert_chunks(new_chunks)
        pruned = vector_index.prune_missing({c.chunk_id for c in all_chunks})
        vectors = vector_index.count()
    finally:
        vector_index.close()

    write_documents_csv(repository.list_documents())

    print(
        f"Готово: векторов добавлено {indexed}, удалено устаревших {pruned}, "
        f"всего векторов {vectors}, чанков в базе {len(all_chunks)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
