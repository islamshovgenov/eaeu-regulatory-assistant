"""Health check of the knowledge base.

    python scripts/check_database.py

Verifies that the SQLite registry, the raw files, the BM25 index and the Qdrant
collection are mutually consistent, and reports what to run when they are not.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import BM25_INDEX_FILE, SQLITE_FILE, configure_logging, get_settings
from app.db.models import DownloadStatus
from app.db.repository import Repository
from app.rag.indexes import BM25Index

logger = configure_logging("scripts.check_database")

OK = "  [OK]  "
WARN = "  [!!]  "
FAIL = "  [XX]  "


def main() -> int:
    settings = get_settings()
    problems = 0
    warnings = 0

    print("=== ПРОВЕРКА БАЗЫ ЗНАНИЙ ===\n")

    # -- SQLite -------------------------------------------------------------
    if not SQLITE_FILE.exists():
        print(f"{FAIL}SQLite не найден: {SQLITE_FILE}")
        print("        Запустите: python _архив_сборки_базы/ingestion/run_ingestion.py")
        return 1
    repository = Repository()
    documents = repository.list_documents()
    chunk_count = repository.count_chunks()
    print(f"{OK}SQLite: {SQLITE_FILE}")
    print(f"        документов: {len(documents)}, чанков: {chunk_count}")

    if not documents:
        print(f"{FAIL}Реестр документов пуст.")
        return 1

    # -- raw files ----------------------------------------------------------
    missing_files = [
        d
        for d in documents
        if d.download_status == DownloadStatus.OK
        and (not d.local_file or not Path(d.local_file).exists())
    ]
    if missing_files:
        print(f"{FAIL}Отсутствуют локальные файлы: {len(missing_files)}")
        for document in missing_files[:5]:
            print(f"        {document.document_id}: {document.local_file}")
        problems += 1
    else:
        print(f"{OK}Все локальные файлы на месте")

    manual = [d for d in documents if d.download_status == DownloadStatus.MANUAL_REQUIRED]
    failed = [d for d in documents if d.download_status == DownloadStatus.FAILED]
    if manual:
        print(f"{WARN}Требуют ручной загрузки: {len(manual)}")
        warnings += 1
    if failed:
        print(f"{WARN}Не удалось загрузить: {len(failed)}")
        warnings += 1

    # -- documents without chunks -------------------------------------------
    unchunked = [
        d
        for d in documents
        if d.download_status == DownloadStatus.OK and d.n_chunks == 0
    ]
    if unchunked:
        print(f"{WARN}Документов без чанков: {len(unchunked)}")
        for document in unchunked[:5]:
            print(f"        {document.document_id} ({Path(document.local_file).suffix})")
        warnings += 1
    else:
        print(f"{OK}У всех загруженных документов есть чанки")

    # -- BM25 ---------------------------------------------------------------
    bm25 = BM25Index.load(BM25_INDEX_FILE)
    if bm25 is None:
        print(f"{FAIL}BM25 индекс не найден: {BM25_INDEX_FILE}")
        problems += 1
    elif len(bm25.chunk_ids) != chunk_count:
        print(
            f"{WARN}BM25 рассинхронизирован: {len(bm25.chunk_ids)} против "
            f"{chunk_count} чанков в SQLite"
        )
        warnings += 1
    else:
        print(f"{OK}BM25 индекс: {len(bm25.chunk_ids)} чанков")

    # -- Qdrant -------------------------------------------------------------
    try:
        from app.rag.indexes import VectorIndex

        index = VectorIndex(settings)
        vectors = index.count()
        index.close()
        if vectors == 0:
            print(f"{FAIL}Векторный индекс пуст (коллекция {settings.qdrant_collection})")
            problems += 1
        elif vectors != chunk_count:
            print(f"{WARN}Векторов {vectors}, чанков {chunk_count} — рассинхронизация")
            warnings += 1
        else:
            print(f"{OK}Qdrant ({settings.qdrant_mode}): {vectors} векторов")
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL}Векторный индекс недоступен: {exc}")
        problems += 1

    # -- summary ------------------------------------------------------------
    print("\n=== ИТОГ ===")
    if problems:
        print(f"  Критических проблем: {problems}, предупреждений: {warnings}")
        print("  Рекомендуется: python _архив_сборки_базы/scripts/rebuild_index.py")
        return 1
    print(f"  Проблем не обнаружено (предупреждений: {warnings})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
