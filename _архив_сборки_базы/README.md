# Архив: код сборки базы знаний

Здесь лежит конвейер, который **один раз** собрал базу знаний проекта:
скачал документы с портала ЕАЭС, извлёк текст (включая OCR 569 сканов),
разметил структуру актов, нарезал на фрагменты и построил индексы.

Результат его работы — то, что лежит в `database/` и `data/`:

| | |
|---|---|
| `database/eaeu.sqlite3` | 755 документов, 11 058 фрагментов |
| `database/qdrant/` | 11 058 векторов |
| `database/bm25_index.pkl` | лексический индекс |
| `data/raw/eaeu/` | исходные PDF |
| `data/registry/documents.csv` | реестр: URL, SHA-256, дата обращения |

**Почему вынесено.** При работе приложения этот код не исполняется: оно только
читает готовые индексы. Пересобирать базу не планируется, поэтому в рабочем
проекте оставлено лишь то, что реально работает при ответе на вопрос.

**Почему не удалено.** Без него нельзя показать, как из PDF получились индексы —
а это половина проделанной работы.

---

## Состав

```
ingestion/
├── run_ingestion.py      дирижёр: discover → download → parse → chunk → index
├── discover_sources.py   сбор списка файлов документов
├── downloader.py         скачивание, реестр, SHA-256, дата обращения
├── http_client.py        HTTP с повторами и ограничением частоты
├── parser.py             PDF / DOCX / HTML / TXT + OCR (Tesseract)
├── structure_parser.py   разметка приложение / раздел / глава / пункт
├── chunker.py            нарезка по границам структуры
├── metadata.py           тип документа, уровень источника, даты, статус
├── normalizer.py         нормализация текста
├── embeddings.py         эмбеддинги (копия рабочего app/rag/embeddings.py)
└── indexer.py            Qdrant + BM25 (копия рабочего app/rag/indexes.py)

scripts/
├── ocr_backfill.py       инкрементальное распознавание сканов
├── rebuild_index.py      полная пересборка индексов
├── update_sources.py     инкрементальное обновление базы
└── demo.py               прогон эталонных запросов без интерфейса

tests/                    46 тест-функций для кода сборки
```

---

## Как вернуть в проект

1. Скопировать `ingestion/` в корень проекта.
2. Скопировать `scripts/*.py` в `scripts/`, `tests/*.py` в `tests/`.
3. Установить зависимости сборки (они помечены в конце `requirements.txt`):

```powershell
pip install requests urllib3 beautifulsoup4 lxml PyMuPDF python-docx
pip install pytesseract pillow          # только если нужен OCR сканов
```

4. Закрыть приложение Streamlit — локальный Qdrant допускает один процесс.
5. Запустить нужное:

```powershell
python ingestion\run_ingestion.py        # полный конвейер
python scripts\rebuild_index.py          # только пересборка индексов
python scripts\ocr_backfill.py --limit 20  # распознать сканы
```

**Совместимость.** Импорты внутри архива указывают на `ingestion.*`, а
приложение теперь использует `app.rag.indexes` и `app.rag.embeddings` —
это копии тех же модулей. Настройки сборки (`chunk_*`, `http_*`, `ocr_*`,
`tesseract_cmd`) остались в `app/config.py` в отдельном блоке, поэтому
архивный код заработает без правок.
