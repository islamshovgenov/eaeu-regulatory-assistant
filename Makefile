# ---------------------------------------------------------------------------
# Convenience targets (Linux/macOS + Windows with GNU Make).
# Windows PowerShell equivalents are listed next to every target and in README.
#
# База знаний собрана заранее; код её сборки лежит в `_архив_сборки_базы/`.
# Здесь остались только цели, нужные для работы и проверки приложения.
# ---------------------------------------------------------------------------

PYTHON ?= python

.PHONY: help venv install stats check eval test app docker-qdrant clean

help:            ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

venv:            ## Create a virtual environment  (PS: python -m venv .venv)
	$(PYTHON) -m venv .venv

install:         ## Install dependencies  (PS: .\.venv\Scripts\pip install -r requirements.txt)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

app:             ## Start the Streamlit UI
	$(PYTHON) -m streamlit run app/main.py

test:            ## Run the test suite
	$(PYTHON) -m pytest tests -q

eval:            ## Retrieval evaluation on data/eval/questions.csv
	$(PYTHON) scripts/evaluate.py

check:           ## Health check of registry, files and indexes
	$(PYTHON) scripts/check_database.py

stats:           ## Corpus statistics -> data/processed/corpus_statistics.json
	$(PYTHON) scripts/corpus_stats.py

docker-qdrant:   ## Start a standalone Qdrant server
	docker compose up -d qdrant

clean:           ## Remove caches and build artefacts (keeps data/ and database/)
	$(PYTHON) -c "import shutil,pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	$(PYTHON) -c "import shutil; shutil.rmtree('.pytest_cache', ignore_errors=True)"
