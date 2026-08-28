"""Demonstration run over the reference queries and product cases.

    python scripts/demo.py                 # retrieval only (no LLM needed)
    python scripts/demo.py --with-llm      # full pipeline, requires an API key
    python scripts/demo.py --cases         # also run data/eval/cases/cases.json

Prints, for each query, the sources the system would cite and verifies that
every citation resolves to a chunk that really exists in the corpus.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import EVAL_DIR, INN_VOCABULARY_JSON, configure_logging
from app.db.repository import Repository
from app.rag.pipeline import RegulatoryPipeline
from app.regulatory.inn import load_vocabulary
from app.regulatory.schemas import (
    DECISION_STATUS_LABELS,
    ProductProfile,
    ProductType,
    ReleaseType,
)

logger = configure_logging("scripts.demo")

DEMO_QUERIES = (
    "Что требуется для регистрации воспроизведённого лекарственного препарата?",
    "Ибупрофен таблетки 400 мг, воспроизведённый препарат, немедленное высвобождение",
    "Когда возможен BCS-based biowaiver?",
    "Какие особенности исследования высоковариабельного лекарственного препарата?",
    "Как выбирается референтный препарат?",
)

CASES_JSON = EVAL_DIR / "cases" / "cases.json"


def _print_sources(result, repository: Repository) -> tuple[int, int]:
    """Print citations and verify each one exists in the corpus."""
    valid = 0
    for citation in result.registry:
        exists = repository.get_chunk(citation.chunk.chunk_id) is not None
        valid += int(exists)
        mark = "OK " if exists else "!! "
        print(
            f"    [{citation.citation_id}] {mark}{citation.tier} "
            f"{citation.header()[:110]}"
        )
        print(f"          {citation.chunk.source_url[:110]}")
    return valid, len(result.registry)


def run_queries(pipeline: RegulatoryPipeline, repository: Repository) -> tuple[int, int]:
    total_valid = total = 0
    for index, query in enumerate(DEMO_QUERIES, start=1):
        print(f"\n--- Запрос {index}: {query}")
        result = pipeline.answer_question(query)
        answer = result.chat_answer
        print(f"    источников: {len(result.registry)} · "
              f"уверенность: {result.confidence.level} ({result.confidence.score:.2f}) · "
              f"LLM: {'да' if result.llm_used else 'нет'}")
        if answer and answer.clarifying_questions:
            print("    уточняющие вопросы:")
            for question in answer.clarifying_questions[:4]:
                print(f"      - {question}")
        if answer and result.llm_used:
            preview = answer.answer_markdown.strip().replace("\n", " ")[:400]
            print(f"    ответ: {preview}…")
        valid, count = _print_sources(result, repository)
        total_valid += valid
        total += count
        if result.validation.messages():
            for message in result.validation.messages():
                print(f"    ! {message}")
    return total_valid, total


def run_cases(pipeline: RegulatoryPipeline, repository: Repository) -> None:
    if not CASES_JSON.exists():
        print(f"\nФайл кейсов не найден: {CASES_JSON}")
        return
    data = json.loads(CASES_JSON.read_text(encoding="utf-8"))
    for case in data.get("cases", []):
        print(f"\n=== {case['case_id']}: {case['title']}")
        raw = dict(case.get("profile", {}))
        profile = ProductProfile(
            inn=raw.get("inn", ""),
            dosage_form=raw.get("dosage_form", ""),
            strength=raw.get("strength", ""),
            route_of_administration=raw.get("route_of_administration", ""),
            product_type=ProductType(raw.get("product_type", "unknown")),
            release_type=ReleaseType(raw.get("release_type", "unknown")),
            additional_strengths=raw.get("additional_strengths", ""),
            highly_variable=raw.get("highly_variable"),
            free_text=raw.get("free_text", ""),
        )
        result = pipeline.assess_product(profile)
        assessment = result.assessment
        if assessment is None:
            print("    оценка не сформирована")
            continue
        print(f"    данных достаточно: {assessment.input_sufficient}")
        print(
            "    биоэквивалентность: "
            f"{DECISION_STATUS_LABELS[assessment.bioequivalence.status]}"
        )
        print(f"    биовейвер: {DECISION_STATUS_LABELS[assessment.biowaiver.status]}")
        print(
            f"    источников: {len(result.registry)} · уверенность: "
            f"{result.confidence.level} ({result.confidence.score:.2f})"
        )
        if assessment.clarifying_questions:
            print("    уточняющие вопросы:")
            for question in assessment.clarifying_questions[:6]:
                print(f"      - {question}")
        _print_sources(result, repository)
        print("    ожидаемое поведение:")
        for expectation in case.get("expected_behaviour", []):
            print(f"      · {expectation}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Демонстрационный прогон системы")
    parser.add_argument("--with-llm", action="store_true", help="использовать LLM")
    parser.add_argument("--cases", action="store_true", help="прогнать кейсы препаратов")
    args = parser.parse_args(argv)

    load_vocabulary(INN_VOCABULARY_JSON)
    repository = Repository()
    pipeline = RegulatoryPipeline(repository=repository)

    if not args.with_llm:
        pipeline._llm_checked = True
        pipeline._llm = None

    if not pipeline.knowledge_base_ready():
        print("Индексы не построены. Запустите: python ingestion/run_ingestion.py")
        return 1

    stats = pipeline.retriever.stats()
    print("=== DEMO ===")
    print(
        f"  документов: {stats['documents']} · чанков: {stats['chunks']} · "
        f"векторов: {stats['vectors']} · LLM: {pipeline.llm_status()}"
    )

    valid, total = run_queries(pipeline, repository)
    if args.cases:
        run_cases(pipeline, repository)

    print("\n=== ИТОГ ===")
    print(f"  проверено цитат: {total}, существуют в корпусе: {valid}")
    if total and valid != total:
        print("  ВНИМАНИЕ: часть цитат не разрешается — индекс рассинхронизирован")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
