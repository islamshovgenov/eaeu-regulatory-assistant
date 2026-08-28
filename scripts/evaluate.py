"""Evaluation of the retrieval and (optionally) generation stages.

    python scripts/evaluate.py                 # retrieval metrics only
    python scripts/evaluate.py --with-answers  # also runs the LLM (costs money)

Retrieval metrics
-----------------
``Recall@5`` / ``Recall@10``  — share of questions whose expected document was
retrieved in the top-k. "Expected document" is matched loosely by document
title/number keywords, because the eval set is annotated by document, not by
paragraph.
``MRR`` — mean reciprocal rank of the first expected document.

Answer metrics (``--with-answers``)
-----------------------------------
``citation_coverage``  — share of normative statements carrying ≥1 citation;
``citation_validity``  — share of cited ids that resolve to a real chunk;
``authority_distribution`` — tier mix of the cited sources;
``abstention_rate`` — share of answers that correctly declined for lack of basis.

**These numbers are not a substitute for expert review.** The eval set is
machine-generated (``needs_expert_validation = true`` for every row) and no
LLM-as-a-judge score is used as a quality gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import EVAL_DIR, INN_VOCABULARY_JSON, PROCESSED_DIR, configure_logging
from app.db.models import SourceTier
from app.rag.pipeline import RegulatoryPipeline
from app.rag.retriever import RegulatoryRetriever
from app.regulatory.inn import load_vocabulary
from app.regulatory.schemas import StatementKind

logger = configure_logging("scripts.evaluate")

QUESTIONS_CSV = EVAL_DIR / "questions.csv"
RESULTS_JSON = PROCESSED_DIR / "evaluation_results.json"

_WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z0-9]{4,}")


@dataclass(slots=True)
class EvalQuestion:
    question_id: str
    category: str
    question: str
    expected_documents: list[str]
    expected_topics: list[str]
    notes: str
    needs_expert_validation: bool


@dataclass(slots=True)
class QuestionOutcome:
    question_id: str
    category: str
    ranks: list[int] = field(default_factory=list)
    retrieved_documents: list[str] = field(default_factory=list)
    first_hit_rank: int | None = None
    tiers: Counter = field(default_factory=Counter)


def load_questions(path: Path = QUESTIONS_CSV) -> list[EvalQuestion]:
    questions: list[EvalQuestion] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            questions.append(
                EvalQuestion(
                    question_id=row["question_id"],
                    category=row["category"],
                    question=row["question"],
                    expected_documents=[
                        d.strip() for d in row["expected_documents"].split("|") if d.strip()
                    ],
                    expected_topics=[
                        t.strip() for t in row["expected_topics"].split("|") if t.strip()
                    ],
                    notes=row.get("expected_answer_notes", ""),
                    needs_expert_validation=row.get("needs_expert_validation", "true")
                    == "true",
                )
            )
    return questions


def _matches_expected(expected: str, document_title: str, document_number: str) -> bool:
    """Loose match of an expected document description against a retrieved doc.

    An expected value like "Решение Совета ЕЭК № 85" matches when the act number
    matches; a descriptive value matches on shared significant words.
    """
    number_match = re.search(r"№\s*(\d{1,4})", expected)
    if number_match:
        return number_match.group(1) == document_number

    expected_words = {w.lower() for w in _WORD_RE.findall(expected)}
    title_words = {w.lower() for w in _WORD_RE.findall(document_title)}
    if not expected_words:
        return False
    overlap = len(expected_words & title_words) / len(expected_words)
    return overlap >= 0.5


def evaluate_retrieval(
    questions: list[EvalQuestion], retriever: RegulatoryRetriever, top_k: int = 10
) -> tuple[dict[str, float], list[QuestionOutcome]]:
    outcomes: list[QuestionOutcome] = []

    for index, question in enumerate(questions, start=1):
        result = retriever.retrieve([question.question], final_top_k=top_k)
        outcome = QuestionOutcome(question.question_id, question.category)

        for rank, item in enumerate(result.items, start=1):
            chunk = item.chunk
            outcome.retrieved_documents.append(
                chunk.document_short_title or chunk.document_title[:80]
            )
            outcome.tiers[str(chunk.source_tier)] += 1
            if any(
                _matches_expected(expected, chunk.document_title, chunk.document_number)
                for expected in question.expected_documents
            ):
                outcome.ranks.append(rank)
                if outcome.first_hit_rank is None:
                    outcome.first_hit_rank = rank

        outcomes.append(outcome)
        logger.info(
            "[%d/%d] %s — первый релевантный документ на позиции %s",
            index,
            len(questions),
            question.question_id,
            outcome.first_hit_rank or "не найден",
        )

    total = len(outcomes) or 1
    recall_5 = sum(1 for o in outcomes if o.first_hit_rank and o.first_hit_rank <= 5) / total
    recall_10 = sum(1 for o in outcomes if o.first_hit_rank and o.first_hit_rank <= 10) / total
    mrr = sum(1 / o.first_hit_rank for o in outcomes if o.first_hit_rank) / total

    tier_totals: Counter = Counter()
    for outcome in outcomes:
        tier_totals.update(outcome.tiers)

    metrics = {
        "questions": len(outcomes),
        "recall_at_5": round(recall_5, 3),
        "recall_at_10": round(recall_10, 3),
        "mrr": round(mrr, 3),
        "questions_with_no_hit": sum(1 for o in outcomes if o.first_hit_rank is None),
        "source_authority_distribution": dict(tier_totals),
    }
    return metrics, outcomes


def evaluate_answers(
    questions: list[EvalQuestion], pipeline: RegulatoryPipeline, limit: int | None
) -> dict[str, object]:
    selected = questions[:limit] if limit else questions
    normative_total = 0
    normative_cited = 0
    citations_total = 0
    citations_valid = 0
    abstentions = 0
    tiers: Counter = Counter()
    failures: list[str] = []

    for index, question in enumerate(selected, start=1):
        logger.info("[%d/%d] генерация ответа: %s", index, len(selected), question.question_id)
        try:
            result = pipeline.answer_question(question.question)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{question.question_id}: {exc}")
            continue

        answer = result.chat_answer
        if answer is None:
            continue
        if not answer.sufficient_basis:
            abstentions += 1

        for statement in answer.statements:
            if statement.kind == StatementKind.NORMATIVE_REQUIREMENT:
                normative_total += 1
                if statement.citation_ids:
                    normative_cited += 1
            for citation_id in statement.citation_ids:
                citations_total += 1
                if result.registry.get(citation_id) is not None:
                    citations_valid += 1
        for citation in result.registry.used(answer.all_citation_ids()):
            tiers[str(citation.tier)] += 1

    return {
        "answers_evaluated": len(selected) - len(failures),
        "citation_coverage": (
            round(normative_cited / normative_total, 3) if normative_total else None
        ),
        "citation_validity": (
            round(citations_valid / citations_total, 3) if citations_total else None
        ),
        "abstention_rate": round(abstentions / max(len(selected), 1), 3),
        "cited_source_tiers": dict(tiers),
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Оценка качества RAG-системы")
    parser.add_argument("--with-answers", action="store_true", help="также оценить ответы LLM")
    parser.add_argument("--limit", type=int, default=None, help="ограничить число вопросов")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)

    load_vocabulary(INN_VOCABULARY_JSON)
    questions = load_questions()
    if args.limit:
        questions = questions[: args.limit]
    print(f"Загружено вопросов: {len(questions)}")

    retriever = RegulatoryRetriever()
    if not retriever.is_ready():
        print("Индексы не построены. Запустите: python _архив_сборки_базы/ingestion/run_ingestion.py")
        return 1

    metrics, outcomes = evaluate_retrieval(questions, retriever, args.top_k)

    report: dict[str, object] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "retrieval": metrics,
        "per_question": [
            {
                "question_id": o.question_id,
                "category": o.category,
                "first_hit_rank": o.first_hit_rank,
                "hits": o.ranks,
                "retrieved_documents": o.retrieved_documents[:5],
            }
            for o in outcomes
        ],
        "disclaimer": (
            "Ожидаемые документы размечены автоматически "
            "(needs_expert_validation = true). Метрики отражают поведение поиска, "
            "а не корректность регуляторного вывода. Требуется экспертная оценка."
        ),
    }

    print("\n=== RETRIEVAL METRICS ===")
    print(f"  Вопросов:              {metrics['questions']}")
    print(f"  Recall@5:              {metrics['recall_at_5']}")
    print(f"  Recall@10:             {metrics['recall_at_10']}")
    print(f"  MRR:                   {metrics['mrr']}")
    print(f"  Без релевантных:       {metrics['questions_with_no_hit']}")
    print("  Распределение источников по уровням:")
    for tier, count in sorted(metrics["source_authority_distribution"].items()):  # type: ignore[union-attr]
        print(f"    {tier:<10} {count}")

    by_category: dict[str, list[QuestionOutcome]] = {}
    for outcome in outcomes:
        by_category.setdefault(outcome.category, []).append(outcome)
    print("\n  Recall@10 по категориям:")
    for category, group in sorted(by_category.items()):
        hits = sum(1 for o in group if o.first_hit_rank and o.first_hit_rank <= 10)
        print(f"    {category:<28} {hits}/{len(group)}")

    if args.with_answers:
        pipeline = RegulatoryPipeline(retriever=retriever)
        if pipeline.llm is None:
            print("\nLLM не настроен — оценка ответов пропущена.")
        else:
            print("\nГенерация ответов (может занять время и расходует токены)…")
            answer_metrics = evaluate_answers(questions, pipeline, args.limit)
            report["answers"] = answer_metrics
            print("\n=== ANSWER METRICS ===")
            for key, value in answer_metrics.items():
                print(f"  {key:<22} {value}")

    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nОтчёт сохранён: {RESULTS_JSON}")
    print(
        "\nВАЖНО: автоматические метрики не являются подтверждением регуляторной "
        "корректности. Требуется экспертная валидация."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
