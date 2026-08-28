"""End-to-end pipeline behaviour with stubbed retrieval and a stubbed LLM.

The critical property under test: **no sources -> no requirements**.  The
pipeline must never reach the model when retrieval is empty, and must never
promote an unsupported model claim to a normative requirement.
"""

from __future__ import annotations

import json

import pytest

from app.config import Settings
from app.db.repository import Repository
from app.rag.generator import LLMClient, LLMResponse
from app.rag.hybrid_search import RetrievedChunk
from app.rag.pipeline import RegulatoryPipeline
from app.rag.retriever import RetrievalResult
from app.regulatory.classifier import QueryIntent, classify_query
from app.regulatory.intake import extract_profile
from app.regulatory.schemas import (
    DecisionStatus,
    ProductProfile,
    ProductType,
    ReleaseType,
    StatementKind,
)
from tests.conftest import make_chunk


class StubRetriever:
    """Returns a fixed set of chunks (or nothing)."""

    def __init__(self, chunks: list | None = None) -> None:
        self._chunks = chunks or []
        self.calls: list[list[str]] = []

    def retrieve(self, queries: list[str], final_top_k: int | None = None) -> RetrievalResult:
        self.calls.append(queries)
        items = [
            RetrievedChunk(chunk=chunk, score=0.05 - 0.001 * index, vector_rank=index + 1)
            for index, chunk in enumerate(self._chunks)
        ]
        return RetrievalResult(items=items, queries=queries, used_reranker="stub")

    def is_ready(self) -> bool:
        return True

    def stats(self) -> dict[str, int]:
        return {"vectors": len(self._chunks)}


class StubLLM(LLMClient):
    """Returns a canned JSON payload and records the prompts it received."""

    provider = "stub"

    def __init__(self, payload: dict) -> None:
        super().__init__("stub-model", Settings())
        self.payload = payload
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> LLMResponse:
        self.prompts.append((system, user))
        return LLMResponse(
            text=json.dumps(self.payload, ensure_ascii=False),
            model=self.model,
            provider=self.provider,
        )


@pytest.fixture
def repository(tmp_path) -> Repository:
    return Repository(tmp_path / "pipeline.sqlite3")


def _pipeline(repository: Repository, chunks=None, payload=None) -> RegulatoryPipeline:
    llm = StubLLM(payload) if payload is not None else None
    pipeline = RegulatoryPipeline(
        retriever=StubRetriever(chunks), repository=repository, llm=llm
    )
    if payload is None:
        pipeline._llm_checked = True
        pipeline._llm = None
    return pipeline


# --------------------------------------------------------------------------- #
# No sources -> no invented requirements
# --------------------------------------------------------------------------- #


def test_empty_retrieval_never_calls_the_model(repository: Repository):
    llm = StubLLM({"answer_markdown": "Требуется исследование биоэквивалентности."})
    pipeline = RegulatoryPipeline(
        retriever=StubRetriever([]), repository=repository, llm=llm
    )
    result = pipeline.answer_question("Какие требования к регистрации в Антарктиде?")

    assert llm.prompts == [], "модель вызвана при пустом контексте"
    assert result.chat_answer is not None
    assert "Достаточное нормативное основание" in result.chat_answer.answer_markdown
    assert result.chat_answer.sufficient_basis is False
    assert result.confidence.score == 0.0


def test_empty_retrieval_assessment_declares_no_basis(repository: Repository):
    pipeline = _pipeline(repository, chunks=[], payload={"input_sufficient": True})
    profile = ProductProfile(
        inn="ибупрофен",
        dosage_form="таблетки",
        strength="400 мг",
        route_of_administration="пероральный",
        product_type=ProductType.GENERIC,
        release_type=ReleaseType.IMMEDIATE,
    )
    result = pipeline.assess_product(profile)

    assert result.assessment is not None
    assert result.assessment.bioequivalence.status == DecisionStatus.NO_LEGAL_BASIS_FOUND
    assert result.assessment.biowaiver.status == DecisionStatus.NO_LEGAL_BASIS_FOUND


def test_model_claim_without_citation_is_not_a_requirement(repository: Repository):
    chunks = [make_chunk("c1", "42. Биовейвер допускается при выполнении условий.")]
    payload = {
        "answer_markdown": "Требуется исследование биоэквивалентности [1].",
        "statements": [
            {
                "kind": "normative_requirement",
                "text": "Регулятор обязательно потребует три исследования.",
                "citation_ids": [],
            }
        ],
        "sufficient_basis": True,
    }
    pipeline = _pipeline(repository, chunks, payload)
    result = pipeline.answer_question("Когда возможен биовейвер?")

    assert result.chat_answer is not None
    statement = result.chat_answer.statements[0]
    assert statement.kind == StatementKind.REGULATORY_INTERPRETATION


def test_model_reference_to_missing_source_is_stripped(repository: Repository):
    chunks = [make_chunk("c1", "42. Биовейвер допускается.")]
    payload = {
        "answer_markdown": "Согласно [1] и [9] биовейвер возможен.",
        "statements": [],
        "sufficient_basis": True,
    }
    pipeline = _pipeline(repository, chunks, payload)
    result = pipeline.answer_question("биовейвер?")

    assert "[9]" not in result.chat_answer.answer_markdown
    assert 9 in result.validation.invalid_citation_ids


# --------------------------------------------------------------------------- #
# Insufficient product data
# --------------------------------------------------------------------------- #


def test_bare_inn_triggers_clarifying_questions(repository: Repository):
    chunks = [make_chunk("c1", "10. Исследование проводится по общему плану.")]
    payload = {
        "answer_markdown": "Ответ.",
        "statements": [],
        "sufficient_basis": True,
        "clarifying_questions": [],
    }
    pipeline = _pipeline(repository, chunks, payload)
    result = pipeline.answer_question("метформин")

    assert result.chat_answer is not None
    questions = " ".join(result.chat_answer.clarifying_questions).lower()
    assert "лекарственную форму" in questions
    assert "тип препарата" in questions


def test_assessment_with_missing_fields_is_marked_insufficient(repository: Repository):
    chunks = [make_chunk("c1", "10. Общие требования.")]
    payload = {"input_sufficient": True, "product_classification": "generic"}
    pipeline = _pipeline(repository, chunks, payload)
    result = pipeline.assess_product(ProductProfile(inn="метформин"))

    assert result.assessment is not None
    assert result.assessment.input_sufficient is False
    assert result.assessment.clarifying_questions


# --------------------------------------------------------------------------- #
# Retrieval-only mode
# --------------------------------------------------------------------------- #


def test_pipeline_works_without_llm(repository: Repository):
    chunks = [make_chunk("c1", "42. Биовейвер допускается.")]
    pipeline = _pipeline(repository, chunks, payload=None)
    result = pipeline.answer_question("биовейвер")

    assert result.llm_used is False
    assert result.has_sources
    assert "без генерации" in result.chat_answer.answer_markdown.lower()


def test_llm_failure_degrades_to_retrieval_only(repository: Repository):
    class BrokenLLM(StubLLM):
        def complete(self, system: str, user: str) -> LLMResponse:
            return LLMResponse(text="не json", model="x", provider="stub")

    chunks = [make_chunk("c1", "42. Биовейвер допускается.")]
    pipeline = RegulatoryPipeline(
        retriever=StubRetriever(chunks), repository=repository, llm=BrokenLLM({})
    )
    result = pipeline.answer_question("биовейвер")

    assert result.chat_answer is not None
    assert result.has_sources
    assert result.notes


# --------------------------------------------------------------------------- #
# Out of scope
# --------------------------------------------------------------------------- #


def test_medical_advice_is_refused(repository: Repository):
    pipeline = _pipeline(repository, [make_chunk("c", "текст")], {"answer_markdown": "x"})
    result = pipeline.answer_question("Какую дозу мне принимать при головной боли?")

    assert result.intent == QueryIntent.OUT_OF_SCOPE
    assert "не даёт медицинских рекомендаций" in result.chat_answer.answer_markdown


# --------------------------------------------------------------------------- #
# Classification / intake
# --------------------------------------------------------------------------- #


def test_classify_general_question():
    classification = classify_query(
        "Что требуется для регистрации воспроизведённого лекарственного препарата?"
    )
    assert classification.intent in (
        QueryIntent.GENERAL_REGULATORY,
        QueryIntent.PRODUCT_ASSESSMENT,
    )


def test_intake_parses_labelled_block():
    profile = extract_profile(
        "МНН: ибупрофен\n"
        "Лекарственная форма: таблетки\n"
        "Дозировка: 400 мг\n"
        "Путь введения: пероральный\n"
        "Тип препарата: воспроизведённый\n"
        "Высвобождение: немедленное"
    )
    assert profile.inn == "ибупрофен"
    assert profile.product_type == ProductType.GENERIC
    assert profile.release_type == ReleaseType.IMMEDIATE
    assert profile.is_sufficient()


def test_intake_parses_free_text():
    profile = extract_profile(
        "Ибупрофен таблетки 400 мг, воспроизведённый препарат, немедленное высвобождение, "
        "приём внутрь"
    )
    assert profile.dosage_form == "таблетки"
    assert profile.strength == "400 мг"
    assert profile.product_type == ProductType.GENERIC


def test_intake_does_not_invent_product_type():
    profile = extract_profile("Нужны ли исследования для препарата X?")
    assert profile.product_type == ProductType.UNKNOWN
    assert profile.release_type == ReleaseType.UNKNOWN
