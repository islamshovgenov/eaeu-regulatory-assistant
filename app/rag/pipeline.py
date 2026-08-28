"""The RAG pipeline: the single entry point used by the UI and by scripts.

    intake/classification -> query building -> hybrid retrieval -> fusion
    -> reranking -> citation registry -> LLM -> validation -> confidence

Every guarantee the project makes is enforced here, not in the prompt alone:
citations are built before generation, the model's output is validated against
them, confidence is computed from evidence, and an empty retrieval short-circuits
to an explicit "no legal basis found" answer without calling the LLM at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings, get_settings
from app.db.repository import Repository
from app.rag.citations import CitationRegistry, ValidationReport
from app.rag.generator import LLMClient, LLMError, LLMNotConfigured, get_llm_client
from app.rag.prompts import (
    SYSTEM_PROMPT,
    build_assessment_prompt,
    build_chat_prompt,
)
from app.rag.retriever import (
    CardDocument,
    RegulatoryRetriever,
    RetrievalResult,
    build_chat_queries,
    build_profile_queries,
)
from app.regulatory.classifier import QueryIntent, classify_product, classify_query
from app.regulatory.intake import clarifying_questions, extract_profile
from app.regulatory.reasoning import (
    compute_confidence,
    detect_conflicts,
    unverified_version_note,
)
from app.regulatory.schemas import (
    ChatAnswer,
    ConfidenceLevel,
    ConfidenceReport,
    Decision,
    DecisionStatus,
    ProductProfile,
    RegulatoryAssessment,
    Statement,
    StatementKind,
)
from app.regulatory.validators import validate_assessment, validate_chat_answer

logger = logging.getLogger(__name__)

NO_BASIS_MESSAGE = (
    "**Достаточное нормативное основание в доступной базе знаний не найдено.**\n\n"
    "В проиндексированной нормативной базе не найдено фрагментов, которые "
    "позволяют сформулировать обоснованный ответ на этот запрос.\n\n"
    "Возможные дальнейшие шаги:\n"
    "- переформулируйте вопрос, используя терминологию нормативных актов ЕАЭС;\n"
    "- расширьте запрос (например, укажите тип препарата и лекарственную форму);\n"
    "- проверьте, загружены ли соответствующие документы "
    "(страница «База знаний»);\n"
    "- обратитесь к эксперту по регуляторным вопросам."
)

#: Chat mode asks the anketa questions only for a genuinely product-specific
#: request — a general question about a category of products must get an answer,
#: not a form.  Three is the most a reader will act on.
_MAX_CLARIFYING_QUESTIONS = 3


def _needs_clarification(profile: ProductProfile) -> bool:
    """True when the question is about *this* product, not a product category."""
    if profile.inn:
        return True
    return bool(profile.dosage_form and profile.strength)


def _cards_markdown(cards: list[CardDocument]) -> str:
    """Point at relevant documents whose text is a scan without a text layer."""
    if not cards:
        return ""
    lines = [
        "",
        "**Документы по теме запроса, текст которых не проиндексирован**",
        "",
        "Эти документы найдены в реестре, но опубликованы как сканы без "
        "текстового слоя, поэтому их положения не могут быть процитированы "
        "системой. Их следует открыть вручную:",
        "",
    ]
    for card in cards:
        link = card.url or card.page_url
        lines.append(f"- [{card.title}]({link})" if link else f"- {card.title}")
    return "\n".join(lines)


@dataclass(slots=True)
class PipelineResult:
    """Everything produced by one run, ready for rendering."""

    registry: CitationRegistry
    retrieval: RetrievalResult
    confidence: ConfidenceReport
    validation: ValidationReport
    assessment: RegulatoryAssessment | None = None
    chat_answer: ChatAnswer | None = None
    intent: QueryIntent = QueryIntent.GENERAL_REGULATORY
    profile: ProductProfile | None = None
    llm_used: bool = False
    llm_model: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def has_sources(self) -> bool:
        return len(self.registry) > 0


class RegulatoryPipeline:
    """Orchestrates retrieval, generation and validation."""

    def __init__(
        self,
        settings: Settings | None = None,
        retriever: RegulatoryRetriever | None = None,
        repository: Repository | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.repository = repository or Repository()
        self.retriever = retriever or RegulatoryRetriever(settings=self.settings)
        self._llm = llm
        self._llm_checked = llm is not None
        self._act_numbers: set[str] | None = None

    # -- LLM ----------------------------------------------------------------
    @property
    def llm(self) -> LLMClient | None:
        if not self._llm_checked:
            self._llm_checked = True
            try:
                self._llm = get_llm_client(self.settings)
            except LLMNotConfigured as exc:
                logger.warning("LLM disabled: %s", exc)
                self._llm = None
        return self._llm

    def _corpus_act_numbers(self) -> set[str]:
        """Act numbers present in the corpus, cached for the pipeline's lifetime."""
        if self._act_numbers is None:
            self._act_numbers = self.repository.document_numbers()
        return self._act_numbers

    def llm_status(self) -> str:
        client = self.llm
        if client is None:
            return "не настроен (режим только поиска)"
        return f"{client.provider}: {client.model}"

    # ------------------------------------------------------------------ chat
    def answer_question(
        self, question: str, history: list[tuple[str, str]] | None = None
    ) -> PipelineResult:
        """Free-form chat mode."""
        classification = classify_query(question)
        profile = classification.profile

        if classification.intent == QueryIntent.OUT_OF_SCOPE:
            return self._out_of_scope_result(classification.reasons, profile)

        # A bare INN is not enough to give a categorical study programme — but a
        # general question about a category of products is not an intake form
        # either, so the anketa is only raised for a concrete product.
        if (
            classification.product_specific
            and _needs_clarification(profile)
            and not profile.is_sufficient()
        ):
            questions = clarifying_questions(profile)[:_MAX_CLARIFYING_QUESTIONS]
        else:
            questions = []

        queries = build_chat_queries(question, profile)
        retrieval = self.retriever.retrieve(queries)
        registry = CitationRegistry.from_retrieval(retrieval.items)

        if registry.valid_ids() == set():
            return self._no_basis_result(retrieval, registry, profile, questions)

        conflicts = detect_conflicts(registry)
        client = self.llm
        if client is None:
            return self._retrieval_only_result(retrieval, registry, profile, questions)

        prompt = build_chat_prompt(question, registry.context_block(), history)
        try:
            raw = client.complete_json(SYSTEM_PROMPT, prompt)
            answer = ChatAnswer.model_validate(_coerce_chat(raw))
        except (LLMError, ValueError) as exc:
            logger.error("Chat generation failed: %s", exc)
            result = self._retrieval_only_result(retrieval, registry, profile, questions)
            result.notes.append(f"Генерация ответа недоступна: {exc}")
            return result

        answer, report = validate_chat_answer(
            answer, registry, self._corpus_act_numbers()
        )
        answer.clarifying_questions = list(
            dict.fromkeys([*questions, *answer.clarifying_questions])
        )[:_MAX_CLARIFYING_QUESTIONS]
        # Only for a product-specific question: for a general one these cards
        # are noise, but for an INN they are exactly the Expert Committee
        # recommendation the reviewer needs to read.
        if classification.product_specific and retrieval.card_only_documents:
            answer.answer_markdown += "\n" + _cards_markdown(
                retrieval.card_only_documents
            )
        if note := unverified_version_note(registry):
            answer.limitations.append(note)
        for conflict in conflicts:
            answer.limitations.append(
                f"Возможное расхождение источников: {conflict.description}"
            )

        cited = sum(1 for s in answer.statements if s.citation_ids)
        answer.confidence = compute_confidence(
            registry,
            answer.all_citation_ids(),
            total_statements=len(answer.statements),
            cited_statements=cited,
            conflicts=conflicts,
            input_sufficient=not questions,
        )

        return PipelineResult(
            registry=registry,
            retrieval=retrieval,
            confidence=answer.confidence,
            validation=report,
            chat_answer=answer,
            intent=classification.intent,
            profile=profile,
            llm_used=True,
            llm_model=f"{client.provider}/{client.model}",
        )

    # ------------------------------------------------------------ assessment
    def assess_product(self, profile: ProductProfile) -> PipelineResult:
        """Structured regulatory assessment mode."""
        if profile.free_text:
            profile = extract_profile(profile.free_text, profile)

        questions = clarifying_questions(profile)
        queries = build_profile_queries(profile)
        retrieval = self.retriever.retrieve(queries)
        registry = CitationRegistry.from_retrieval(retrieval.items)

        if not registry.valid_ids():
            result = self._no_basis_result(retrieval, registry, profile, questions)
            result.assessment = _empty_assessment(profile, questions)
            result.assessment.confidence = result.confidence
            return result

        conflicts = detect_conflicts(registry)
        client = self.llm
        if client is None:
            result = self._retrieval_only_result(retrieval, registry, profile, questions)
            result.assessment = _empty_assessment(profile, questions)
            result.assessment.limitations.insert(
                0,
                "LLM-провайдер не настроен: показаны только найденные нормативные "
                "фрагменты без сгенерированного анализа.",
            )
            result.assessment.confidence = result.confidence
            return result

        inn_note = ""
        if retrieval.inn_specific_documents:
            inn_note = (
                "\nВ контексте присутствуют рекомендации Экспертного комитета ЕЭК, "
                "относящиеся к конкретному МНН: "
                + "; ".join(retrieval.inn_specific_documents[:5])
                + ". Учитывай их приоритет для этого МНН."
            )

        prompt = build_assessment_prompt(
            profile, registry.context_block(), questions, inn_note
        )
        try:
            raw = client.complete_json(SYSTEM_PROMPT, prompt)
            assessment = RegulatoryAssessment.model_validate(_coerce_assessment(raw))
        except (LLMError, ValueError) as exc:
            logger.error("Assessment generation failed: %s", exc)
            result = self._retrieval_only_result(retrieval, registry, profile, questions)
            result.assessment = _empty_assessment(profile, questions)
            result.assessment.limitations.insert(0, f"Генерация недоступна: {exc}")
            result.assessment.confidence = result.confidence
            result.notes.append(str(exc))
            return result

        assessment, report = validate_assessment(
            assessment, registry, self._corpus_act_numbers()
        )

        if questions:
            assessment.input_sufficient = False
            assessment.clarifying_questions = list(
                dict.fromkeys([*questions, *assessment.clarifying_questions])
            )
        if not assessment.product_classification:
            assessment.product_classification = classify_product(profile)
        for conflict in conflicts:
            if conflict.description not in {c.description for c in assessment.conflicts}:
                assessment.conflicts.append(conflict)
        if note := unverified_version_note(registry):
            assessment.limitations.append(note)
        if cards := retrieval.card_only_documents:
            assessment.limitations.append(
                "Найдены документы по теме запроса, опубликованные как сканы без "
                "текстового слоя; их положения не учтены в анализе: "
                + "; ".join(card.title for card in cards)
            )
        assessment.limitations.append(
            "Система носит вспомогательный характер. Итоговое решение о программе "
            "исследований принимает уполномоченный регуляторный эксперт."
        )

        statements = [
            *assessment.registration_strategy,
            *assessment.regulatory_risks,
            *assessment.potential_authority_questions,
        ]
        cited = sum(1 for s in statements if s.citation_ids)
        cited += sum(1 for s in assessment.required_studies if s.citation_ids)
        assessment.confidence = compute_confidence(
            registry,
            assessment.all_citation_ids(),
            total_statements=len(statements) + len(assessment.required_studies),
            cited_statements=cited,
            conflicts=assessment.conflicts,
            input_sufficient=assessment.input_sufficient,
        )

        return PipelineResult(
            registry=registry,
            retrieval=retrieval,
            confidence=assessment.confidence,
            validation=report,
            assessment=assessment,
            intent=QueryIntent.PRODUCT_ASSESSMENT,
            profile=profile,
            llm_used=True,
            llm_model=f"{client.provider}/{client.model}",
        )

    # ------------------------------------------------------------- fallbacks
    def _no_basis_result(
        self,
        retrieval: RetrievalResult,
        registry: CitationRegistry,
        profile: ProductProfile,
        questions: list[str],
    ) -> PipelineResult:
        answer = ChatAnswer(
            answer_markdown=NO_BASIS_MESSAGE
            + _cards_markdown(retrieval.card_only_documents),
            sufficient_basis=False,
            clarifying_questions=questions,
            limitations=[
                "Поиск по проиндексированной базе не вернул релевантных фрагментов.",
            ],
            confidence=ConfidenceReport(
                level=ConfidenceLevel.LOW,
                score=0.0,
                factors=["Нормативные источники не найдены."],
            ),
        )
        return PipelineResult(
            registry=registry,
            retrieval=retrieval,
            confidence=answer.confidence,
            validation=ValidationReport(),
            chat_answer=answer,
            profile=profile,
            llm_used=False,
        )

    def _retrieval_only_result(
        self,
        retrieval: RetrievalResult,
        registry: CitationRegistry,
        profile: ProductProfile,
        questions: list[str],
    ) -> PipelineResult:
        lines = [
            "**Режим без генерации ответа.**",
            "",
            "LLM-провайдер не настроен или недоступен, поэтому показаны только "
            "найденные нормативные фрагменты. Ниже — источники, отобранные "
            "гибридным поиском; интерпретация остаётся за экспертом.",
        ]
        answer = ChatAnswer(
            answer_markdown="\n".join(lines),
            sufficient_basis=False,
            clarifying_questions=questions,
            limitations=[
                "Ответ не сгенерирован: работает только поисковая часть системы.",
            ],
        )
        answer.confidence = compute_confidence(
            registry,
            registry.valid_ids(),
            total_statements=0,
            cited_statements=0,
            conflicts=detect_conflicts(registry),
            input_sufficient=not questions,
        )
        return PipelineResult(
            registry=registry,
            retrieval=retrieval,
            confidence=answer.confidence,
            validation=ValidationReport(),
            chat_answer=answer,
            profile=profile,
            llm_used=False,
        )

    def _out_of_scope_result(
        self, reasons: list[str], profile: ProductProfile
    ) -> PipelineResult:
        answer = ChatAnswer(
            answer_markdown=(
                "Этот вопрос выходит за пределы назначения системы.\n\n"
                "EAEU Regulatory Assistant поддерживает регуляторный анализ "
                "лекарственных препаратов (требования к регистрации, исследованиям, "
                "досье) и не даёт медицинских рекомендаций пациентам."
            ),
            sufficient_basis=False,
            limitations=reasons,
            confidence=ConfidenceReport(
                level=ConfidenceLevel.LOW, score=0.0, factors=reasons
            ),
        )
        return PipelineResult(
            registry=CitationRegistry([]),
            retrieval=RetrievalResult(),
            confidence=answer.confidence,
            validation=ValidationReport(),
            chat_answer=answer,
            intent=QueryIntent.OUT_OF_SCOPE,
            profile=profile,
        )

    # -- health -------------------------------------------------------------
    def knowledge_base_ready(self) -> bool:
        return self.retriever.is_ready()


# --------------------------------------------------------------------------- #
# Tolerant coercion of model output
# --------------------------------------------------------------------------- #


def _coerce_statements(value: Any, default_kind: str) -> list[dict[str, Any]]:
    """Accept both ``["text", ...]`` and ``[{"text": ...}, ...]`` shapes."""
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, str):
            result.append({"kind": default_kind, "text": item, "citation_ids": []})
        elif isinstance(item, dict) and item.get("text"):
            item.setdefault("kind", default_kind)
            item.setdefault("citation_ids", [])
            item["citation_ids"] = [
                int(i) for i in item["citation_ids"] if isinstance(i, (int, str)) and str(i).isdigit()
            ]
            result.append(item)
    return result


def _coerce_decision(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return Decision().model_dump(mode="json")
    status = str(value.get("status", "")).lower()
    if status not in {s.value for s in DecisionStatus}:
        status = DecisionStatus.INSUFFICIENT_DATA.value
    return {
        "status": status,
        "rationale": str(value.get("rationale", "")),
        "conditions": [str(c) for c in value.get("conditions", []) if c],
        "citation_ids": [int(i) for i in value.get("citation_ids", []) if str(i).isdigit()],
    }


def _coerce_chat(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "answer_markdown": str(raw.get("answer_markdown", "")),
        "statements": _coerce_statements(
            raw.get("statements"), StatementKind.REGULATORY_INTERPRETATION.value
        ),
        "sufficient_basis": bool(raw.get("sufficient_basis", False)),
        "clarifying_questions": [str(q) for q in raw.get("clarifying_questions", []) if q],
        "limitations": [str(x) for x in raw.get("limitations", []) if x],
    }


def _coerce_assessment(raw: dict[str, Any]) -> dict[str, Any]:
    studies: list[dict[str, Any]] = []
    for item in raw.get("required_studies", []) or []:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        category = str(item.get("category", "clinical")).lower()
        if category not in ("pharmaceutical", "nonclinical", "clinical"):
            category = "clinical"
        studies.append(
            {
                "category": category,
                "name": str(item["name"]),
                "necessity": str(item.get("necessity", "возможно требуется")),
                "rationale": str(item.get("rationale", "")),
                "kind": str(
                    item.get("kind", StatementKind.REGULATORY_INTERPRETATION.value)
                ),
                "citation_ids": [
                    int(i) for i in item.get("citation_ids", []) if str(i).isdigit()
                ],
            }
        )

    conflicts: list[dict[str, Any]] = []
    for item in raw.get("conflicts", []) or []:
        if isinstance(item, dict) and item.get("description"):
            conflicts.append(
                {
                    "description": str(item["description"]),
                    "citation_ids": [
                        int(i) for i in item.get("citation_ids", []) if str(i).isdigit()
                    ],
                    "resolution_note": str(item.get("resolution_note", "")),
                }
            )

    return {
        "input_sufficient": bool(raw.get("input_sufficient", False)),
        "clarifying_questions": [
            str(q) for q in raw.get("clarifying_questions", []) or [] if q
        ],
        "product_classification": str(raw.get("product_classification", "")),
        "registration_strategy": _coerce_statements(
            raw.get("registration_strategy"),
            StatementKind.REGULATORY_INTERPRETATION.value,
        ),
        "required_studies": studies,
        "bioequivalence": _coerce_decision(raw.get("bioequivalence")),
        "biowaiver": _coerce_decision(raw.get("biowaiver")),
        "additional_strengths": _coerce_decision(raw.get("additional_strengths")),
        "regulatory_risks": _coerce_statements(
            raw.get("regulatory_risks"), StatementKind.REGULATORY_RISK.value
        ),
        "potential_authority_questions": _coerce_statements(
            raw.get("potential_authority_questions"),
            StatementKind.POTENTIAL_AUTHORITY_QUESTION.value,
        ),
        "conflicts": conflicts,
        "limitations": [str(x) for x in raw.get("limitations", []) or [] if x],
    }


def _empty_assessment(
    profile: ProductProfile, questions: list[str]
) -> RegulatoryAssessment:
    """Assessment skeleton used when nothing could be generated."""
    return RegulatoryAssessment(
        input_sufficient=not questions,
        clarifying_questions=questions,
        product_classification=classify_product(profile),
        registration_strategy=[
            Statement(
                kind=StatementKind.INSUFFICIENT_BASIS,
                text=(
                    "Регистрационная стратегия не сформирована: отсутствует "
                    "сгенерированный анализ или нормативное основание."
                ),
            )
        ],
        bioequivalence=Decision(status=DecisionStatus.NO_LEGAL_BASIS_FOUND),
        biowaiver=Decision(status=DecisionStatus.NO_LEGAL_BASIS_FOUND),
        additional_strengths=Decision(status=DecisionStatus.NO_LEGAL_BASIS_FOUND),
        limitations=[
            "Вывод не сформирован автоматически; требуется экспертная оценка.",
        ],
    )
