"""Post-generation validation of the model's structured output.

Everything the model produced is treated as untrusted until checked:

* citation ids must exist in the registry — unknown ids are stripped;
* a ``normative_requirement`` without a citation is *downgraded* to
  ``regulatory_interpretation`` (it cannot be presented as law);
* a normative requirement citing only TIER-4 material is downgraded too;
* act numbers mentioned in prose are checked against the corpus;
* decisions with no citations cannot claim ``required``/``not_required``.
"""

from __future__ import annotations

import logging
import re

from app.db.models import DocumentRecord, SourceTier
from app.rag.citations import (
    CitationRegistry,
    ValidationReport,
    check_act_numbers,
    validate_citation_ids,
)
from app.regulatory.schemas import (
    ChatAnswer,
    Decision,
    DecisionStatus,
    RegulatoryAssessment,
    RequiredStudy,
    Statement,
    StatementKind,
)

logger = logging.getLogger(__name__)


def _validate_statement(
    statement: Statement, registry: CitationRegistry, report: ValidationReport
) -> Statement:
    statement.citation_ids = validate_citation_ids(
        statement.citation_ids, registry, report
    )
    if statement.kind == StatementKind.NORMATIVE_REQUIREMENT:
        if not statement.citation_ids:
            report.statements_without_citation += 1
            statement.kind = StatementKind.REGULATORY_INTERPRETATION
            statement.text = (
                "[без прямой нормативной ссылки в найденном контексте] "
                + statement.text
            )
        elif not _has_binding_citation(statement.citation_ids, registry):
            statement.kind = StatementKind.REGULATORY_INTERPRETATION
            statement.text = (
                "[основано только на справочных источниках, не на обязательном "
                "акте ЕАЭС] " + statement.text
            )
    return statement


def _has_binding_citation(ids: list[int], registry: CitationRegistry) -> bool:
    for citation_id in ids:
        citation = registry.get(citation_id)
        if citation and citation.tier in (
            SourceTier.TIER_1,
            SourceTier.TIER_2,
            SourceTier.TIER_3,
        ):
            return True
    return False


def _validate_decision(
    decision: Decision, registry: CitationRegistry, report: ValidationReport
) -> Decision:
    decision.citation_ids = validate_citation_ids(
        decision.citation_ids, registry, report
    )
    if not decision.citation_ids and decision.status in (
        DecisionStatus.REQUIRED,
        DecisionStatus.NOT_REQUIRED,
    ):
        logger.info(
            "Downgrading uncited decision %s -> no_legal_basis_found", decision.status
        )
        decision.status = DecisionStatus.NO_LEGAL_BASIS_FOUND
        decision.rationale = (
            "Вывод не подтверждён найденными нормативными фрагментами, поэтому "
            "категорический ответ не формируется. " + (decision.rationale or "")
        ).strip()
    return decision


def _validate_study(
    study: RequiredStudy, registry: CitationRegistry, report: ValidationReport
) -> RequiredStudy:
    study.citation_ids = validate_citation_ids(study.citation_ids, registry, report)
    if study.kind == StatementKind.NORMATIVE_REQUIREMENT and not study.citation_ids:
        study.kind = StatementKind.REGULATORY_INTERPRETATION
        report.statements_without_citation += 1
        if study.necessity == "требуется":
            study.necessity = "возможно требуется"
    return study


def validate_assessment(
    assessment: RegulatoryAssessment,
    registry: CitationRegistry,
    documents: list[DocumentRecord] | set[str] | None = None,
) -> tuple[RegulatoryAssessment, ValidationReport]:
    """Enforce the grounding contract on a generated assessment."""
    report = ValidationReport()

    assessment.registration_strategy = [
        _validate_statement(s, registry, report) for s in assessment.registration_strategy
    ]
    assessment.regulatory_risks = [
        _validate_statement(s, registry, report) for s in assessment.regulatory_risks
    ]
    assessment.potential_authority_questions = [
        _validate_statement(s, registry, report)
        for s in assessment.potential_authority_questions
    ]
    assessment.required_studies = [
        _validate_study(s, registry, report) for s in assessment.required_studies
    ]
    assessment.bioequivalence = _validate_decision(
        assessment.bioequivalence, registry, report
    )
    assessment.biowaiver = _validate_decision(assessment.biowaiver, registry, report)
    assessment.additional_strengths = _validate_decision(
        assessment.additional_strengths, registry, report
    )
    for conflict in assessment.conflicts:
        conflict.citation_ids = validate_citation_ids(
            conflict.citation_ids, registry, report
        )

    if documents is not None:
        prose = " ".join(
            [
                assessment.product_classification,
                *(s.text for s in assessment.registration_strategy),
                *(s.text for s in assessment.regulatory_risks),
                assessment.bioequivalence.rationale,
                assessment.biowaiver.rationale,
            ]
        )
        report.unsupported_act_numbers = check_act_numbers(prose, registry, documents)

    if len(registry) == 0:
        assessment.limitations.insert(
            0,
            "Достаточное нормативное основание в доступной базе знаний не найдено — "
            "выводы носят исключительно ориентировочный характер.",
        )
    assessment.limitations.extend(report.messages())
    return assessment, report


def validate_chat_answer(
    answer: ChatAnswer,
    registry: CitationRegistry,
    documents: list[DocumentRecord] | set[str] | None = None,
) -> tuple[ChatAnswer, ValidationReport]:
    """Enforce the grounding contract on a chat answer."""
    report = ValidationReport()
    answer.statements = [
        _validate_statement(s, registry, report) for s in answer.statements
    ]

    valid_ids = registry.valid_ids()
    answer.answer_markdown = _strip_unknown_inline_refs(
        answer.answer_markdown, valid_ids, report
    )

    if documents is not None:
        report.unsupported_act_numbers = check_act_numbers(
            answer.answer_markdown, registry, documents
        )

    if len(registry) == 0:
        answer.sufficient_basis = False
        answer.limitations.insert(
            0,
            "Достаточное нормативное основание в доступной базе знаний не найдено.",
        )
    answer.limitations.extend(report.messages())
    return answer, report


#: Inline references the model writes into markdown, e.g. "[3]" or "[3, 5]".
_INLINE_REF_RE = re.compile(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]")


def _strip_unknown_inline_refs(
    text: str, valid_ids: set[int], report: ValidationReport
) -> str:
    """Remove ``[N]`` markers that do not resolve to a real citation."""

    def replace(match: re.Match[str]) -> str:
        ids = [int(part) for part in match.group(1).replace(" ", "").split(",")]
        kept = [i for i in ids if i in valid_ids]
        for invalid in (i for i in ids if i not in valid_ids):
            report.invalid_citation_ids.append(invalid)
            report.removed_citations += 1
        return f"[{', '.join(str(i) for i in kept)}]" if kept else ""

    return _INLINE_REF_RE.sub(replace, text or "")
