"""Explainable confidence scoring and conflict detection.

Confidence is **computed in code from observable facts**, never asked of the
LLM: a model's self-reported certainty is not evidence.  The factors below are
each reported back to the user so the number can be argued with.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

from app.db.models import DocumentStatus, SourceAuthority, SourceTier, VersionStatus
from app.rag.citations import Citation, CitationRegistry
from app.regulatory.schemas import (
    ConfidenceLevel,
    ConfidenceReport,
    SourceConflict,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ConfidenceWeights:
    """Contribution of each factor to the 0..1 confidence score."""

    has_any_source: float = 0.20
    has_binding_source: float = 0.25
    multiple_binding_sources: float = 0.10
    statements_are_cited: float = 0.20
    retrieval_strength: float = 0.10
    version_confirmed: float = 0.10
    inn_specific_source: float = 0.05

    #: Penalties (subtracted).
    conflict_penalty: float = 0.15
    superseded_penalty: float = 0.15
    only_supplementary_penalty: float = 0.25
    input_insufficient_penalty: float = 0.20


DEFAULT_WEIGHTS = ConfidenceWeights()


def compute_confidence(
    registry: CitationRegistry,
    used_citation_ids: set[int],
    total_statements: int,
    cited_statements: int,
    conflicts: list[SourceConflict],
    input_sufficient: bool,
    weights: ConfidenceWeights = DEFAULT_WEIGHTS,
) -> ConfidenceReport:
    """Score the answer and explain every contribution."""
    score = 0.0
    factors: list[str] = []

    used = registry.used(used_citation_ids) or registry.citations
    if not used:
        return ConfidenceReport(
            level=ConfidenceLevel.LOW,
            score=0.0,
            factors=["Нормативные источники не найдены — вывод не обоснован."],
        )
    if not used_citation_ids:
        # Fragments were retrieved but the answer cites none of them, so nothing
        # in it is actually supported: retrieval quality must not raise the score.
        return ConfidenceReport(
            level=ConfidenceLevel.LOW,
            score=0.0,
            factors=[
                f"Поиск вернул {len(used)} фрагментов, но ни один из них не "
                "подтверждает утверждения ответа — обоснование отсутствует.",
            ],
        )

    score += weights.has_any_source
    factors.append(f"Найдено источников: {len(used)} (+{weights.has_any_source:.2f}).")

    binding = [c for c in used if c.is_binding_eaeu]
    # An Expert Committee recommendation (TIER 3) is an EAEU act about one
    # specific INN — for a question about that INN it is the document a
    # reviewer applies.  It is not "binding" in the TIER 1–2 sense, but it is
    # certainly not the supplementary international material the penalty below
    # is aimed at, so it stops that penalty from firing.
    eaeu_expert = [
        c
        for c in used
        if c.tier == SourceTier.TIER_3
        and c.chunk.source_authority == SourceAuthority.EAEU
    ]
    if binding:
        score += weights.has_binding_source
        factors.append(
            f"Есть обязательные акты ЕАЭС (TIER 1–2): {len(binding)} "
            f"(+{weights.has_binding_source:.2f})."
        )
        if len({c.document_id for c in binding}) > 1:
            score += weights.multiple_binding_sources
            factors.append(
                "Вывод подтверждён более чем одним обязательным актом "
                f"(+{weights.multiple_binding_sources:.2f})."
            )
    elif eaeu_expert:
        score += weights.has_binding_source
        factors.append(
            f"Вывод опирается на рекомендации Экспертного комитета ЕЭК: "
            f"{len(eaeu_expert)} — акт ЕАЭС по конкретному МНН "
            f"(+{weights.has_binding_source:.2f})."
        )
    else:
        score -= weights.only_supplementary_penalty
        factors.append(
            "Актов ЕАЭС среди источников нет — вывод опирается только "
            f"на справочные/международные материалы (−{weights.only_supplementary_penalty:.2f})."
        )

    if total_statements:
        share = cited_statements / total_statements
        contribution = weights.statements_are_cited * share
        score += contribution
        factors.append(
            f"Доля утверждений со ссылками: {share:.0%} (+{contribution:.2f})."
        )

    best_score = max((c.retrieval_score for c in used), default=0.0)
    # RRF scores live around 0.01–0.06; tanh keeps the factor bounded.
    strength = min(1.0, best_score * 25)
    score += weights.retrieval_strength * strength
    factors.append(
        f"Релевантность лучшего фрагмента (нормированная): {strength:.2f} "
        f"(+{weights.retrieval_strength * strength:.2f})."
    )

    confirmed = [
        c
        for c in used
        if c.chunk.version_status != VersionStatus.REQUIRES_EXPERT_VALIDATION
    ]
    if confirmed:
        contribution = weights.version_confirmed * (len(confirmed) / len(used))
        score += contribution
        factors.append(
            f"Редакция подтверждена для {len(confirmed)} из {len(used)} источников "
            f"(+{contribution:.2f})."
        )
    else:
        factors.append(
            "Ни для одного источника действующая редакция не подтверждена "
            "автоматически — требуется экспертная проверка."
        )

    if any(c.tier == SourceTier.TIER_3 for c in used):
        score += weights.inn_specific_source
        factors.append(
            "Найдена рекомендация Экспертного комитета по конкретному МНН "
            f"(+{weights.inn_specific_source:.2f})."
        )

    superseded = [c for c in used if c.chunk.status == DocumentStatus.SUPERSEDED]
    if superseded:
        score -= weights.superseded_penalty
        factors.append(
            f"Среди источников есть утратившие силу документы: {len(superseded)} "
            f"(−{weights.superseded_penalty:.2f})."
        )

    if conflicts:
        score -= weights.conflict_penalty
        factors.append(
            f"Обнаружены потенциальные расхождения между источниками: {len(conflicts)} "
            f"(−{weights.conflict_penalty:.2f})."
        )

    if not input_sufficient:
        score -= weights.input_insufficient_penalty
        factors.append(
            "Исходных данных о препарате недостаточно для однозначного вывода "
            f"(−{weights.input_insufficient_penalty:.2f})."
        )

    score = max(0.0, min(1.0, score))
    if score >= 0.7:
        level = ConfidenceLevel.HIGH
    elif score >= 0.45:
        level = ConfidenceLevel.MEDIUM
    else:
        level = ConfidenceLevel.LOW

    factors.append(f"Итоговая оценка: {score:.2f} → {level}.")
    return ConfidenceReport(level=level, score=round(score, 3), factors=factors)


# --------------------------------------------------------------------------- #
# Conflict detection
# --------------------------------------------------------------------------- #


def detect_conflicts(registry: CitationRegistry) -> list[SourceConflict]:
    """Flag source combinations a reviewer must reconcile manually.

    The system never decides which provision prevails: recency alone does not
    repeal an act, and a general rule does not automatically override a special
    one.  It only surfaces the combinations that require that judgement.
    """
    conflicts: list[SourceConflict] = []
    citations = registry.citations
    if len(citations) < 2:
        return conflicts

    # 1. An act present alongside a document that amends/supersedes it.
    superseded = [c for c in citations if c.chunk.status == DocumentStatus.SUPERSEDED]
    if superseded and any(c.chunk.status != DocumentStatus.SUPERSEDED for c in citations):
        conflicts.append(
            SourceConflict(
                description=(
                    "Среди найденных источников есть документы, помеченные как "
                    "утратившие силу, и действующие документы по тому же вопросу. "
                    "Положения утративших силу актов не могут использоваться как "
                    "действующее требование."
                ),
                citation_ids=[c.citation_id for c in superseded],
                resolution_note=(
                    "Проверьте по официальному источнику, каким актом документ "
                    "признан утратившим силу и что применяется вместо него."
                ),
            )
        )

    # 2. Same act present in several revisions (base act + amending act).
    by_number: dict[str, list[Citation]] = defaultdict(list)
    for citation in citations:
        if citation.chunk.document_number:
            by_number[citation.chunk.document_number].append(citation)
    for number, group in by_number.items():
        documents = {c.document_id for c in group}
        if len(documents) > 1:
            conflicts.append(
                SourceConflict(
                    description=(
                        f"Найдено несколько документов с номером № {number} "
                        "(вероятно, основной акт и акт о внесении изменений). "
                        "Приведённые формулировки могут относиться к разным редакциям."
                    ),
                    citation_ids=[c.citation_id for c in group],
                    resolution_note=(
                        "Система не формирует консолидированную редакцию. "
                        "Действующая формулировка должна быть проверена экспертом "
                        "по официальному источнику."
                    ),
                )
            )

    # 3. Binding EAEU act vs supplementary international guidance.
    binding = [c for c in citations if c.is_binding_eaeu]
    supplementary = [
        c for c in citations if c.chunk.source_authority != SourceAuthority.EAEU
    ]
    if binding and supplementary:
        conflicts.append(
            SourceConflict(
                description=(
                    "В контексте присутствуют и обязательные акты ЕАЭС, и "
                    "международные руководства (ICH/EMA/WHO). Международные "
                    "документы носят справочный характер и не заменяют требования "
                    "ЕАЭС."
                ),
                citation_ids=[c.citation_id for c in supplementary],
                resolution_note=(
                    "Обязательными являются требования актов ЕАЭС; международные "
                    "источники используйте только как научный/методический контекст."
                ),
            )
        )

    return conflicts


def unverified_version_note(registry: CitationRegistry) -> str | None:
    """Standard limitation text when no source revision could be confirmed."""
    unverified = [
        c
        for c in registry
        if c.chunk.version_status == VersionStatus.REQUIRES_EXPERT_VALIDATION
    ]
    if not unverified:
        return None
    return (
        f"Для {len(unverified)} из {len(registry)} источников действующая редакция "
        "не подтверждена автоматически (version_status = requires_expert_validation). "
        "Перед использованием вывода сверьте редакцию по официальному источнику."
    )
