"""Pydantic schemas for the structured regulatory output.

The LLM is required to emit JSON matching :class:`RegulatoryAssessment`.  The
schema encodes the project's core safety distinctions:

* a *normative requirement* (``StatementKind.NORMATIVE_REQUIREMENT``) is a
  provision quoted from a retrieved EAEU act;
* a *regulatory interpretation* is our reading of it;
* a *potential authority question* / *regulatory risk* is an expectation, never
  presented as law.

Every statement carries ``citation_ids`` that must resolve to citations built
programmatically from retrieval results — see :mod:`app.rag.citations`.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

# --------------------------------------------------------------------------- #
# Product intake
# --------------------------------------------------------------------------- #


class ProductType(StrEnum):
    ORIGINAL = "original"
    GENERIC = "generic"
    HYBRID = "hybrid"
    WELL_ESTABLISHED_USE = "well_established_use"
    BIOLOGICAL = "biological"
    BIOSIMILAR = "biosimilar"
    COMBINATION = "combination"
    UNKNOWN = "unknown"


PRODUCT_TYPE_LABELS: dict[ProductType, str] = {
    ProductType.ORIGINAL: "оригинальный (референтный)",
    ProductType.GENERIC: "воспроизведённый (генерик)",
    ProductType.HYBRID: "гибридный",
    ProductType.WELL_ESTABLISHED_USE: "с хорошо изученным применением",
    ProductType.BIOLOGICAL: "биологический",
    ProductType.BIOSIMILAR: "биоаналоговый (биоподобный)",
    ProductType.COMBINATION: "комбинированный",
    ProductType.UNKNOWN: "не определён",
}


class ReleaseType(StrEnum):
    IMMEDIATE = "immediate"
    MODIFIED = "modified"
    DELAYED = "delayed"
    PROLONGED = "prolonged"
    UNKNOWN = "unknown"


RELEASE_TYPE_LABELS: dict[ReleaseType, str] = {
    ReleaseType.IMMEDIATE: "немедленное высвобождение",
    ReleaseType.MODIFIED: "модифицированное высвобождение",
    ReleaseType.DELAYED: "отсроченное (кишечнорастворимое) высвобождение",
    ReleaseType.PROLONGED: "пролонгированное высвобождение",
    ReleaseType.UNKNOWN: "не указан",
}


class ProductProfile(BaseModel):
    """Everything the user told us about the product."""

    model_config = ConfigDict(use_enum_values=False)

    inn: str = ""
    dosage_form: str = ""
    strength: str = ""
    route_of_administration: str = ""
    product_type: ProductType = ProductType.UNKNOWN
    release_type: ReleaseType = ReleaseType.UNKNOWN

    reference_product: str = ""
    combination_product: bool | None = None
    additional_strengths: str = ""
    new_indication: bool | None = None
    new_route: bool | None = None
    new_dosage_form: bool | None = None
    biological: bool | None = None
    biosimilar: bool | None = None
    narrow_therapeutic_index: bool | None = None
    highly_variable: bool | None = None
    free_text: str = ""

    #: Fields without which no study programme can be determined.
    REQUIRED_FIELDS: tuple[str, ...] = (
        "inn",
        "dosage_form",
        "strength",
        "route_of_administration",
        "product_type",
        "release_type",
    )

    def missing_fields(self) -> list[str]:
        missing: list[str] = []
        for field_name in self.REQUIRED_FIELDS:
            value = getattr(self, field_name)
            if value in ("", None) or value in (
                ProductType.UNKNOWN,
                ReleaseType.UNKNOWN,
            ):
                missing.append(field_name)
        return missing

    def is_sufficient(self) -> bool:
        return not self.missing_fields()

    def summary_lines(self) -> list[str]:
        return [
            f"МНН: {self.inn or '—'}",
            f"Лекарственная форма: {self.dosage_form or '—'}",
            f"Дозировка: {self.strength or '—'}",
            f"Путь введения: {self.route_of_administration or '—'}",
            f"Категория препарата: {PRODUCT_TYPE_LABELS[self.product_type]}",
            f"Характер высвобождения: {RELEASE_TYPE_LABELS[self.release_type]}",
            f"Референтный препарат: {self.reference_product or 'не указан'}",
        ]


# --------------------------------------------------------------------------- #
# Statements, studies, citations
# --------------------------------------------------------------------------- #


class StatementKind(StrEnum):
    NORMATIVE_REQUIREMENT = "normative_requirement"
    REGULATORY_INTERPRETATION = "regulatory_interpretation"
    REGULATORY_RISK = "regulatory_risk"
    POTENTIAL_AUTHORITY_QUESTION = "potential_authority_question"
    INSUFFICIENT_BASIS = "insufficient_basis"


STATEMENT_KIND_LABELS: dict[StatementKind, str] = {
    StatementKind.NORMATIVE_REQUIREMENT: "Нормативное требование",
    StatementKind.REGULATORY_INTERPRETATION: "Регуляторная интерпретация",
    StatementKind.REGULATORY_RISK: "Регуляторный риск",
    StatementKind.POTENTIAL_AUTHORITY_QUESTION: "Потенциальный вопрос эксперта",
    StatementKind.INSUFFICIENT_BASIS: "Недостаточно нормативного основания",
}


class Statement(BaseModel):
    """One assertion with its kind and citation ids."""

    kind: StatementKind = StatementKind.REGULATORY_INTERPRETATION
    text: str
    citation_ids: list[int] = Field(default_factory=list)

    @field_validator("text")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Statement.text must not be empty")
        return value.strip()


class StudyCategory(StrEnum):
    PHARMACEUTICAL = "pharmaceutical"
    NONCLINICAL = "nonclinical"
    CLINICAL = "clinical"


class RequiredStudy(BaseModel):
    category: StudyCategory
    name: str
    necessity: str = Field(
        default="возможно требуется",
        description="требуется / возможно требуется / не требуется / недостаточно данных",
    )
    rationale: str = ""
    kind: StatementKind = StatementKind.REGULATORY_INTERPRETATION
    citation_ids: list[int] = Field(default_factory=list)


class DecisionStatus(StrEnum):
    REQUIRED = "required"
    NOT_REQUIRED = "not_required"
    POSSIBLE = "possible"
    INSUFFICIENT_DATA = "insufficient_data"
    NO_LEGAL_BASIS_FOUND = "no_legal_basis_found"


DECISION_STATUS_LABELS: dict[DecisionStatus, str] = {
    DecisionStatus.REQUIRED: "Требуется",
    DecisionStatus.NOT_REQUIRED: "Не требуется",
    DecisionStatus.POSSIBLE: "Возможно (при выполнении условий)",
    DecisionStatus.INSUFFICIENT_DATA: "Недостаточно данных",
    DecisionStatus.NO_LEGAL_BASIS_FOUND: (
        "Достаточное нормативное основание в доступной базе знаний не найдено"
    ),
}


class Decision(BaseModel):
    """A yes/no/insufficient conclusion (bioequivalence, biowaiver, …)."""

    status: DecisionStatus = DecisionStatus.INSUFFICIENT_DATA
    rationale: str = ""
    conditions: list[str] = Field(default_factory=list)
    citation_ids: list[int] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Confidence & conflicts
# --------------------------------------------------------------------------- #


class ConfidenceLevel(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ConfidenceReport(BaseModel):
    """Explainable confidence — computed in code, never by the LLM."""

    level: ConfidenceLevel = ConfidenceLevel.LOW
    score: float = 0.0
    factors: list[str] = Field(default_factory=list)


class SourceConflict(BaseModel):
    """Potentially divergent provisions found among the retrieved sources."""

    description: str
    citation_ids: list[int] = Field(default_factory=list)
    resolution_note: str = ""


# --------------------------------------------------------------------------- #
# Top-level assessment
# --------------------------------------------------------------------------- #


class RegulatoryAssessment(BaseModel):
    """Validated structured output of a regulatory analysis."""

    model_config = ConfigDict(use_enum_values=False)

    input_sufficient: bool = False
    clarifying_questions: list[str] = Field(default_factory=list)

    product_classification: str = ""
    registration_strategy: list[Statement] = Field(default_factory=list)

    required_studies: list[RequiredStudy] = Field(default_factory=list)

    bioequivalence: Decision = Field(default_factory=Decision)
    biowaiver: Decision = Field(default_factory=Decision)
    additional_strengths: Decision = Field(default_factory=Decision)

    regulatory_risks: list[Statement] = Field(default_factory=list)
    potential_authority_questions: list[Statement] = Field(default_factory=list)

    conflicts: list[SourceConflict] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    #: Filled in by the pipeline, not by the model.
    confidence: ConfidenceReport = Field(default_factory=ConfidenceReport)

    def all_citation_ids(self) -> set[int]:
        ids: set[int] = set()
        for statement in (
            *self.registration_strategy,
            *self.regulatory_risks,
            *self.potential_authority_questions,
        ):
            ids.update(statement.citation_ids)
        for study in self.required_studies:
            ids.update(study.citation_ids)
        for decision in (self.bioequivalence, self.biowaiver, self.additional_strengths):
            ids.update(decision.citation_ids)
        for conflict in self.conflicts:
            ids.update(conflict.citation_ids)
        return ids


class ChatAnswer(BaseModel):
    """Structured output of the free-form chat mode."""

    answer_markdown: str = ""
    statements: list[Statement] = Field(default_factory=list)
    sufficient_basis: bool = False
    clarifying_questions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    confidence: ConfidenceReport = Field(default_factory=ConfidenceReport)

    def all_citation_ids(self) -> set[int]:
        ids: set[int] = set()
        for statement in self.statements:
            ids.update(statement.citation_ids)
        return ids
