"""Query classification and product-type reasoning.

Two jobs:

1. Decide *what the user is asking for* — a general regulatory question, a
   product-specific assessment, or something outside the system's scope.
2. Derive the regulatory consequences of the declared product type that are
   purely definitional (e.g. a biosimilar is a biological product), **without**
   asserting any study requirement — those must come from retrieved text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from app.regulatory.schemas import ProductProfile, ProductType, ReleaseType


class QueryIntent(StrEnum):
    GENERAL_REGULATORY = "general_regulatory"
    PRODUCT_ASSESSMENT = "product_assessment"
    DOCUMENT_LOOKUP = "document_lookup"
    OUT_OF_SCOPE = "out_of_scope"


#: Vocabulary that marks a regulatory question about medicines.
_REGULATORY_MARKERS = (
    "регистрац", "экспертиз", "досье", "биоэквивалент", "биовейвер", "biowaiver",
    "референтн", "исследован", "клиническ", "доклиническ", "gcp", "glp", "gmp",
    "фармаконадзор", "охлп", "инструкц", "стабильн", "валидац", "спецификац",
    "примес", "биоаналог", "биоподобн", "иммуноген", "лекарств", "препарат",
    "дозировк", "высвобожден", "фармакокинет", "auc", "cmax", "модуль",
    "надлежащей практики", "еаэс", "решени", "приложени", "пункт",
)

#: Requests that this tool must not answer (clinical/medical advice).
_OUT_OF_SCOPE_MARKERS = (
    "какую дозу мне принимать", "можно ли мне принимать", "как лечить",
    "поставь диагноз", "назначь лечение", "мне плохо",
)

_DOCUMENT_LOOKUP_RE = re.compile(
    r"(текст|скачать|где найти|дай ссылку|покажи документ|решение\s*№|"
    r"рекомендаци[яю]\s*№)",
    re.IGNORECASE,
)


@dataclass(slots=True)
class QueryClassification:
    intent: QueryIntent
    profile: ProductProfile
    reasons: list[str] = field(default_factory=list)
    product_specific: bool = False


def classify_query(text: str, profile: ProductProfile | None = None) -> QueryClassification:
    """Classify a free-form user message."""
    from app.regulatory.intake import extract_profile

    lowered = text.lower().replace("ё", "е")
    extracted = extract_profile(text, profile)
    reasons: list[str] = []

    if any(marker in lowered for marker in _OUT_OF_SCOPE_MARKERS):
        return QueryClassification(
            intent=QueryIntent.OUT_OF_SCOPE,
            profile=extracted,
            reasons=[
                "Запрос выглядит как просьба о медицинской консультации; система "
                "предназначена только для регуляторного анализа."
            ],
        )

    product_specific = bool(
        extracted.inn
        or extracted.dosage_form
        or extracted.product_type != ProductType.UNKNOWN
        or extracted.release_type != ReleaseType.UNKNOWN
    )

    if _DOCUMENT_LOOKUP_RE.search(text) and not product_specific:
        reasons.append("Запрос похож на поиск конкретного документа или пункта.")
        return QueryClassification(QueryIntent.DOCUMENT_LOOKUP, extracted, reasons)

    if product_specific:
        reasons.append("В запросе распознаны характеристики конкретного препарата.")
        return QueryClassification(
            QueryIntent.PRODUCT_ASSESSMENT, extracted, reasons, product_specific=True
        )

    if any(marker in lowered for marker in _REGULATORY_MARKERS):
        reasons.append("Общий вопрос по нормативным требованиям ЕАЭС.")
        return QueryClassification(QueryIntent.GENERAL_REGULATORY, extracted, reasons)

    reasons.append(
        "Не удалось распознать регуляторную тематику; поиск выполняется по общему "
        "тексту запроса."
    )
    return QueryClassification(QueryIntent.GENERAL_REGULATORY, extracted, reasons)


# --------------------------------------------------------------------------- #
# Definitional consequences of the product type
# --------------------------------------------------------------------------- #


def classify_product(profile: ProductProfile) -> str:
    """A neutral, non-normative description of the product's category.

    This describes *what the user declared*, and never states what studies are
    needed — that is exclusively the job of the retrieved regulatory text.
    """
    from app.regulatory.schemas import PRODUCT_TYPE_LABELS, RELEASE_TYPE_LABELS

    if profile.product_type == ProductType.UNKNOWN:
        return (
            "Категория препарата не определена по введённым данным. "
            "Без указания типа препарата регистрационная стратегия не может быть "
            "определена."
        )

    parts = [
        f"Заявлен как {PRODUCT_TYPE_LABELS[profile.product_type]}",
    ]
    if profile.dosage_form:
        parts.append(f"лекарственная форма — {profile.dosage_form}")
    if profile.release_type != ReleaseType.UNKNOWN:
        parts.append(RELEASE_TYPE_LABELS[profile.release_type])
    if profile.route_of_administration:
        parts.append(f"путь введения — {profile.route_of_administration}")

    notes: list[str] = []
    if profile.product_type == ProductType.BIOSIMILAR:
        notes.append(
            "Биоаналоговый препарат по определению относится к биологическим "
            "лекарственным препаратам."
        )
    if profile.product_type == ProductType.GENERIC and not profile.reference_product:
        notes.append("Референтный лекарственный препарат не указан.")
    if profile.highly_variable:
        notes.append("Пользователь указал на высокую вариабельность препарата.")
    if profile.narrow_therapeutic_index:
        notes.append("Пользователь указал на узкий терапевтический диапазон.")

    return ". ".join([", ".join(parts), *notes]).strip()
