"""Intake: turning free-form user input into a :class:`ProductProfile`.

Deliberately *deterministic and rule-based*. The system must never guess a
product's category and then produce a categorical study programme from that
guess — so extraction only fills a field when the user's wording states it, and
everything else stays ``unknown`` and triggers a clarifying question.
"""

from __future__ import annotations

import re

from app.regulatory.schemas import ProductProfile, ProductType, ReleaseType

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

_DOSAGE_FORMS: tuple[tuple[str, str], ...] = (
    ("таблетк", "таблетки"),
    ("капсул", "капсулы"),
    ("суспенз", "суспензия"),
    ("сироп", "сироп"),
    ("раствор для инъекц", "раствор для инъекций"),
    ("раствор для инфуз", "раствор для инфузий"),
    ("лиофилизат", "лиофилизат"),
    ("порошок", "порошок"),
    ("гранул", "гранулы"),
    ("суппозитор", "суппозитории"),
    ("мазь", "мазь"),
    ("крем", "крем"),
    ("гель", "гель"),
    ("пластыр", "трансдермальный пластырь"),
    ("спрей", "спрей"),
    ("аэрозол", "аэрозоль"),
    ("капл", "капли"),
    ("раствор", "раствор"),
)

_ROUTES: tuple[tuple[str, str], ...] = (
    ("перорал", "пероральный"),
    ("внутрь", "пероральный"),
    ("внутривен", "внутривенный"),
    ("внутримышеч", "внутримышечный"),
    ("подкожн", "подкожный"),
    ("ингаляц", "ингаляционный"),
    ("накожн", "накожный"),
    ("наружн", "наружный"),
    ("трансдермал", "трансдермальный"),
    ("ректал", "ректальный"),
    ("вагинал", "вагинальный"),
    ("офтальм", "офтальмологический"),
    ("интраназал", "интраназальный"),
    ("сублингвал", "сублингвальный"),
)

_PRODUCT_TYPE_PATTERNS: tuple[tuple[str, ProductType], ...] = (
    ("воспроизвед", ProductType.GENERIC),
    ("генерик", ProductType.GENERIC),
    ("дженерик", ProductType.GENERIC),
    ("generic", ProductType.GENERIC),
    ("гибридн", ProductType.HYBRID),
    ("hybrid", ProductType.HYBRID),
    ("биоаналог", ProductType.BIOSIMILAR),
    ("биоподобн", ProductType.BIOSIMILAR),
    ("biosimilar", ProductType.BIOSIMILAR),
    ("биологическ", ProductType.BIOLOGICAL),
    ("biological", ProductType.BIOLOGICAL),
    ("оригинальн", ProductType.ORIGINAL),
    ("референтн препарат разраб", ProductType.ORIGINAL),
    ("original", ProductType.ORIGINAL),
    ("хорошо изученн", ProductType.WELL_ESTABLISHED_USE),
    ("well.established", ProductType.WELL_ESTABLISHED_USE),
    ("комбинированн", ProductType.COMBINATION),
    ("фиксированн комбинац", ProductType.COMBINATION),
)

_RELEASE_PATTERNS: tuple[tuple[str, ReleaseType], ...] = (
    ("немедленн", ReleaseType.IMMEDIATE),
    ("immediate", ReleaseType.IMMEDIATE),
    ("отсроченн", ReleaseType.DELAYED),
    ("кишечнораствор", ReleaseType.DELAYED),
    ("delayed", ReleaseType.DELAYED),
    ("пролонгированн", ReleaseType.PROLONGED),
    ("продлённ", ReleaseType.PROLONGED),
    ("prolonged", ReleaseType.PROLONGED),
    ("модифицированн", ReleaseType.MODIFIED),
    ("modified", ReleaseType.MODIFIED),
)

# "400 мг", "20 мг/мл", "0,5 г"
_STRENGTH_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:мг|мкг|г|ме|ед|ммоль)(?:\s*/\s*(?:мл|г|доза|таблетк\w*))?",
    re.IGNORECASE,
)

_LABELLED_FIELD_RE = re.compile(
    r"^\s*(?P<label>мнн|международное непатентованное наименование|"
    r"лекарственная форма|дозировка|путь введения|тип препарата|категория препарата|"
    r"высвобождение|характер высвобождения|референтный препарат)\s*[:\-–]\s*(?P<value>.+)$",
    re.IGNORECASE | re.MULTILINE,
)

_HIGH_VARIABILITY_RE = re.compile(r"высоковариабельн|highly\s+variable", re.IGNORECASE)
_NTI_RE = re.compile(
    r"узк(?:им|ий|ого)\s+терапевтическ|narrow\s+therapeutic", re.IGNORECASE
)

_QUESTION_LABELS: dict[str, str] = {
    "inn": "международное непатентованное наименование (МНН) действующего вещества",
    "dosage_form": "лекарственную форму",
    "strength": "дозировку (силу действия)",
    "route_of_administration": "путь введения",
    "product_type": (
        "тип препарата (оригинальный, воспроизведённый, гибридный, "
        "биологический, биоаналоговый, комбинированный)"
    ),
    "release_type": "характер высвобождения (немедленное / модифицированное)",
}


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


def _first_match(text: str, table: tuple[tuple[str, str], ...]) -> str:
    for needle, value in table:
        if needle in text:
            return value
    return ""


def extract_profile(text: str, base: ProductProfile | None = None) -> ProductProfile:
    """Build a :class:`ProductProfile` from free text or a labelled block.

    Labelled lines ("МНН: ибупрофен") take precedence; otherwise the whole text
    is scanned for known vocabulary.  Unrecognised aspects stay ``unknown``.
    """
    profile = (base or ProductProfile()).model_copy(deep=True)
    profile.free_text = text.strip()
    lowered = text.lower().replace("ё", "е")

    labelled: dict[str, str] = {}
    for match in _LABELLED_FIELD_RE.finditer(text):
        labelled[match.group("label").lower()] = match.group("value").strip()

    # -- INN ---------------------------------------------------------------
    inn = labelled.get("мнн") or labelled.get(
        "международное непатентованное наименование", ""
    )
    if inn:
        profile.inn = inn.strip(" .;")
    elif not profile.inn:
        profile.inn = _guess_inn(text)

    # -- dosage form -------------------------------------------------------
    form = labelled.get("лекарственная форма", "")
    profile.dosage_form = (
        form.strip(" .;") or profile.dosage_form or _first_match(lowered, _DOSAGE_FORMS)
    )

    # -- strength ----------------------------------------------------------
    strength = labelled.get("дозировка", "")
    if strength:
        profile.strength = strength.strip(" .;")
    elif not profile.strength:
        found = _STRENGTH_RE.search(text)
        profile.strength = found.group(0).strip() if found else ""

    # -- route -------------------------------------------------------------
    route = labelled.get("путь введения", "")
    profile.route_of_administration = (
        route.strip(" .;")
        or profile.route_of_administration
        or _first_match(lowered, _ROUTES)
    )

    # -- product type ------------------------------------------------------
    type_text = (
        labelled.get("тип препарата", "") or labelled.get("категория препарата", "")
    ).lower()
    if profile.product_type == ProductType.UNKNOWN:
        for needle, value in _PRODUCT_TYPE_PATTERNS:
            if re.search(needle, type_text or lowered):
                profile.product_type = value
                break

    # -- release type ------------------------------------------------------
    release_text = (
        labelled.get("характер высвобождения", "") or labelled.get("высвобождение", "")
    ).lower()
    if profile.release_type == ReleaseType.UNKNOWN:
        for needle, value in _RELEASE_PATTERNS:
            if needle in (release_text or lowered):
                profile.release_type = value
                break

    # -- reference product -------------------------------------------------
    reference = labelled.get("референтный препарат", "")
    if reference and reference.lower() not in ("не выбран", "нет", "-", "—"):
        profile.reference_product = reference.strip(" .;")

    # -- flags -------------------------------------------------------------
    if _HIGH_VARIABILITY_RE.search(text):
        profile.highly_variable = True
    if _NTI_RE.search(text):
        profile.narrow_therapeutic_index = True
    if profile.product_type in (ProductType.BIOLOGICAL, ProductType.BIOSIMILAR):
        profile.biological = True
        profile.biosimilar = profile.product_type == ProductType.BIOSIMILAR
    if profile.product_type == ProductType.COMBINATION:
        profile.combination_product = True

    return profile


_STOP_FOR_INN = {
    "препарат", "лекарственный", "таблетки", "капсулы", "раствор", "воспроизведенный",
    "оригинальный", "регистрация", "исследование", "биоэквивалентность", "мнн",
    "дозировка", "форма", "введения", "высвобождение",
}


def _guess_inn(text: str) -> str:
    """Very conservative INN guess: a single unknown Cyrillic word input.

    Only fires when the message is short and contains one plausible substance
    name — enough to support the "пользователь написал «метформин»" scenario
    without inventing an INN out of a long sentence.
    """
    # Imported lazily: the INN vocabulary is extended at ingestion time, so the
    # live module state must be consulted rather than a value bound at import.
    from app.regulatory.inn import detect_inns

    detected = detect_inns(text)
    if detected:
        return detected[0]

    words = re.findall(r"[А-Яа-яЁё]{5,30}", text)
    candidates = [
        word
        for word in words
        if word.lower().replace("ё", "е") not in _STOP_FOR_INN
    ]
    if len(text.split()) <= 4 and len(candidates) == 1:
        return candidates[0].lower()
    return ""


# --------------------------------------------------------------------------- #
# Sufficiency
# --------------------------------------------------------------------------- #


#: Order in which missing data is asked for, most decision-changing first: the
#: product type determines the whole registration route, while the route of
#: administration rarely changes the answer on its own.  Callers that show only
#: the first few questions therefore show the ones that matter.
_QUESTION_PRIORITY: tuple[str, ...] = (
    "product_type",
    "dosage_form",
    "release_type",
    "strength",
    "route_of_administration",
    "inn",
)


def clarifying_questions(profile: ProductProfile) -> list[str]:
    """Questions that must be answered before a study programme can be derived."""
    missing = sorted(
        profile.missing_fields(),
        key=lambda name: _QUESTION_PRIORITY.index(name)
        if name in _QUESTION_PRIORITY
        else len(_QUESTION_PRIORITY),
    )
    questions = [
        f"Укажите {_QUESTION_LABELS[name]}."
        for name in missing
        if name in _QUESTION_LABELS
    ]
    if not profile.reference_product and profile.product_type in (
        ProductType.GENERIC,
        ProductType.HYBRID,
        ProductType.BIOSIMILAR,
        ProductType.UNKNOWN,
    ):
        questions.append(
            "Выбран ли референтный лекарственный препарат и на основании чего?"
        )
    return questions
