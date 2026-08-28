"""Extraction and classification of document metadata.

Everything here is *derived from the source itself* (page title, anchor text,
URL slug, file name).  Nothing is invented: when a field cannot be established
it stays empty and the document is flagged
``version_status = requires_expert_validation``.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from urllib.parse import unquote, urlparse

from app.db.models import (
    DocumentStatus,
    DocumentType,
    SourceAuthority,
    SourceTier,
    VersionStatus,
)

# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #

_RU_MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6,
    "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11,
    "декабря": 12,
}

# "03.11.2016" / "3.11.2016"
_DOT_DATE_RE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b")
# "3 ноября 2016"
_RU_DATE_RE = re.compile(
    r"\b(\d{1,2})\s+(" + "|".join(_RU_MONTHS) + r")\s+(\d{4})", re.IGNORECASE
)
# slug fragment "cncd_21112016_85" -> 21112016
_SLUG_DATE_RE = re.compile(r"_(\d{2})(\d{2})(\d{4})_")


def parse_date(text: str) -> date | None:
    """Parse the first date found in *text*; ``None`` when absent/invalid."""
    if not text:
        return None
    match = _DOT_DATE_RE.search(text)
    if match:
        day, month, year = (int(g) for g in match.groups())
        try:
            return date(year, month, day)
        except ValueError:
            return None
    match = _RU_DATE_RE.search(text)
    if match:
        day = int(match.group(1))
        month = _RU_MONTHS[match.group(2).lower()]
        year = int(match.group(3))
        try:
            return date(year, month, day)
        except ValueError:
            return None
    return None


def date_from_slug(url: str) -> date | None:
    """Publication date encoded in docs.eaeunion.org slugs (``err_14052024_30``)."""
    match = _SLUG_DATE_RE.search(url)
    if not match:
        return None
    day, month, year = (int(g) for g in match.groups())
    try:
        return date(year, month, day)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# Titles from docs.eaeunion.org
# --------------------------------------------------------------------------- #

# "Решение Совета ЕЭК № 85 от 03.11.2016 Об утверждении Правил ..."
_TITLE_RE = re.compile(
    r"^(?P<kind>Решение\s+Совета|Решение\s+Коллегии|Рекомендация\s+Коллегии|"
    r"Рекомендация\s+Совета|Распоряжение\s+Совета|Распоряжение\s+Коллегии|"
    r"Договор|Соглашение|Протокол)"
    r"[^№]*?№\s*(?P<number>[\dA-Za-zА-Яа-я\-/]+)"
    r"(?:\s*от\s*(?P<date>[\d.]{8,10}))?"
    r"\s*(?P<rest>.*)$",
    re.IGNORECASE | re.DOTALL,
)

_KIND_TO_TYPE = {
    "решение совета": DocumentType.COUNCIL_DECISION,
    "решение коллегии": DocumentType.COLLEGE_DECISION,
    "рекомендация коллегии": DocumentType.COLLEGE_RECOMMENDATION,
    "рекомендация совета": DocumentType.COUNCIL_RECOMMENDATION,
    "распоряжение совета": DocumentType.COUNCIL_DISPOSITION,
    "распоряжение коллегии": DocumentType.COUNCIL_DISPOSITION,
    "договор": DocumentType.AGREEMENT,
    "соглашение": DocumentType.AGREEMENT,
    "протокол": DocumentType.AGREEMENT,
}

_SHORT_KIND = {
    DocumentType.COUNCIL_DECISION: "Решение Совета ЕЭК",
    DocumentType.COLLEGE_DECISION: "Решение Коллегии ЕЭК",
    DocumentType.COLLEGE_RECOMMENDATION: "Рекомендация Коллегии ЕЭК",
    DocumentType.COUNCIL_RECOMMENDATION: "Рекомендация Совета ЕЭК",
    DocumentType.COUNCIL_DISPOSITION: "Распоряжение Совета ЕЭК",
    DocumentType.AGREEMENT: "Соглашение/Договор ЕАЭС",
    DocumentType.EXPERT_COMMITTEE_RECOMMENDATION: "Рекомендация Экспертного комитета ЕЭК",
}


class ParsedTitle:
    """Structured view of a docs.eaeunion.org page title."""

    __slots__ = ("title", "document_type", "number", "adoption_date", "short_title")

    def __init__(
        self,
        title: str,
        document_type: DocumentType,
        number: str,
        adoption_date: date | None,
        short_title: str,
    ) -> None:
        self.title = title
        self.document_type = document_type
        self.number = number
        self.adoption_date = adoption_date
        self.short_title = short_title

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"ParsedTitle(type={self.document_type}, number={self.number!r}, "
            f"date={self.adoption_date}, title={self.title[:60]!r})"
        )


def parse_document_title(raw_title: str) -> ParsedTitle:
    """Split a page title into type / number / date / subject."""
    clean = re.sub(r"\s+", " ", (raw_title or "")).strip()
    match = _TITLE_RE.match(clean)
    if not match:
        return ParsedTitle(
            title=clean,
            document_type=DocumentType.UNKNOWN,
            number="",
            adoption_date=parse_date(clean),
            short_title=clean[:120],
        )

    kind = re.sub(r"\s+", " ", match.group("kind")).strip().lower()
    doc_type = _KIND_TO_TYPE.get(kind, DocumentType.UNKNOWN)
    number = (match.group("number") or "").strip()
    adopted = parse_date(match.group("date") or "") or parse_date(clean)
    subject = (match.group("rest") or "").strip(" «»\"—-")

    short = _SHORT_KIND.get(doc_type, "Документ ЕЭК")
    if number:
        short += f" № {number}"
    if adopted:
        short += f" от {adopted.strftime('%d.%m.%Y')}"

    title = clean if subject == "" else f"{short}. {subject}"
    return ParsedTitle(title, doc_type, number, adopted, short)


# --------------------------------------------------------------------------- #
# Amendments
# --------------------------------------------------------------------------- #

_AMENDMENT_MARKERS = (
    "о внесении изменен",
    "о внесении измен",
    "внесении изменений",
    "о признании утратив",
)


def looks_like_amendment(title: str) -> bool:
    lowered = title.lower()
    return any(marker in lowered for marker in _AMENDMENT_MARKERS)


_AMENDED_TARGET_RE = re.compile(
    r"№\s*(\d{1,3})\b", re.IGNORECASE
)


def amended_document_numbers(title: str) -> list[str]:
    """Decision numbers referenced by an amending act (first is its own number)."""
    numbers = _AMENDED_TARGET_RE.findall(title)
    return numbers[1:] if len(numbers) > 1 else []


# --------------------------------------------------------------------------- #
# Topics
# --------------------------------------------------------------------------- #

TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "registration": ("регистрац", "экспертиз", "регистрационн", "досье"),
    "bioequivalence": ("биоэквивалент", "биодоступн", "biowaiver", "биовейвер"),
    "gcp": ("надлежащей клинической практики", "gcp"),
    "glp": ("надлежащей лабораторной практики", "glp"),
    "gmp": ("надлежащей производственной практики", "gmp"),
    "gvp": ("фармаконадзор",),
    "clinical": ("клиническ", "клинических исследован"),
    "nonclinical": ("доклиническ", "токсич", "канцероген", "генотоксич", "репродуктивн"),
    "biological": ("биологическ", "биоаналог", "биоподобн", "иммуноген"),
    "quality": ("качеств", "спецификац", "валидац", "примес", "стабильн", "фармацевтическ разработ"),
    "smpc": ("общей характеристик", "инструкции по медицинскому применению", "охлп", "маркиров"),
    "modified_release": ("модифицированным высвобожден", "пролонгированн"),
    "pharmacopoeia": ("фармакопе",),
    "reference_product": ("референтн",),
    "pediatric": ("педиатрическ", "детей"),
    "herbal": ("растительн",),
    "inspection": ("инспект",),
    "labeling": ("маркиров",),
}


def infer_topics(*texts: str) -> list[str]:
    """Keyword-based topic tags (used for filtering and corpus statistics)."""
    blob = " ".join(t.lower() for t in texts if t)
    topics = [
        topic
        for topic, keywords in TOPIC_KEYWORDS.items()
        if any(keyword in blob for keyword in keywords)
    ]
    return sorted(set(topics))


# --------------------------------------------------------------------------- #
# Language of a file
# --------------------------------------------------------------------------- #

# docs.eaeunion.org publishes the same act in five official languages.  These
# markers appear in the *file name* of the non-Russian versions.
_NON_RUSSIAN_MARKERS = (
    "_arm", "-arm", "_hy", "_kaz", "-kaz", "_kz", "_kk", "_kyr", "-kyr", "_kg",
    "_ky", "_bel", "-bel", "_blr", "-blr", "_be", "_eng", "-eng", "_en", "-en",
    "rashenne", "shesh", "chech", "voroshum", "ethx", "sheshimi", "karar",
    "toktom", "_arm.", "arm.docx", "_alb",
)

_RUSSIAN_MARKERS = (
    "reshenie", "rekomendatsiya", "rasporyazhenie", "_doc", "_rus", "-rus",
    "soveta", "kollegii", "prilozhenie", "izmeneniya",
)

_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")


def file_looks_russian(url: str) -> bool:
    """Heuristic language filter for EEC attachment file names.

    The EEC publishes each act in Russian, Armenian, Belarusian, Kazakh and
    Kyrgyz.  Only the Russian text is authoritative for this project's corpus,
    so non-Russian variants are filtered out by their file-name markers.
    """
    name = unquote(urlparse(url).path.rsplit("/", 1)[-1]).lower()
    stem = name.rsplit(".", 1)[0]

    if any(marker in stem for marker in _NON_RUSSIAN_MARKERS):
        # "_en"/"-en" are short; require they are a suffix or bounded token.
        for marker in ("_en", "-en", "_be", "_ky", "_kk", "_hy"):
            if stem.endswith(marker):
                return False
        for marker in _NON_RUSSIAN_MARKERS:
            if marker in ("_en", "-en", "_be", "_ky", "_kk", "_hy"):
                continue
            if marker in stem:
                return False
    if _CYRILLIC_RE.search(unquote(name)):
        return True
    return any(marker in stem for marker in _RUSSIAN_MARKERS) or bool(
        re.match(r"^(cncd|clcd|clcr|err|itia)_", stem)
    )


# --------------------------------------------------------------------------- #
# Identifiers
# --------------------------------------------------------------------------- #


def slugify(value: str, max_length: int = 80) -> str:
    """ASCII-ish slug suitable for file names and document ids."""
    normalised = unicodedata.normalize("NFKD", value)
    ascii_only = normalised.encode("ascii", "ignore").decode("ascii")
    if not ascii_only.strip():
        ascii_only = re.sub(r"[^0-9A-Za-zА-Яа-яЁё]+", "_", value)
    slug = re.sub(r"[^0-9A-Za-zА-Яа-яЁё]+", "_", ascii_only).strip("_")
    return slug[:max_length] or "document"


def build_document_id(
    source_authority: SourceAuthority,
    document_type: DocumentType,
    number: str,
    adoption: date | None,
    fallback: str,
) -> str:
    """Stable, human-readable document identifier.

    Example: ``EAEU_COUNCIL_DECISION_85_2016``.
    """
    parts: list[str] = [str(source_authority)]
    if document_type != DocumentType.UNKNOWN:
        parts.append(str(document_type).upper())
    if number:
        parts.append(re.sub(r"[^0-9A-Za-z]+", "", number).upper())
    if adoption:
        parts.append(str(adoption.year))
    if len(parts) <= 2:
        parts.append(slugify(fallback, 40).upper())
    return "_".join(p for p in parts if p)


# --------------------------------------------------------------------------- #
# Tier / status
# --------------------------------------------------------------------------- #


def tier_for(
    source_authority: SourceAuthority, document_type: DocumentType
) -> SourceTier:
    """Normative weight of a document, derived from its issuing body and kind.

    TIER 1 — binding acts (Council/Collegium decisions, dispositions, the
    agreements themselves, and acts amending them);
    TIER 2 — official guidance (Collegium/Council *recommendations*, guidelines);
    TIER 3 — Expert Committee recommendations;
    TIER 4 — ICH / EMA / WHO.
    """
    if source_authority != SourceAuthority.EAEU:
        return SourceTier.TIER_4
    if document_type == DocumentType.SUPPLEMENTARY:
        return SourceTier.TIER_4
    if document_type == DocumentType.EXPERT_COMMITTEE_RECOMMENDATION:
        return SourceTier.TIER_3
    if document_type in (
        DocumentType.COLLEGE_RECOMMENDATION,
        DocumentType.COUNCIL_RECOMMENDATION,
        DocumentType.GUIDELINE,
    ):
        return SourceTier.TIER_2
    return SourceTier.TIER_1


def initial_status(title: str, is_obsolete_section: bool) -> DocumentStatus:
    """Status derived from explicit source markers only."""
    lowered = title.lower()
    if is_obsolete_section or "утратил силу" in lowered or "утратили силу" in lowered:
        return DocumentStatus.SUPERSEDED
    return DocumentStatus.UNKNOWN


def version_status_for(has_amendments: bool, is_amendment: bool) -> VersionStatus:
    """We never synthesise a consolidated text, so the honest values are:"""
    if is_amendment:
        return VersionStatus.ORIGINAL_WITH_KNOWN_AMENDMENTS
    if has_amendments:
        return VersionStatus.ORIGINAL_WITH_KNOWN_AMENDMENTS
    return VersionStatus.REQUIRES_EXPERT_VALIDATION
