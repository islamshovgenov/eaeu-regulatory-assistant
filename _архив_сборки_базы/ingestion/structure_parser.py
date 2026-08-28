"""Structure-aware parsing of EAEU regulatory texts.

Regulatory acts of the EAEU follow a stable layout::

    <Решение / Правила>
      ПРИЛОЖЕНИЕ № 1
        I. Общие положения                 <- раздел (римская нумерация)
          Глава 2. ...                     <- глава
            12. Текст пункта ...           <- пункт
              а) подпункт ...              <- подпункт

This module converts a flat list of :class:`~ingestion.parser.TextBlock` into
:class:`StructuralUnit` objects that carry their *full parent context*, so a
retrieved fragment can always be cited as
``Приложение № 1 · Раздел III · Глава 2 · п. 12, подп. а``.

The regexes below are deliberately explicit and commented — they encode the
drafting conventions of EEC acts, not generic heuristics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.db.models import ChunkMetadata
from ingestion.parser import TextBlock

# --------------------------------------------------------------------------- #
# Structural markers
# --------------------------------------------------------------------------- #

# "ПРИЛОЖЕНИЕ № 2", "Приложение №1 к Правилам", "ПРИЛОЖЕНИЕ" (без номера)
APPENDIX_RE = re.compile(
    r"^\s*ПРИЛОЖЕНИ[ЕЯ]\s*(?:№\s*(?P<number>[\dIVXА-Яа-я]+))?\b",
    re.IGNORECASE,
)

# "I. ОБЩИЕ ПОЛОЖЕНИЯ", "III. Требования к ..." — римский раздел
ROMAN_SECTION_RE = re.compile(
    r"^\s*(?P<number>[IVXLC]{1,6})\.\s+(?P<title>[^\d].{2,200})$"
)

# "Раздел III", "РАЗДЕЛ 2. Наименование"
SECTION_RE = re.compile(
    r"^\s*РАЗДЕЛ\s+(?P<number>[IVXLC\d]+)\.?\s*(?P<title>.*)$", re.IGNORECASE
)

# "Глава 3. Наименование"
CHAPTER_RE = re.compile(
    r"^\s*ГЛАВА\s+(?P<number>[IVXLC\d]+)\.?\s*(?P<title>.*)$", re.IGNORECASE
)

# "Часть II"
PART_RE = re.compile(r"^\s*ЧАСТЬ\s+(?P<number>[IVXLC\d]+)\.?\s*(?P<title>.*)$", re.IGNORECASE)

# Пункт: "12." / "12.3." / "12.3.1." в начале абзаца, далее текст
PARAGRAPH_RE = re.compile(r"^\s*(?P<number>\d{1,3}(?:\.\d{1,3}){0,3})\.\s+(?P<text>\S.*)$")

# Подпункт: "а)" "б)" "1)" "12)" в начале абзаца
SUBPARAGRAPH_RE = re.compile(
    r"^\s*(?P<number>[а-яa-z]|\d{1,2})\)\s+(?P<text>\S.*)$", re.IGNORECASE
)

# Заголовок без нумерации: короткая строка без завершающей точки, много заглавных
_UPPER_HEADING_RE = re.compile(r"^[^a-zа-я]{6,120}$")

_STRUCTURAL_PREFIXES = (
    APPENDIX_RE,
    SECTION_RE,
    CHAPTER_RE,
    PART_RE,
    ROMAN_SECTION_RE,
    PARAGRAPH_RE,
    SUBPARAGRAPH_RE,
)


def starts_new_structural_unit(line: str) -> bool:
    """``True`` when *line* begins a new numbered/structural unit.

    Used by the PDF paragraph re-assembler so a hard-wrapped page does not glue
    a new пункт onto the previous one.
    """
    return any(pattern.match(line) for pattern in _STRUCTURAL_PREFIXES)


# --------------------------------------------------------------------------- #
# Units
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class StructuralUnit:
    """A paragraph-level fragment together with its structural coordinates."""

    text: str
    metadata: ChunkMetadata
    is_heading: bool = False
    order: int = 0

    def context_key(self) -> tuple[str, ...]:
        """Identity of the parent context — units may only merge inside it."""
        return (
            self.metadata.appendix,
            self.metadata.part,
            self.metadata.section,
            self.metadata.chapter,
        )


@dataclass(slots=True)
class _State:
    appendix: str = ""
    part: str = ""
    section: str = ""
    chapter: str = ""
    subsection: str = ""
    heading: str = ""
    paragraph: str = ""
    subparagraph: str = ""
    pending_headings: list[str] = field(default_factory=list)

    def snapshot(
        self, page: int | None, is_table: bool
    ) -> ChunkMetadata:
        return ChunkMetadata(
            appendix=self.appendix,
            part=self.part,
            section=self.section,
            chapter=self.chapter,
            subsection=self.subsection,
            heading=self.heading,
            paragraph=self.paragraph,
            subparagraph=self.subparagraph,
            is_table=is_table,
            page_start=page,
            page_end=page,
        )


def _looks_like_heading(text: str, style: str) -> bool:
    if style.lower().startswith("heading") or style.lower().startswith("заголовок"):
        return True
    if len(text) > 160:
        return False
    if text.endswith((".", ";", ":")) and not text.isupper():
        return False
    return bool(_UPPER_HEADING_RE.match(text))


def parse_structure(blocks: list[TextBlock]) -> list[StructuralUnit]:
    """Convert flat blocks into context-carrying structural units."""
    state = _State()
    units: list[StructuralUnit] = []
    order = 0

    for block in blocks:
        text = block.text.strip()
        if not text:
            continue

        # --- containers ----------------------------------------------------
        match = APPENDIX_RE.match(text)
        if match and len(text) < 220:
            number = (match.group("number") or "").strip()
            state.appendix = f"№ {number}" if number else "б/н"
            state.part = state.section = state.chapter = ""
            state.subsection = state.heading = ""
            state.paragraph = state.subparagraph = ""
            units.append(
                StructuralUnit(text, state.snapshot(block.page, False), True, order)
            )
            order += 1
            continue

        match = PART_RE.match(text)
        if match and len(text) < 220:
            state.part = match.group("number")
            state.section = state.chapter = ""
            state.paragraph = state.subparagraph = ""
            state.heading = match.group("title").strip()
            units.append(
                StructuralUnit(text, state.snapshot(block.page, False), True, order)
            )
            order += 1
            continue

        match = SECTION_RE.match(text) or ROMAN_SECTION_RE.match(text)
        if match and len(text) < 220:
            state.section = match.group("number")
            state.chapter = ""
            state.subsection = ""
            state.paragraph = state.subparagraph = ""
            state.heading = (match.groupdict().get("title") or "").strip()
            units.append(
                StructuralUnit(text, state.snapshot(block.page, False), True, order)
            )
            order += 1
            continue

        match = CHAPTER_RE.match(text)
        if match and len(text) < 220:
            state.chapter = match.group("number")
            state.subsection = ""
            state.paragraph = state.subparagraph = ""
            state.heading = match.group("title").strip()
            units.append(
                StructuralUnit(text, state.snapshot(block.page, False), True, order)
            )
            order += 1
            continue

        # --- numbered provisions -------------------------------------------
        match = PARAGRAPH_RE.match(text)
        if match:
            state.paragraph = match.group("number")
            state.subparagraph = ""
            units.append(
                StructuralUnit(
                    text, state.snapshot(block.page, block.is_table), False, order
                )
            )
            order += 1
            continue

        match = SUBPARAGRAPH_RE.match(text)
        if match and state.paragraph:
            state.subparagraph = match.group("number")
            units.append(
                StructuralUnit(
                    text, state.snapshot(block.page, block.is_table), False, order
                )
            )
            order += 1
            continue

        # --- unnumbered heading --------------------------------------------
        if _looks_like_heading(text, block.style):
            state.subsection = text[:160]
            state.paragraph = state.subparagraph = ""
            units.append(
                StructuralUnit(text, state.snapshot(block.page, False), True, order)
            )
            order += 1
            continue

        # --- continuation of the current provision --------------------------
        units.append(
            StructuralUnit(
                text, state.snapshot(block.page, block.is_table), False, order
            )
        )
        order += 1

    return units


def merge_continuations(units: list[StructuralUnit]) -> list[StructuralUnit]:
    """Glue continuation lines back onto the provision they belong to.

    A unit with identical coordinates and no own numbering is appended to the
    previous unit; this repairs paragraphs split by page breaks.
    """
    merged: list[StructuralUnit] = []
    for unit in units:
        if (
            merged
            and not unit.is_heading
            and not merged[-1].is_heading
            and unit.metadata.paragraph == merged[-1].metadata.paragraph
            and unit.metadata.subparagraph == merged[-1].metadata.subparagraph
            and unit.context_key() == merged[-1].context_key()
            and unit.metadata.is_table == merged[-1].metadata.is_table
        ):
            previous = merged[-1]
            previous.text = f"{previous.text} {unit.text}".strip()
            if unit.metadata.page_end:
                previous.metadata.page_end = unit.metadata.page_end
            continue
        merged.append(unit)
    return merged
