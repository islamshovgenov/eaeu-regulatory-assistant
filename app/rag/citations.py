"""Citation construction and validation.

Citation identifiers are assigned **programmatically** from retrieval results
before the LLM is called.  The model may only reference existing ids; it can
never invent a document, a paragraph number or a URL, because it never writes a
citation object — only an integer that must already exist.

:func:`validate_citations` enforces that contract after generation and strips
anything that does not resolve.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date

from app.db.models import (
    Chunk,
    DocumentRecord,
    DocumentStatus,
    DocumentType,
    SourceAuthority,
    SourceTier,
    VersionStatus,
)
from app.rag.hybrid_search import RetrievedChunk

logger = logging.getLogger(__name__)

TIER_LABELS: dict[SourceTier, str] = {
    SourceTier.TIER_1: "Обязательный нормативный акт ЕАЭС",
    SourceTier.TIER_2: "Руководство/рекомендация ЕЭК",
    SourceTier.TIER_3: "Рекомендация Экспертного комитета ЕЭК",
    SourceTier.TIER_4: "Дополнительный международный источник (не обязателен в ЕАЭС)",
    SourceTier.TIER_5: "Научный источник (не нормативный)",
}

STATUS_LABELS: dict[DocumentStatus, str] = {
    DocumentStatus.ACTIVE: "действует",
    DocumentStatus.AMENDED: "действует с изменениями",
    DocumentStatus.SUPERSEDED: "утратил силу / заменён",
    DocumentStatus.EXPIRED: "истёк срок действия",
    DocumentStatus.UNKNOWN: "статус не подтверждён автоматически",
}

VERSION_STATUS_LABELS: dict[VersionStatus, str] = {
    VersionStatus.CONSOLIDATED: "консолидированная редакция источника",
    VersionStatus.ORIGINAL_WITH_KNOWN_AMENDMENTS: (
        "первоначальная редакция; известны акты о внесении изменений"
    ),
    VersionStatus.REQUIRES_EXPERT_VALIDATION: (
        "действующая редакция не подтверждена автоматически — требуется "
        "экспертная проверка"
    ),
}


@dataclass(slots=True)
class Citation:
    """One numbered source shown to the user and referenced by the model."""

    citation_id: int
    chunk: Chunk
    retrieval_score: float = 0.0
    retrieval_methods: list[str] = field(default_factory=list)

    # -- derived views ------------------------------------------------------
    @property
    def document_id(self) -> str:
        return self.chunk.document_id

    @property
    def tier(self) -> SourceTier:
        return self.chunk.source_tier

    @property
    def tier_label(self) -> str:
        return TIER_LABELS.get(self.tier, "Источник")

    @property
    def status_label(self) -> str:
        return STATUS_LABELS.get(self.chunk.status, "статус неизвестен")

    @property
    def version_label(self) -> str:
        return VERSION_STATUS_LABELS.get(self.chunk.version_status, "")

    @property
    def is_binding_eaeu(self) -> bool:
        return self.chunk.source_authority == SourceAuthority.EAEU and self.tier in (
            SourceTier.TIER_1,
            SourceTier.TIER_2,
        )

    def location(self) -> str:
        return self.chunk.breadcrumb()

    def pages(self) -> str:
        start = self.chunk.metadata.page_start
        end = self.chunk.metadata.page_end
        if start and end and start != end:
            return f"с. {start}–{end}"
        if start:
            return f"с. {start}"
        return ""

    def header(self) -> str:
        """``[3] Решение Совета ЕЭК № 85 от 03.11.2016 · Приложение № 1 · п. 42``"""
        parts = [self.chunk.document_short_title or self.chunk.document_title[:110]]
        if location := self.location():
            parts.append(location)
        if pages := self.pages():
            parts.append(pages)
        return " · ".join(parts)

    def snippet(self, max_chars: int = 700) -> str:
        text = self.chunk.text
        # Drop the bracketed context prefix added by the chunker.
        text = re.sub(r"^\[[^\]]{0,300}\]\n", "", text, count=1)
        return text if len(text) <= max_chars else text[:max_chars].rstrip() + "…"


#: Angle-bracket runs are the fragment delimiters; neutralising them inside the
#: quoted text prevents a document (or an attacker who got text into one) from
#: forging a fragment boundary and appearing to speak outside the data block.
_DELIMITER_RE = re.compile(r"<{3,}|>{3,}")


def _neutralise_delimiters(text: str) -> str:
    return _DELIMITER_RE.sub(lambda m: m.group(0)[0] * 2, text)


class CitationRegistry:
    """Numbered citations for one answer, plus helpers to validate references."""

    def __init__(self, citations: list[Citation]) -> None:
        self._citations = citations
        self._by_id = {c.citation_id: c for c in citations}

    # -- construction -------------------------------------------------------
    @classmethod
    def from_retrieval(
        cls, items: list[RetrievedChunk], access_dates: dict[str, date] | None = None
    ) -> "CitationRegistry":
        citations = [
            Citation(
                citation_id=index,
                chunk=item.chunk,
                retrieval_score=item.score,
                retrieval_methods=item.retrieval_methods(),
            )
            for index, item in enumerate(items, start=1)
        ]
        return cls(citations)

    # -- access -------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._citations)

    def __iter__(self):
        return iter(self._citations)

    @property
    def citations(self) -> list[Citation]:
        return list(self._citations)

    def get(self, citation_id: int) -> Citation | None:
        return self._by_id.get(citation_id)

    def valid_ids(self) -> set[int]:
        return set(self._by_id)

    def binding_ids(self) -> set[int]:
        return {c.citation_id for c in self._citations if c.is_binding_eaeu}

    def used(self, ids: set[int]) -> list[Citation]:
        return [c for c in self._citations if c.citation_id in ids]

    # -- rendering ----------------------------------------------------------
    def context_block(self, max_chars_per_chunk: int = 2600) -> str:
        """The RETRIEVED REGULATORY CONTEXT block handed to the LLM.

        Each fragment is wrapped in explicit delimiters and labelled with its
        citation id, authority tier and status — this is *data*, and the prompt
        says so (see :mod:`app.rag.prompts`).
        """
        blocks: list[str] = []
        for citation in self._citations:
            chunk = citation.chunk
            text = _neutralise_delimiters(chunk.text[:max_chars_per_chunk])
            header = (
                f"[{citation.citation_id}] {citation.header()}\n"
                f"    источник: {chunk.source_authority} | {citation.tier_label}\n"
                f"    статус: {citation.status_label} | {citation.version_label}"
            )
            blocks.append(
                f"<<<ФРАГМЕНТ {citation.citation_id} НАЧАЛО>>>\n"
                f"{header}\n"
                f"    текст:\n{text}\n"
                f"<<<ФРАГМЕНТ {citation.citation_id} КОНЕЦ>>>"
            )
        return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

#: Patterns the model must not produce outside a citation: an explicit act
#: reference with a number.  Any such reference is checked against the corpus.
_ACT_REFERENCE_RE = re.compile(
    r"(?:решени|рекомендаци|распоряжени)[а-яё]*\s+"
    r"(?:совета|коллегии)?[^№.]{0,40}№\s*(\d{1,4})",
    re.IGNORECASE,
)


@dataclass(slots=True)
class ValidationReport:
    """Outcome of validating a generated answer against the citation registry."""

    invalid_citation_ids: list[int] = field(default_factory=list)
    unsupported_act_numbers: list[str] = field(default_factory=list)
    statements_without_citation: int = 0
    removed_citations: int = 0

    @property
    def ok(self) -> bool:
        return not (self.invalid_citation_ids or self.unsupported_act_numbers)

    def messages(self) -> list[str]:
        messages: list[str] = []
        if self.invalid_citation_ids:
            messages.append(
                "Из ответа удалены несуществующие ссылки на источники: "
                + ", ".join(f"[{i}]" for i in sorted(self.invalid_citation_ids))
            )
        if self.unsupported_act_numbers:
            messages.append(
                "В тексте ответа упомянуты номера актов, отсутствующие в найденном "
                "контексте: "
                + ", ".join(f"№ {n}" for n in sorted(set(self.unsupported_act_numbers)))
                + ". Эти упоминания требуют экспертной проверки."
            )
        if self.statements_without_citation:
            messages.append(
                f"Утверждений без нормативной ссылки: {self.statements_without_citation}. "
                "Они помечены как интерпретация, а не как требование."
            )
        return messages


def validate_citation_ids(
    ids: list[int], registry: CitationRegistry, report: ValidationReport
) -> list[int]:
    """Keep only ids that exist in the registry; record the rest."""
    valid = registry.valid_ids()
    kept: list[int] = []
    for citation_id in ids:
        if citation_id in valid:
            kept.append(citation_id)
        else:
            report.invalid_citation_ids.append(citation_id)
            report.removed_citations += 1
    return kept


def check_act_numbers(
    text: str,
    registry: CitationRegistry,
    documents: list[DocumentRecord] | set[str],
) -> list[str]:
    """Report act numbers mentioned in *text* that the corpus does not contain.

    ``documents`` accepts either the document records or a ready-made set of
    act numbers (cheaper when the caller already has one).
    """
    if isinstance(documents, set):
        corpus_numbers = set(documents)
    else:
        corpus_numbers = {
            document.document_number
            for document in documents
            if document.document_number
        }
    corpus_numbers.update(
        citation.chunk.document_number
        for citation in registry
        if citation.chunk.document_number
    )
    mentioned = set(_ACT_REFERENCE_RE.findall(text or ""))
    return sorted(mentioned - corpus_numbers)


def document_type_is_supplementary(document_type: DocumentType) -> bool:
    return document_type == DocumentType.SUPPLEMENTARY
