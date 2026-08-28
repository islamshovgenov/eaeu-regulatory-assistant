"""Data models shared by ingestion, storage and retrieval.

These Pydantic models are the single source of truth for the shape of a
regulatory *document* and of a retrievable *chunk*.  The SQLite schema
(:mod:`app.db.repository`) and the Qdrant payload are both derived from them,
so metadata cannot silently diverge between stores.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class SourceAuthority(StrEnum):
    """Body that issued the document."""

    EAEU = "EAEU"
    ICH = "ICH"
    EMA = "EMA"
    WHO = "WHO"
    OTHER = "OTHER"


class DocumentType(StrEnum):
    """Legal/technical nature of the document."""

    AGREEMENT = "agreement"
    COUNCIL_DECISION = "council_decision"
    COUNCIL_DISPOSITION = "council_disposition"
    COUNCIL_RECOMMENDATION = "council_recommendation"
    COLLEGE_DECISION = "college_decision"
    COLLEGE_RECOMMENDATION = "college_recommendation"
    EXPERT_COMMITTEE_RECOMMENDATION = "expert_committee_recommendation"
    GUIDELINE = "guideline"
    AMENDMENT = "amendment"
    SUPPLEMENTARY = "supplementary"
    UNKNOWN = "unknown"


class DocumentStatus(StrEnum):
    """Lifecycle status of the *document as published*."""

    ACTIVE = "active"
    AMENDED = "amended"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


class VersionStatus(StrEnum):
    """Whether we can prove the stored file is the currently applicable text."""

    #: Source explicitly publishes a consolidated / current revision.
    CONSOLIDATED = "consolidated"
    #: Original text; amendments exist and are indexed separately.
    ORIGINAL_WITH_KNOWN_AMENDMENTS = "original_with_known_amendments"
    #: We cannot establish the applicable revision automatically.
    REQUIRES_EXPERT_VALIDATION = "requires_expert_validation"


class DownloadStatus(StrEnum):
    OK = "ok"
    MANUAL_REQUIRED = "manual_required"
    FAILED = "failed"
    SKIPPED = "skipped"


class SourceTier(StrEnum):
    """Normative weight of the source (see docs/rag_design.md)."""

    TIER_1 = "TIER_1"  # binding EAEU/EEC act
    TIER_2 = "TIER_2"  # official EEC guideline / recommendation
    TIER_3 = "TIER_3"  # EEC Expert Committee recommendation
    TIER_4 = "TIER_4"  # ICH / EMA / WHO
    TIER_5 = "TIER_5"  # scientific literature


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #


class DocumentRecord(BaseModel):
    """One regulatory document (one downloaded file) plus its provenance."""

    model_config = ConfigDict(use_enum_values=False)

    document_id: str
    title: str
    short_title: str = ""
    authority: str = "Евразийская экономическая комиссия"
    document_type: DocumentType = DocumentType.UNKNOWN
    document_number: str = ""
    adoption_date: date | None = None
    effective_date: date | None = None
    expiration_date: date | None = None
    status: DocumentStatus = DocumentStatus.UNKNOWN
    version_status: VersionStatus = VersionStatus.REQUIRES_EXPERT_VALIDATION
    jurisdiction: str = "EAEU"
    language: str = "ru"
    source_url: str = ""
    page_url: str = ""
    source_authority: SourceAuthority = SourceAuthority.EAEU
    source_tier: SourceTier = SourceTier.TIER_1
    parent_document: str = ""
    amends_document: str = ""
    amended_by: list[str] = Field(default_factory=list)
    version: str = "1"
    topic: list[str] = Field(default_factory=list)
    download_date: datetime | None = None
    download_status: DownloadStatus = DownloadStatus.OK
    sha256: str = ""
    size_bytes: int = 0
    content_type: str = ""
    local_file: str = ""
    n_pages: int | None = None
    n_chunks: int = 0
    notes: str = ""

    # -- convenience --------------------------------------------------------
    @property
    def is_binding_eaeu(self) -> bool:
        return self.source_authority == SourceAuthority.EAEU and self.source_tier in (
            SourceTier.TIER_1,
            SourceTier.TIER_2,
        )

    def human_citation(self) -> str:
        """Short human-readable reference to the document itself."""
        parts: list[str] = []
        if self.short_title:
            parts.append(self.short_title)
        else:
            parts.append(self.title[:160])
        if self.adoption_date:
            parts.append(f"от {self.adoption_date.strftime('%d.%m.%Y')}")
        return " ".join(p for p in parts if p)


# --------------------------------------------------------------------------- #
# Chunks
# --------------------------------------------------------------------------- #

_WS_RE = re.compile(r"\s+")

#: Heading given to the metadata-only chunk of a document without a text layer.
CARD_ONLY_HEADING = "Карточка документа (текст не извлечён)"

#: Letterhead lines that the structure parser picks up as a section title on
#: OCR'd documents.  They name the issuing body, which the citation already
#: shows, so as a location ("где в документе") they are pure noise.
_LETTERHEAD_LINES = frozenset(
    {
        "евразийская экономическая комиссия",
        "экспертный комитет по лекарственным средствам",
        "экспертный комитет по лекарственным средствам при коллегии",
        "рекомендация",
        "решение",
        "коллегия евразийской экономической комиссии",
        "совет евразийской экономической комиссии",
    }
)


def _is_letterhead(value: str) -> bool:
    return value.strip().lower().replace("ё", "е") in _LETTERHEAD_LINES


class ChunkMetadata(BaseModel):
    """Structural coordinates of a chunk inside its document."""

    appendix: str = ""  # приложение
    part: str = ""  # часть
    section: str = ""  # раздел
    chapter: str = ""  # глава
    subsection: str = ""  # подраздел
    paragraph: str = ""  # пункт
    subparagraph: str = ""  # подпункт
    heading: str = ""
    is_table: bool = False
    page_start: int | None = None
    page_end: int | None = None

    def breadcrumb(self) -> str:
        """Human-readable path, e.g. ``Приложение № 2 · Раздел III · п. 42``."""
        bits: list[str] = []
        if self.appendix:
            bits.append(f"Приложение {self.appendix}")
        if self.part:
            bits.append(f"Часть {self.part}")
        if self.section:
            bits.append(f"Раздел {self.section}")
        if self.chapter:
            bits.append(f"Глава {self.chapter}")
        if self.subsection and not _is_letterhead(self.subsection):
            bits.append(self.subsection)
        if (
            self.heading
            and self.heading not in bits
            and not _is_letterhead(self.heading)
        ):
            bits.append(self.heading)
        if self.paragraph:
            para = f"п. {self.paragraph}"
            if self.subparagraph:
                para += f", подп. {self.subparagraph}"
            bits.append(para)
        return " · ".join(b.strip() for b in bits if b.strip())


class Chunk(BaseModel):
    """A retrievable, citable fragment of a regulatory document."""

    chunk_id: str
    document_id: str
    document_title: str = ""
    document_short_title: str = ""
    document_number: str = ""
    document_type: DocumentType = DocumentType.UNKNOWN
    source_authority: SourceAuthority = SourceAuthority.EAEU
    source_tier: SourceTier = SourceTier.TIER_1
    status: DocumentStatus = DocumentStatus.UNKNOWN
    version_status: VersionStatus = VersionStatus.REQUIRES_EXPERT_VALIDATION
    adoption_date: date | None = None
    #: Date the source file was fetched — part of the citation's provenance.
    access_date: date | None = None
    source_url: str = ""
    page_url: str = ""
    language: str = "ru"
    topic: list[str] = Field(default_factory=list)
    inn_mentions: list[str] = Field(default_factory=list)
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)
    text: str
    char_count: int = 0
    token_estimate: int = 0

    def breadcrumb(self) -> str:
        return self.metadata.breadcrumb()

    def location_label(self) -> str:
        """``Решение № 85 · Приложение № 1 · п. 42`` — used in the source list."""
        head = self.document_short_title or self.document_title[:80]
        crumb = self.breadcrumb()
        return f"{head} · {crumb}" if crumb else head

    @property
    def is_metadata_card(self) -> bool:
        """True for the registry-card stand-in of a document with no text layer.

        Scanned PDFs (most Expert Committee recommendations) are indexed as a
        single card holding the title, number and URL — see
        ``ingestion.chunker.build_metadata_only_chunk``.  Such a chunk can point
        at a document but can never support a normative statement, so it must
        stay out of the generation context.
        """
        return self.metadata.heading == CARD_ONLY_HEADING

    def to_payload(self) -> dict[str, Any]:
        """Flat dict stored as the Qdrant payload (and in SQLite)."""
        payload = self.model_dump(mode="json")
        payload["breadcrumb"] = self.breadcrumb()
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Chunk":
        data = dict(payload)
        data.pop("breadcrumb", None)
        return cls.model_validate(data)


def make_chunk_id(
    document_id: str,
    metadata: ChunkMetadata,
    text: str,
    ordinal: int,
) -> str:
    """Deterministic, collision-resistant chunk identifier.

    Built from ``document_id`` + structural coordinates + a hash of the
    normalised text, so re-running ingestion on an unchanged document yields
    identical ids (which keeps citations stable across index rebuilds).
    """
    coords = "|".join(
        (
            metadata.appendix,
            metadata.part,
            metadata.section,
            metadata.chapter,
            metadata.subsection,
            metadata.paragraph,
            metadata.subparagraph,
        )
    )
    normalised = _WS_RE.sub(" ", text).strip().lower()
    digest = hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:12]
    coord_slug = re.sub(r"[^0-9A-Za-zА-Яа-яЁё]+", "-", coords).strip("-") or "root"
    return f"{document_id}#{coord_slug}#{ordinal:04d}#{digest}"
