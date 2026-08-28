"""Structure-aware chunking.

Chunks are built from :class:`~ingestion.structure_parser.StructuralUnit`
objects, **never** by slicing every N characters:

* a chunk never spans two приложения / разделы / главы;
* a пункт is the atomic unit — it is split only when it alone exceeds the hard
  size limit, and then on sentence boundaries with overlap;
* every chunk carries the full parent context (приложение, раздел, глава,
  пункт, подпункт, заголовок, страницы), so it is independently citable;
* the heading path is prepended to the chunk text so the embedding sees the
  context, not a bare sentence.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.config import get_settings
from app.db.models import (
    CARD_ONLY_HEADING,
    Chunk,
    ChunkMetadata,
    DocumentRecord,
    make_chunk_id,
)
from app.regulatory.inn import detect_inns
from ingestion.normalizer import estimate_tokens
from ingestion.structure_parser import StructuralUnit, merge_continuations, parse_structure
from ingestion.parser import ParsedDocument

logger = logging.getLogger(__name__)

#: Sentence boundary for Russian legal prose: a period/semicolon followed by a
#: space and a capital letter or a list marker.  Abbreviations such as "п." or
#: "ст." are protected by requiring at least two characters before the dot.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[а-яa-z0-9)][.;])\s+(?=[А-ЯA-Z0-9])")


@dataclass(slots=True)
class ChunkingConfig:
    target_chars: int
    max_chars: int
    overlap_chars: int
    min_chars: int = 120

    @classmethod
    def from_settings(cls) -> "ChunkingConfig":
        settings = get_settings()
        return cls(
            target_chars=settings.chunk_target_chars,
            max_chars=settings.chunk_max_chars,
            overlap_chars=settings.chunk_overlap_chars,
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _context_prefix(document: DocumentRecord, metadata: ChunkMetadata) -> str:
    """Heading path prepended to the chunk body before embedding."""
    parts = [document.short_title or document.title[:100]]
    crumb = metadata.breadcrumb()
    if crumb:
        parts.append(crumb)
    return " / ".join(parts)


def _split_long_unit(text: str, config: ChunkingConfig) -> list[str]:
    """Split an over-long provision on sentence boundaries, with overlap."""
    sentences = _SENTENCE_SPLIT_RE.split(text)
    if len(sentences) == 1:
        # No sentence boundaries (e.g. a wide table): hard-split with overlap.
        step = max(config.target_chars - config.overlap_chars, 400)
        return [
            text[start : start + config.target_chars]
            for start in range(0, len(text), step)
        ]

    pieces: list[str] = []
    buffer = ""
    for sentence in sentences:
        candidate = f"{buffer} {sentence}".strip() if buffer else sentence
        if len(candidate) > config.target_chars and buffer:
            pieces.append(buffer)
            tail = buffer[-config.overlap_chars :] if config.overlap_chars else ""
            buffer = f"{tail} {sentence}".strip()
        else:
            buffer = candidate
    if buffer:
        pieces.append(buffer)
    return pieces


def _merge_metadata(first: ChunkMetadata, last: ChunkMetadata) -> ChunkMetadata:
    """Metadata for a chunk covering several units within one context."""
    merged = first.model_copy(deep=True)
    merged.page_end = last.page_end or first.page_end
    if first.paragraph and last.paragraph and first.paragraph != last.paragraph:
        merged.paragraph = f"{first.paragraph}–{last.paragraph}"
        merged.subparagraph = ""
    return merged


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #


def group_units(
    units: list[StructuralUnit], config: ChunkingConfig
) -> list[tuple[list[StructuralUnit], ChunkMetadata]]:
    """Group units into chunk-sized batches that share a parent context."""
    groups: list[tuple[list[StructuralUnit], ChunkMetadata]] = []
    current: list[StructuralUnit] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            metadata = _merge_metadata(current[0].metadata, current[-1].metadata)
            groups.append((current, metadata))
            current = []
            current_len = 0

    for unit in units:
        unit_len = len(unit.text)

        if current and unit.context_key() != current[0].context_key():
            flush()

        # An over-long provision becomes its own group (split downstream).
        if unit_len > config.max_chars:
            flush()
            groups.append(([unit], unit.metadata))
            continue

        if current_len + unit_len > config.target_chars and current_len >= config.min_chars:
            flush()

        current.append(unit)
        current_len += unit_len + 1

    flush()
    return groups


def build_metadata_only_chunk(document: DocumentRecord, reason: str) -> Chunk:
    """A single citable chunk for a document whose text could not be extracted.

    Scanned documents (Expert Committee recommendations are published as image
    PDFs) would otherwise be invisible to retrieval.  The chunk contains only
    verbatim registry metadata — the official title, number and URL — plus an
    explicit statement that the body text is unavailable, so the assistant can
    point at the document without ever quoting text it has not read.
    """
    metadata = ChunkMetadata(heading=CARD_ONLY_HEADING)
    body = (
        f"Название документа: {document.title}\n"
        f"Тип документа: {document.document_type}\n"
        f"Номер: {document.document_number or 'не указан'}\n"
        f"Издающий орган: {document.authority}\n"
        f"Ссылка на официальный файл: {document.source_url}\n"
        f"ВНИМАНИЕ: текст этого документа не извлечён автоматически ({reason}). "
        f"Содержание документа не проиндексировано; используйте ссылку на "
        f"официальный источник и экспертную проверку."
    )
    text = f"[{_context_prefix(document, metadata)}]\n{body}"
    return Chunk(
        chunk_id=make_chunk_id(document.document_id, metadata, body, 0),
        document_id=document.document_id,
        document_title=document.title,
        document_short_title=document.short_title,
        document_number=document.document_number,
        document_type=document.document_type,
        source_authority=document.source_authority,
        source_tier=document.source_tier,
        status=document.status,
        version_status=document.version_status,
        adoption_date=document.adoption_date,
        access_date=(document.download_date.date() if document.download_date else None),
        source_url=document.source_url,
        page_url=document.page_url,
        language=document.language,
        topic=document.topic,
        inn_mentions=detect_inns(document.title),
        metadata=metadata,
        text=text,
        char_count=len(text),
        token_estimate=estimate_tokens(text),
    )


def build_chunks(
    document: DocumentRecord,
    parsed: ParsedDocument,
    config: ChunkingConfig | None = None,
) -> list[Chunk]:
    """Full pipeline: blocks -> structural units -> context-preserving chunks."""
    config = config or ChunkingConfig.from_settings()
    units = merge_continuations(parse_structure(parsed.blocks))
    if not units:
        logger.warning("No structural units extracted from %s", document.document_id)
        return []

    groups = group_units(units, config)
    chunks: list[Chunk] = []
    ordinal = 0

    for group_units_, metadata in groups:
        body = "\n".join(unit.text for unit in group_units_).strip()
        if len(body) < config.min_chars and len(group_units_) == 1 and group_units_[0].is_heading:
            continue  # a lone heading carries no normative content
        if not body:
            continue

        pieces = (
            _split_long_unit(body, config) if len(body) > config.max_chars else [body]
        )
        for piece in pieces:
            piece = piece.strip()
            if len(piece) < 40:
                continue
            prefix = _context_prefix(document, metadata)
            text = f"[{prefix}]\n{piece}"
            chunk_id = make_chunk_id(document.document_id, metadata, piece, ordinal)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    document_title=document.title,
                    document_short_title=document.short_title,
                    document_number=document.document_number,
                    document_type=document.document_type,
                    source_authority=document.source_authority,
                    source_tier=document.source_tier,
                    status=document.status,
                    version_status=document.version_status,
                    adoption_date=document.adoption_date,
                    access_date=(
                        document.download_date.date() if document.download_date else None
                    ),
                    source_url=document.source_url,
                    page_url=document.page_url,
                    language=document.language,
                    topic=document.topic,
                    inn_mentions=detect_inns(f"{document.title} {piece}"),
                    metadata=metadata.model_copy(deep=True),
                    text=text,
                    char_count=len(text),
                    token_estimate=estimate_tokens(text),
                )
            )
            ordinal += 1

    logger.info(
        "%s: %d units -> %d chunks", document.document_id, len(units), len(chunks)
    )
    return chunks
