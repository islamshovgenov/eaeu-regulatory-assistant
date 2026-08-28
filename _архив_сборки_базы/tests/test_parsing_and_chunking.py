"""Parsing, normalisation and structure-aware chunking."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.db.models import DocumentRecord
from ingestion.chunker import ChunkingConfig, build_chunks, build_metadata_only_chunk
from ingestion.normalizer import (
    detect_script,
    estimate_tokens,
    is_russian_text,
    normalize_for_search,
    normalize_text,
)
from ingestion.parser import UnsupportedFormat, parse_file
from ingestion.structure_parser import (
    merge_continuations,
    parse_structure,
    starts_new_structural_unit,
)


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #


def test_normalize_collapses_unicode_spaces_and_hyphenation():
    raw = "лекарствен-\nных  препаратов  ,   раздел"
    assert normalize_text(raw) == "лекарственных препаратов, раздел"


def test_normalize_drops_bare_page_numbers():
    assert "\n12\n" not in normalize_text("текст\n12\nпродолжение")


def test_normalize_for_search_folds_yo_and_quotes():
    assert normalize_for_search('«Всё» ЛЕКАРСТВО') == "все лекарство"


def test_estimate_tokens_is_monotonic():
    assert estimate_tokens("") == 0
    assert estimate_tokens("а" * 26) > estimate_tokens("а" * 13)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Настоящие Правила устанавливают требования к исследованию" * 5, True),
        ("ԵՎՐԱՍԻԱԿԱՆ ՏՆՏԵՍԱԿԱՆ ՀԱՆՁՆԱԺՈՂՈՎ ԽՈՐՀՈՒՐԴ ՈՐՈՇՈՒՄ" * 5, False),
        ("The guideline describes bioequivalence requirements" * 5, False),
        ("Осы Қағидалар дәрілік заттарға қойылатын талаптарды белгілейді" * 5, False),
    ],
)
def test_is_russian_text_distinguishes_eaeu_languages(text: str, expected: bool):
    assert is_russian_text(text) is expected


def test_detect_script():
    assert detect_script("Настоящие Правила устанавливают требования" * 3) == "cyrillic"
    assert detect_script("short") == "unknown"


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def test_parse_txt_produces_blocks(act_file: Path):
    parsed = parse_file(act_file)
    assert parsed.parser_used == "plaintext"
    assert parsed.blocks
    assert "биоэквивалентности" in parsed.normalized_text
    assert parsed.raw_text  # original text is preserved separately


def test_parse_does_not_modify_source_file(act_file: Path):
    before = act_file.read_bytes()
    parse_file(act_file)
    assert act_file.read_bytes() == before


def test_unsupported_format_raises(tmp_path: Path):
    path = tmp_path / "file.xyz"
    path.write_text("data", encoding="utf-8")
    with pytest.raises(UnsupportedFormat):
        parse_file(path)


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "line",
    [
        "ПРИЛОЖЕНИЕ № 1",
        "I. Общие положения",
        "Глава 2. Требования",
        "42. Освобождение допускается",
        "а) первый подпункт",
    ],
)
def test_structural_markers_are_recognised(line: str):
    assert starts_new_structural_unit(line)


def test_structure_carries_parent_context(act_file: Path):
    parsed = parse_file(act_file)
    units = merge_continuations(parse_structure(parsed.blocks))

    paragraph_42 = [u for u in units if u.metadata.paragraph == "42"]
    assert paragraph_42, "пункт 42 не распознан"
    unit = paragraph_42[0]
    assert unit.metadata.appendix == "№ 1"
    assert unit.metadata.section == "III"
    assert "Приложение № 1" in unit.metadata.breadcrumb()
    assert "п. 42" in unit.metadata.breadcrumb()


def test_subparagraph_is_tracked(act_file: Path):
    parsed = parse_file(act_file)
    units = parse_structure(parsed.blocks)
    subparagraphs = [u for u in units if u.metadata.subparagraph]
    assert subparagraphs
    assert subparagraphs[0].metadata.paragraph == "11"


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #


def test_chunks_are_built_with_metadata(act_file: Path, document: DocumentRecord):
    parsed = parse_file(act_file)
    chunks = build_chunks(document, parsed, ChunkingConfig.from_settings())

    assert chunks
    for chunk in chunks:
        assert chunk.document_id == document.document_id
        assert chunk.text.startswith("[")  # heading path prefix
        assert chunk.char_count == len(chunk.text)
        assert chunk.source_url == document.source_url


def test_chunk_ids_are_stable_and_unique(act_file: Path, document: DocumentRecord):
    parsed = parse_file(act_file)
    first = build_chunks(document, parsed, ChunkingConfig.from_settings())
    second = build_chunks(document, parse_file(act_file), ChunkingConfig.from_settings())

    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first)


def test_chunk_never_spans_two_appendices(act_file: Path, document: DocumentRecord):
    parsed = parse_file(act_file)
    chunks = build_chunks(document, parsed, ChunkingConfig.from_settings())
    for chunk in chunks:
        # A chunk carries exactly one appendix value; the grouping key forbids
        # merging units from different appendices.
        assert chunk.metadata.appendix in ("", "№ 1")


def test_long_paragraph_is_split_with_overlap(document: DocumentRecord):
    from ingestion.parser import ParsedDocument, TextBlock

    sentence = "Настоящее положение устанавливает требования к исследованию. "
    long_text = "5. " + sentence * 120
    parsed = ParsedDocument(
        path=Path("synthetic.txt"),
        blocks=[TextBlock(text=long_text)],
        normalized_text=long_text,
    )
    config = ChunkingConfig(target_chars=800, max_chars=1200, overlap_chars=100)
    chunks = build_chunks(document, parsed, config)

    assert len(chunks) > 1
    assert all(c.metadata.paragraph == "5" for c in chunks)


def test_metadata_only_chunk_declares_missing_text(document: DocumentRecord):
    chunk = build_metadata_only_chunk(document, "отсканированный документ")
    assert "не извлечён автоматически" in chunk.text
    assert document.title in chunk.text
    assert chunk.source_url == document.source_url
