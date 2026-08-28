"""Metadata extraction, deduplication and the SQLite registry."""

from __future__ import annotations

from datetime import date

import pytest

from app.db.models import (
    DocumentRecord,
    DocumentStatus,
    DocumentType,
    SourceAuthority,
    VersionStatus,
    make_chunk_id,
)
from app.db.repository import Repository
from ingestion.downloader import _link_amendments, _resolve_id_collisions
from ingestion.metadata import (
    amended_document_numbers,
    build_document_id,
    date_from_slug,
    file_looks_russian,
    infer_topics,
    looks_like_amendment,
    parse_date,
    parse_document_title,
)
from tests.conftest import make_chunk


# --------------------------------------------------------------------------- #
# Titles and dates
# --------------------------------------------------------------------------- #


def test_parse_council_decision_title():
    parsed = parse_document_title(
        "Решение Совета ЕЭК № 85 от 03.11.2016 Об утверждении Правил проведения "
        "исследований биоэквивалентности лекарственных препаратов"
    )
    assert parsed.document_type == DocumentType.COUNCIL_DECISION
    assert parsed.number == "85"
    assert parsed.adoption_date == date(2016, 11, 3)
    assert parsed.short_title == "Решение Совета ЕЭК № 85 от 03.11.2016"
    assert "биоэквивалентности" in parsed.title


def test_parse_college_recommendation_title():
    parsed = parse_document_title(
        "Рекомендация Коллегии ЕЭК № 2 от 16.01.2018 О Руководстве по препаратам "
        "с модифицированным высвобождением"
    )
    assert parsed.document_type == DocumentType.COLLEGE_RECOMMENDATION
    assert parsed.number == "2"


def test_unparseable_title_stays_unknown_and_is_not_invented():
    parsed = parse_document_title("Какой-то документ без реквизитов")
    assert parsed.document_type == DocumentType.UNKNOWN
    assert parsed.number == ""


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("от 03.11.2016", date(2016, 11, 3)),
        ("от 3 ноября 2016 г.", date(2016, 11, 3)),
        ("без даты", None),
        ("32.13.2016", None),
    ],
)
def test_parse_date(text: str, expected: date | None):
    assert parse_date(text) == expected


def test_date_from_slug():
    assert date_from_slug("/docs/ru-ru/01444549/err_14052024_30") == date(2024, 5, 14)


# --------------------------------------------------------------------------- #
# Amendments and language filtering
# --------------------------------------------------------------------------- #


def test_amendment_detection_and_target():
    title = "Решение Совета ЕЭК № 30 от 12.04.2024 О внесении изменений в Решение № 85"
    assert looks_like_amendment(title)
    assert amended_document_numbers(title) == ["85"]


def test_non_amendment_is_not_flagged():
    assert not looks_like_amendment("Об утверждении Правил регистрации")


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("/upload/x/cncd_21112016_85_doc.pdf", True),
        ("/upload/x/Reshenie-Soveta-_12.docx", True),
        ("/upload/x/ETHX_voroshum_N12_2025_arm.docx", False),
        ("/upload/x/Rashenne-Saveta-_-12-ad-22-studzenya.docx", False),
    ],
)
def test_file_language_filter(url: str, expected: bool):
    assert file_looks_russian(url) is expected


def test_infer_topics():
    topics = infer_topics("Правила проведения исследований биоэквивалентности")
    assert "bioequivalence" in topics
    assert infer_topics("") == []


# --------------------------------------------------------------------------- #
# Identifiers and deduplication
# --------------------------------------------------------------------------- #


def test_iso_date_from_discovery_is_parsed():
    """Regression: dates are stored in discovered.json in ISO form."""
    from ingestion.downloader import _parse_iso_date

    assert _parse_iso_date("2016-11-03") == date(2016, 11, 3)
    assert _parse_iso_date("от 3 ноября 2016 г.") == date(2016, 11, 3)
    assert _parse_iso_date("") is None
    assert _parse_iso_date(None) is None


@pytest.mark.parametrize(
    ("authority", "doc_type", "expected"),
    [
        (SourceAuthority.EAEU, DocumentType.COUNCIL_DECISION, "TIER_1"),
        (SourceAuthority.EAEU, DocumentType.COLLEGE_DECISION, "TIER_1"),
        (SourceAuthority.EAEU, DocumentType.AMENDMENT, "TIER_1"),
        (SourceAuthority.EAEU, DocumentType.COLLEGE_RECOMMENDATION, "TIER_2"),
        (SourceAuthority.EAEU, DocumentType.COUNCIL_RECOMMENDATION, "TIER_2"),
        (SourceAuthority.EAEU, DocumentType.EXPERT_COMMITTEE_RECOMMENDATION, "TIER_3"),
        (SourceAuthority.ICH, DocumentType.SUPPLEMENTARY, "TIER_4"),
        (SourceAuthority.EMA, DocumentType.SUPPLEMENTARY, "TIER_4"),
    ],
)
def test_tier_follows_document_kind(authority, doc_type, expected):
    from ingestion.metadata import tier_for

    assert str(tier_for(authority, doc_type)) == expected


def test_build_document_id_is_deterministic():
    first = build_document_id(
        SourceAuthority.EAEU, DocumentType.COUNCIL_DECISION, "85", date(2016, 11, 3), "x"
    )
    second = build_document_id(
        SourceAuthority.EAEU, DocumentType.COUNCIL_DECISION, "85", date(2016, 11, 3), "y"
    )
    assert first == second == "EAEU_COUNCIL_DECISION_85_2016"


def test_duplicate_document_ids_are_disambiguated():
    records = [
        DocumentRecord(document_id="EAEU_X", title="A"),
        DocumentRecord(document_id="EAEU_X", title="B"),
        DocumentRecord(document_id="EAEU_X", title="C"),
    ]
    resolved = _resolve_id_collisions(records)
    assert [r.document_id for r in resolved] == ["EAEU_X", "EAEU_X_2", "EAEU_X_3"]


def test_amendment_linking_marks_base_act_as_amended():
    base = DocumentRecord(
        document_id="EAEU_COUNCIL_DECISION_85_2016",
        title="Правила биоэквивалентности",
        document_type=DocumentType.COUNCIL_DECISION,
        document_number="85",
    )
    amendment = DocumentRecord(
        document_id="EAEU_AMENDMENT_30_2024",
        title="О внесении изменений в Решение № 85",
        document_type=DocumentType.AMENDMENT,
        document_number="30",
        amends_document="85",
    )
    _link_amendments([base, amendment])

    assert base.status == DocumentStatus.AMENDED
    assert amendment.document_id in base.amended_by
    assert base.version_status == VersionStatus.ORIGINAL_WITH_KNOWN_AMENDMENTS


def test_chunk_id_changes_with_text_but_not_with_reruns():
    from app.db.models import ChunkMetadata

    metadata = ChunkMetadata(appendix="№ 1", section="III", paragraph="42")
    first = make_chunk_id("DOC", metadata, "текст пункта", 0)
    same = make_chunk_id("DOC", metadata, "текст   пункта", 0)  # whitespace-insensitive
    different = make_chunk_id("DOC", metadata, "иной текст", 0)

    assert first == same
    assert first != different


# --------------------------------------------------------------------------- #
# Repository
# --------------------------------------------------------------------------- #


def test_repository_roundtrip(temp_repository: Repository, document: DocumentRecord):
    temp_repository.upsert_document(document)
    loaded = temp_repository.get_document(document.document_id)

    assert loaded is not None
    assert loaded.title == document.title
    assert loaded.adoption_date == document.adoption_date
    assert loaded.source_tier == document.source_tier


def test_repository_replace_chunks_updates_counts(
    temp_repository: Repository, document: DocumentRecord
):
    temp_repository.upsert_document(document)
    chunks = [make_chunk(f"c{i}", f"текст {i}") for i in range(3)]
    for chunk in chunks:
        chunk.document_id = document.document_id

    assert temp_repository.replace_chunks(document.document_id, chunks) == 3
    assert temp_repository.count_chunks() == 3
    reloaded = temp_repository.get_document(document.document_id)
    assert reloaded is not None and reloaded.n_chunks == 3

    temp_repository.replace_chunks(document.document_id, chunks[:1])
    assert temp_repository.count_chunks() == 1


def test_repository_detects_duplicate_by_sha(temp_repository: Repository):
    first = DocumentRecord(document_id="A", title="A", sha256="deadbeef")
    temp_repository.upsert_document(first)
    assert temp_repository.get_document_by_sha256("deadbeef") is not None
    assert temp_repository.get_document_by_sha256("nope") is None


def test_prune_removes_documents_no_longer_discovered(
    temp_repository: Repository, document: DocumentRecord
):
    other = DocumentRecord(document_id="STALE", title="Из листинговой страницы")
    temp_repository.upsert_documents([document, other])
    temp_repository.replace_chunks("STALE", [make_chunk("s1", "текст", document_id="STALE")])

    removed = temp_repository.delete_documents_not_in([document.document_id])

    assert removed == ["STALE"]
    assert temp_repository.get_document("STALE") is None
    assert temp_repository.get_document(document.document_id) is not None
    assert temp_repository.count_chunks() == 0


def test_repository_statistics(temp_repository: Repository, document: DocumentRecord):
    temp_repository.upsert_document(document)
    assert temp_repository.counts_by("source_authority") == {"EAEU": 1}
    assert temp_repository.counts_by_year() == {"2016": 1}
    with pytest.raises(ValueError):
        temp_repository.counts_by("title; DROP TABLE documents")
