"""Shared pytest fixtures.

Tests never touch the real knowledge base: every fixture builds a small
synthetic corpus in a temporary directory.  The synthetic documents imitate the
*shape* of EAEU acts (приложения, разделы, пункты) — they are clearly marked as
test fixtures and are never mixed into ``data/``.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.models import (  # noqa: E402
    Chunk,
    ChunkMetadata,
    DocumentRecord,
    DocumentStatus,
    DocumentType,
    SourceAuthority,
    SourceTier,
    VersionStatus,
)
from app.db.repository import Repository  # noqa: E402
from app.rag.citations import CitationRegistry  # noqa: E402
from app.rag.hybrid_search import RetrievedChunk  # noqa: E402

FIXTURE_ACT_TEXT = """\
УТВЕРЖДЕНЫ
Решением Совета Тестовой комиссии
от 3 ноября 2016 г. № 999

ПРАВИЛА
проведения тестовых исследований

I. Общие положения
1. Настоящие Правила устанавливают требования к разработке дизайна тестовых
исследований и являются учебной имитацией структуры нормативного акта.
2. Для целей настоящих Правил используются понятия, означающие следующее.

II. Требования к исследованию
10. Исследование биоэквивалентности проводится в соответствии с общим планом,
утверждённым до начала исследования.
11. Оценка проводится по параметрам AUC(0-t) и Cmax с использованием 90%
доверительного интервала.
а) первый подпункт требований;
б) второй подпункт требований.

ПРИЛОЖЕНИЕ № 1
к Правилам проведения тестовых исследований

III. Условия освобождения от исследования
42. Освобождение от проведения исследования биоэквивалентности (биовейвер)
допускается при выполнении условий, установленных настоящим разделом.
43. Высоковариабельным считается препарат с внутрииндивидуальной
вариабельностью более 30 процентов.
"""


@pytest.fixture
def temp_repository(tmp_path: Path) -> Repository:
    return Repository(tmp_path / "test.sqlite3")


@pytest.fixture
def act_file(tmp_path: Path) -> Path:
    path = tmp_path / "test_act.txt"
    path.write_text(FIXTURE_ACT_TEXT, encoding="utf-8")
    return path


@pytest.fixture
def document() -> DocumentRecord:
    return DocumentRecord(
        document_id="TEST_COUNCIL_DECISION_999_2016",
        title="Решение Совета Тестовой комиссии № 999 от 03.11.2016. О тестовых правилах",
        short_title="Решение Совета ЕЭК № 999 от 03.11.2016",
        document_type=DocumentType.COUNCIL_DECISION,
        document_number="999",
        adoption_date=date(2016, 11, 3),
        status=DocumentStatus.UNKNOWN,
        version_status=VersionStatus.REQUIRES_EXPERT_VALIDATION,
        source_authority=SourceAuthority.EAEU,
        source_tier=SourceTier.TIER_1,
        source_url="https://example.invalid/test_act.txt",
        page_url="https://example.invalid/page",
        topic=["bioequivalence"],
    )


def make_chunk(
    chunk_id: str,
    text: str,
    *,
    document_id: str = "TEST_COUNCIL_DECISION_999_2016",
    number: str = "999",
    tier: SourceTier = SourceTier.TIER_1,
    authority: SourceAuthority = SourceAuthority.EAEU,
    paragraph: str = "42",
    status: DocumentStatus = DocumentStatus.UNKNOWN,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        document_title=f"Тестовый документ {number}",
        document_short_title=f"Решение № {number}",
        document_number=number,
        document_type=DocumentType.COUNCIL_DECISION,
        source_authority=authority,
        source_tier=tier,
        status=status,
        source_url="https://example.invalid/file.pdf",
        metadata=ChunkMetadata(appendix="№ 1", section="III", paragraph=paragraph),
        text=text,
        char_count=len(text),
        token_estimate=max(1, len(text) // 3),
    )


@pytest.fixture
def registry() -> CitationRegistry:
    chunks = [
        make_chunk("c1", "42. Биовейвер допускается при выполнении условий."),
        make_chunk(
            "c2",
            "11. Оценка проводится по AUC(0-t) и Cmax.",
            paragraph="11",
        ),
        make_chunk(
            "c3",
            "Supplementary guidance on biowaiver.",
            document_id="ICH_M9",
            number="M9",
            tier=SourceTier.TIER_4,
            authority=SourceAuthority.ICH,
        ),
    ]
    items = [RetrievedChunk(chunk=c, score=0.05 - 0.01 * i) for i, c in enumerate(chunks)]
    return CitationRegistry.from_retrieval(items)
