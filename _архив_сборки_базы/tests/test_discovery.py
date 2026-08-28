"""Source discovery: URL handling, format preference, listing-page guard."""

from __future__ import annotations

import pytest
from bs4 import BeautifulSoup

from app.db.models import SourceAuthority
from ingestion.discover_sources import (
    MAX_ATTACHMENTS_PER_DOCUMENT_PAGE,
    SourceDiscoverer,
    _absolute,
    _deduplicate,
    _select_preferred_files,
    DiscoveredFile,
    load_sources,
)


# --------------------------------------------------------------------------- #
# URL normalisation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("href", "expected"),
    [
        # A percent-encoded trailing space turns a document URL into the
        # portal's generic listing page — it must be stripped.
        ("https://docs.eaeunion.org/documents/447/9877/%20",
         "https://docs.eaeunion.org/documents/447/9877/"),
        # http:// links to the same host are normalised to https://
        ("http://docs.eaeunion.org/docs/ru-ru/01411942/cncd_21112016_85",
         "https://docs.eaeunion.org/docs/ru-ru/01411942/cncd_21112016_85"),
        # Legitimate encoded spaces inside a file name are preserved.
        ("/upload/files/Rekomendatsiya%2042.pdf",
         "https://docs.eaeunion.org/upload/files/Rekomendatsiya%2042.pdf"),
    ],
)
def test_absolute_url_normalisation(href: str, expected: str):
    assert _absolute("https://docs.eaeunion.org/page", href) == expected


# --------------------------------------------------------------------------- #
# Format preference
# --------------------------------------------------------------------------- #


def test_pdf_is_preferred_over_docx_for_the_same_stem():
    # On docs.eaeunion.org the Russian text is the PDF; the DOCX with the same
    # stem carries another national language.
    chosen = _select_preferred_files(
        [
            "https://x/cncd_21112016_85_doc.docx",
            "https://x/cncd_21112016_85_doc.pdf",
        ]
    )
    assert chosen == ["https://x/cncd_21112016_85_doc.pdf"]


def test_different_stems_are_all_kept():
    urls = [
        "https://x/Reshenie-Soveta-_12.pdf",
        "https://x/Reshenie-Soveta-EEK-_12-ot-22.01.2025.docx",
    ]
    assert sorted(_select_preferred_files(urls)) == sorted(urls)


def test_deduplicate_keeps_first_occurrence():
    items = [
        DiscoveredFile(
            file_url="https://x/a.pdf", page_url="p", source_name="s",
            source_authority="EAEU", document_type="unknown", default_tier="TIER_1",
            raw_title="t", title="t", short_title="t", document_number="",
            adoption_date=None, language="ru", is_amendment=False, is_obsolete=False,
            file_extension=".pdf", file_name="a.pdf",
        )
    ] * 3
    assert len(_deduplicate(items)) == 1


# --------------------------------------------------------------------------- #
# Listing-page guard
# --------------------------------------------------------------------------- #


class _FakeClient:
    def __init__(self, html: str) -> None:
        self.html = html

    def fetch(self, url: str, stream: bool = False):  # noqa: ANN001
        class _Result:
            text = self.html

        return _Result()


def _page_with_attachments(count: int) -> str:
    links = "".join(
        f'<a href="/upload/iblock/x/file_{i}_doc.pdf">файл {i}</a>' for i in range(count)
    )
    return f"<html><head><title>Правовой портал</title></head><body>{links}</body></html>"


def test_listing_page_with_many_attachments_is_skipped():
    discoverer = SourceDiscoverer.__new__(SourceDiscoverer)
    discoverer.client = _FakeClient(
        _page_with_attachments(MAX_ATTACHMENTS_PER_DOCUMENT_PAGE + 20)
    )
    found = discoverer._discover_document_page(
        {"name": "s", "source_authority": "EAEU", "default_tier": "TIER_1"},
        "https://docs.eaeunion.org/documents/447/9877/",
        "",
        False,
    )
    assert found == []


def test_normal_document_page_is_processed():
    discoverer = SourceDiscoverer.__new__(SourceDiscoverer)
    discoverer.client = _FakeClient(
        "<html><head><title>Решение Совета ЕЭК № 85 от 03.11.2016 "
        "Об утверждении Правил</title></head><body>"
        '<a href="/upload/iblock/x/cncd_21112016_85_doc.pdf">ru</a>'
        '<a href="/upload/iblock/y/cncd_21112016_85_doc_arm.docx">hy</a>'
        "</body></html>"
    )
    found = discoverer._discover_document_page(
        {"name": "acts", "source_authority": "EAEU", "default_tier": "TIER_1"},
        "https://docs.eaeunion.org/docs/ru-ru/01411942/cncd_21112016_85",
        "",
        False,
    )
    assert len(found) == 1
    assert found[0].document_number == "85"
    assert found[0].file_extension == ".pdf"


# --------------------------------------------------------------------------- #
# sources.yaml
# --------------------------------------------------------------------------- #


def test_sources_yaml_is_valid():
    sources = load_sources()
    assert sources
    names = {s["name"] for s in sources}
    assert "eaeu_regulatory_acts" in names
    assert "eaeu_expert_committee" in names
    for source in sources:
        assert source["source_authority"] in {a.value for a in SourceAuthority}
        assert source["base_url"].startswith("https://")


def test_per_link_obsolete_annotation_is_detected():
    from ingestion.discover_sources import _obsolete_anchors

    soup = BeautifulSoup(
        "<html><body>"
        '<p><a href="/a">Решение № 85 «Об утверждении Правил»</a></p>'
        '<p><a href="/b">Рекомендация № 30 «О правилах»</a>'
        '&nbsp;<span style="color: #ee1d24;">(утратила силу)</span></p>'
        "</body></html>",
        "lxml",
    )
    obsolete = _obsolete_anchors(soup)
    assert obsolete == {"/b"}


def test_navigation_button_does_not_condemn_the_whole_page():
    """Regression: the EEC page has a nav button labelled «Утратили силу».

    Treating it as the start of an obsolete section marked every act below it —
    including the acts the assistant must cite as being in force.
    """
    from ingestion.discover_sources import _obsolete_anchors

    soup = BeautifulSoup(
        "<html><body>"
        '<p><a href="#5" class="btn btn_border-red">Утратили силу</a></p>'
        '<p><a href="/act-78">Решение № 78</a></p>'
        '<p><a href="/act-85">Решение № 85</a></p>'
        "</body></html>",
        "lxml",
    )
    assert _obsolete_anchors(soup) == set()
