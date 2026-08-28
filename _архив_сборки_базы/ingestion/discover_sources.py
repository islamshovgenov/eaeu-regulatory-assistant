"""Discovery of downloadable regulatory documents from official sources.

Reads ``data/registry/sources.yaml``, crawls each enabled source and produces
``data/registry/discovered.json`` — a list of :class:`DiscoveredFile` entries.
No file is downloaded here; discovery is deliberately separate from download so
the candidate list can be reviewed before any bytes are fetched.

Run directly::

    python -m ingestion.discover_sources
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlparse

import yaml
from bs4 import BeautifulSoup

if __package__ in (None, ""):  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DISCOVERED_JSON, SOURCES_YAML, configure_logging, ensure_directories
from app.db.models import DocumentType, SourceAuthority
from ingestion.http_client import AccessForbidden, PoliteHttpClient
from ingestion.metadata import (
    date_from_slug,
    file_looks_russian,
    looks_like_amendment,
    parse_document_title,
)

logger = configure_logging(__name__)

DOCUMENT_EXTENSIONS = (".pdf", ".docx", ".doc", ".rtf", ".txt", ".html", ".htm")

#: Per-link annotation the EEC puts next to an act that is no longer in force,
#: e.g. ``… № 30 "О правилах …" (утратила силу)``.
_OBSOLETE_MARKER_RE = re.compile(r"\(\s*утратил[аио]?\w*\s+силу\s*\)", re.IGNORECASE)

#: An act is published in five official languages, sometimes with a separate
#: appendix file, so a genuine document page carries at most ~12 attachments.
MAX_ATTACHMENTS_PER_DOCUMENT_PAGE = 12


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class DiscoveredFile:
    """One downloadable file plus everything we know about it before download."""

    file_url: str
    page_url: str
    source_name: str
    source_authority: str
    document_type: str
    default_tier: str
    raw_title: str
    title: str
    short_title: str
    document_number: str
    adoption_date: str | None
    language: str
    is_amendment: bool
    is_obsolete: bool
    file_extension: str
    file_name: str
    notes: str = ""
    referenced_numbers: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _absolute(base: str, href: str) -> str:
    # Some EEC anchors carry a percent-encoded trailing space, which turns a
    # document URL into a request for the portal's generic listing page.
    cleaned = href.strip().replace("%20", " ").strip().replace(" ", "%20")
    while cleaned.endswith("%20"):
        cleaned = cleaned[:-3]
    url = urljoin(base, cleaned)
    # EEC pages mix http:// and https:// links to the same host.
    return url.replace("http://docs.eaeunion.org", "https://docs.eaeunion.org")


def _extension_of(url: str) -> str:
    path = unquote(urlparse(url).path)
    suffix = Path(path).suffix.lower()
    return suffix


def _file_name_of(url: str) -> str:
    return unquote(urlparse(url).path).rsplit("/", 1)[-1] or "document"


def _page_title(soup: BeautifulSoup) -> str:
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    return h1.get_text(" ", strip=True) if h1 else ""


def _obsolete_anchors(soup: BeautifulSoup) -> set[str]:
    """Hrefs explicitly annotated as no longer in force on an EEC index page.

    The EEC marks an individual act by appending a red ``(утратила силу)`` note
    immediately after its link.  Only that per-link annotation is trusted: the
    page also has a navigation button labelled «Утратили силу» near the top, and
    treating *that* as the start of an obsolete section would wrongly condemn
    every act listed below it.
    """
    obsolete: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        following: list[str] = []
        for element in anchor.next_elements:
            if getattr(element, "name", None) == "a":
                break  # reached the next document link
            if isinstance(element, str):
                following.append(element)
                if sum(len(part) for part in following) > 200:
                    break
        if _OBSOLETE_MARKER_RE.search(" ".join(following)):
            obsolete.add(anchor["href"])
    return obsolete


# --------------------------------------------------------------------------- #
# Discovery strategies
# --------------------------------------------------------------------------- #


class SourceDiscoverer:
    """Crawls the configured sources and yields candidate files."""

    def __init__(self, client: PoliteHttpClient | None = None) -> None:
        self.client = client or PoliteHttpClient()

    # -- entry point --------------------------------------------------------
    def discover_all(self, sources: list[dict[str, Any]]) -> list[DiscoveredFile]:
        results: list[DiscoveredFile] = []
        for source in sources:
            if not source.get("enabled", True):
                logger.info("Source %s disabled — skipped", source.get("name"))
                continue
            method = source.get("discovery_method", "static_list")
            logger.info("Discovering source %s (%s)", source.get("name"), method)
            try:
                if method == "eec_acts_index":
                    results.extend(self._discover_acts_index(source))
                elif method == "eec_file_index":
                    results.extend(self._discover_file_index(source))
                elif method == "eaeu_doc_page":
                    results.extend(
                        self._discover_document_page(source, source["base_url"], "", False)
                    )
                elif method == "static_list":
                    results.extend(self._discover_static(source))
                elif method == "manual":
                    results.extend(self._discover_manual(source))
                else:
                    logger.warning("Unknown discovery_method %r — skipped", method)
            except Exception as exc:  # noqa: BLE001 - one bad source must not stop all
                logger.error("Discovery failed for %s: %s", source.get("name"), exc)
        return _deduplicate(results)

    # -- strategy: EEC index -> docs.eaeunion.org document pages ------------
    def _discover_acts_index(self, source: dict[str, Any]) -> list[DiscoveredFile]:
        base_url = source["base_url"]
        soup = self._soup(base_url)
        obsolete_hrefs = _obsolete_anchors(soup)
        link_filter = source.get("link_filter", "docs.eaeunion.org")

        doc_pages: list[tuple[str, str, bool]] = []  # (url, anchor text, obsolete)
        seen: set[str] = set()
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            if not re.search(link_filter, href):
                continue
            url = _absolute(base_url, href)
            if url in seen:
                continue
            seen.add(url)
            doc_pages.append(
                (url, anchor.get_text(" ", strip=True), href in obsolete_hrefs)
            )

        logger.info("  %d document pages linked from %s", len(doc_pages), base_url)

        results: list[DiscoveredFile] = []
        for index, (url, anchor_text, obsolete) in enumerate(doc_pages, start=1):
            logger.debug("  [%d/%d] %s", index, len(doc_pages), url)
            try:
                results.extend(
                    self._discover_document_page(source, url, anchor_text, obsolete)
                )
            except AccessForbidden as exc:
                logger.warning("  forbidden: %s", exc)
            except Exception as exc:  # noqa: BLE001
                logger.warning("  failed to read %s: %s", url, exc)
        return results

    def _discover_document_page(
        self,
        source: dict[str, Any],
        page_url: str,
        anchor_text: str,
        obsolete: bool,
    ) -> list[DiscoveredFile]:
        soup = self._soup(page_url)
        raw_title = _page_title(soup) or anchor_text
        parsed = parse_document_title(raw_title)
        adoption = parsed.adoption_date or date_from_slug(page_url)

        candidates: list[str] = []
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            if "/upload/" not in href:
                continue
            url = _absolute(page_url, href)
            if _extension_of(url) not in DOCUMENT_EXTENSIONS:
                continue
            candidates.append(url)

        # A document page publishes one act in up to five languages, i.e. a
        # handful of files.  A much larger number means we landed on a listing
        # page (the portal falls back to one for malformed URLs), whose files
        # belong to unrelated acts and must not inherit this page's metadata.
        if len(candidates) > MAX_ATTACHMENTS_PER_DOCUMENT_PAGE:
            logger.warning(
                "  %s looks like a listing page (%d attachments) — skipped",
                page_url,
                len(candidates),
            )
            return []

        russian = [u for u in dict.fromkeys(candidates) if file_looks_russian(u)]
        if not russian:
            logger.debug("  no Russian attachment found on %s", page_url)
            return []

        chosen = _select_preferred_files(russian)
        is_amendment = looks_like_amendment(parsed.title)

        return [
            DiscoveredFile(
                file_url=url,
                page_url=page_url,
                source_name=source["name"],
                source_authority=source.get("source_authority", "EAEU"),
                document_type=(
                    str(DocumentType.AMENDMENT)
                    if is_amendment
                    else str(parsed.document_type)
                ),
                default_tier=source.get("default_tier", "TIER_1"),
                raw_title=raw_title,
                title=parsed.title,
                short_title=parsed.short_title,
                document_number=parsed.number,
                adoption_date=adoption.isoformat() if adoption else None,
                language=source.get("language", "ru"),
                is_amendment=is_amendment,
                is_obsolete=obsolete,
                file_extension=_extension_of(url),
                file_name=_file_name_of(url),
                notes=source.get("notes", "").strip(),
            )
            for url in chosen
        ]

    # -- strategy: EEC page linking files directly --------------------------
    def _discover_file_index(self, source: dict[str, Any]) -> list[DiscoveredFile]:
        base_url = source["base_url"]
        soup = self._soup(base_url)
        pattern = re.compile(source.get("link_filter", r"\.pdf$"), re.IGNORECASE)
        doc_type = source.get("document_type", str(DocumentType.GUIDELINE))

        results: list[DiscoveredFile] = []
        seen: set[str] = set()
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            url = _absolute(base_url, href)
            if _extension_of(url) not in DOCUMENT_EXTENSIONS:
                continue
            if not pattern.search(unquote(urlparse(url).path)):
                continue
            if url in seen:
                continue
            seen.add(url)

            anchor_text = anchor.get_text(" ", strip=True)
            file_name = _file_name_of(url)
            title = anchor_text or _title_from_filename(file_name)
            number = _number_from_filename(file_name)
            results.append(
                DiscoveredFile(
                    file_url=url,
                    page_url=base_url,
                    source_name=source["name"],
                    source_authority=source.get("source_authority", "EAEU"),
                    document_type=doc_type,
                    default_tier=source.get("default_tier", "TIER_3"),
                    raw_title=title,
                    title=title,
                    short_title=title[:120],
                    document_number=number,
                    adoption_date=None,
                    language=source.get("language", "ru"),
                    is_amendment=False,
                    is_obsolete=False,
                    file_extension=_extension_of(url),
                    file_name=file_name,
                    notes=source.get("notes", "").strip(),
                )
            )
        logger.info("  %d files discovered on %s", len(results), base_url)
        return results

    # -- strategy: explicit URL list ----------------------------------------
    def _discover_static(self, source: dict[str, Any]) -> list[DiscoveredFile]:
        results: list[DiscoveredFile] = []
        for url in source.get("urls", []):
            file_name = _file_name_of(url)
            title = _title_from_filename(file_name)
            results.append(
                DiscoveredFile(
                    file_url=url,
                    page_url=source.get("base_url", url),
                    source_name=source["name"],
                    source_authority=source.get("source_authority", "OTHER"),
                    document_type=source.get("document_type", str(DocumentType.SUPPLEMENTARY)),
                    default_tier=source.get("default_tier", "TIER_4"),
                    raw_title=title,
                    title=title,
                    short_title=title[:120],
                    document_number="",
                    adoption_date=None,
                    language=source.get("language", "en"),
                    is_amendment=False,
                    is_obsolete=False,
                    file_extension=_extension_of(url) or ".pdf",
                    file_name=file_name,
                    notes=source.get("notes", "").strip(),
                )
            )
        return results

    # -- strategy: manual ---------------------------------------------------
    def _discover_manual(self, source: dict[str, Any]) -> list[DiscoveredFile]:
        results = self._discover_static(source)
        for item in results:
            item.notes = (
                (item.notes + " ").strip()
                + " [manual_required: файл не загружается автоматически]"
            ).strip()
        return results

    # -- io -----------------------------------------------------------------
    def _soup(self, url: str) -> BeautifulSoup:
        result = self.client.fetch(url)
        return BeautifulSoup(result.text, "lxml")


# --------------------------------------------------------------------------- #
# Post-processing
# --------------------------------------------------------------------------- #


#: Preference order when a document page offers the same act in several formats.
#: PDF comes first for a concrete, verified reason: on docs.eaeunion.org the
#: five official language versions share one file stem, and the *Russian* text
#: is published as ``<stem>.pdf`` while ``<stem>.docx`` carries one of the other
#: national languages (Armenian, Kazakh, Kyrgyz, Belarusian).  Preferring DOCX
#: silently pulls a non-Russian corpus.  Language is verified again from the
#: file contents at parse time (:func:`ingestion.normalizer.is_russian_text`).
#:
#: KNOWN GAP: on some Collegium pages ``<stem>_doc.pdf`` is the *English*
#: translation and the Russian text carries a different file name, so those acts
#: are dropped by the language check and never reach the index.  See
#: docs/limitations.md § 2.1 for the affected documents and the fix.
_FORMAT_PREFERENCE = (".pdf", ".docx", ".doc", ".rtf", ".txt", ".htm", ".html")


def _select_preferred_files(urls: list[str]) -> list[str]:
    """Pick one file per logical document, using :data:`_FORMAT_PREFERENCE`."""
    by_stem: dict[str, dict[str, str]] = {}
    for url in urls:
        stem = Path(unquote(urlparse(url).path)).stem.lower()
        by_stem.setdefault(stem, {})[_extension_of(url)] = url

    chosen: list[str] = []
    for formats in by_stem.values():
        for extension in _FORMAT_PREFERENCE:
            if extension in formats:
                chosen.append(formats[extension])
                break
    return chosen


def _deduplicate(items: list[DiscoveredFile]) -> list[DiscoveredFile]:
    seen: set[str] = set()
    unique: list[DiscoveredFile] = []
    for item in items:
        if item.file_url in seen:
            continue
        seen.add(item.file_url)
        unique.append(item)
    return unique


_FILENAME_NUMBER_RE = re.compile(r"(\d{1,3})(?:\D*)$")


def _number_from_filename(file_name: str) -> str:
    stem = Path(unquote(file_name)).stem
    match = _FILENAME_NUMBER_RE.search(re.sub(r"[_\-]", " ", stem))
    return match.group(1) if match else ""


def _title_from_filename(file_name: str) -> str:
    stem = Path(unquote(file_name)).stem
    return re.sub(r"[_\-]+", " ", stem).strip()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def load_sources(path: Path | None = None) -> list[dict[str, Any]]:
    """Read and validate ``sources.yaml``."""
    path = path or SOURCES_YAML
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    sources = data.get("sources", [])
    if not isinstance(sources, list):
        raise ValueError(f"{path}: 'sources' must be a list")
    for source in sources:
        for required in ("name", "authority", "base_url", "discovery_method"):
            if required not in source:
                raise ValueError(f"{path}: source missing required key {required!r}")
        authority = source.get("source_authority", "EAEU")
        if authority not in {a.value for a in SourceAuthority}:
            raise ValueError(f"{path}: unknown source_authority {authority!r}")
    return sources


def discover(
    sources_path: Path | None = None,
    output_path: Path | None = None,
    client: PoliteHttpClient | None = None,
) -> list[DiscoveredFile]:
    """Crawl every enabled source and persist the candidate list."""
    ensure_directories()
    sources = load_sources(sources_path)
    discoverer = SourceDiscoverer(client)
    found = discoverer.discover_all(sources)

    output = output_path or DISCOVERED_JSON
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump([f.to_json() for f in found], handle, ensure_ascii=False, indent=2)

    logger.info("Discovered %d candidate files -> %s", len(found), output)
    return found


def load_discovered(path: Path | None = None) -> list[DiscoveredFile]:
    path = path or DISCOVERED_JSON
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return [DiscoveredFile(**item) for item in raw]


def main() -> int:
    found = discover()
    by_source: dict[str, int] = {}
    for item in found:
        by_source[item.source_name] = by_source.get(item.source_name, 0) + 1
    print(f"\nDiscovered {len(found)} candidate files:")
    for name, count in sorted(by_source.items()):
        print(f"  {name:<32} {count:>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
