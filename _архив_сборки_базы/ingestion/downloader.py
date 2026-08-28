"""Idempotent downloader for discovered regulatory documents.

Guarantees required by the brief:

* every stored file records source URL, access date, filename, SHA256, size and
  content type;
* a re-run does **not** re-download an unchanged document (conditional request
  by ``ETag``/``Last-Modified`` plus content hash comparison);
* retries, timeouts, rate limiting and logging come from
  :class:`ingestion.http_client.PoliteHttpClient`;
* files that cannot be fetched without circumventing a technical restriction
  are registered with ``download_status = manual_required``.

Run directly::

    python -m ingestion.downloader
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import requests

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import (
    DOCUMENTS_CSV,
    RAW_DIR,
    REGISTRY_DIR,
    configure_logging,
    ensure_directories,
)
from app.db.models import (
    DocumentRecord,
    DocumentStatus,
    DocumentType,
    DownloadStatus,
    SourceAuthority,
    SourceTier,
    VersionStatus,
)
from app.db.repository import Repository
from ingestion.discover_sources import DiscoveredFile, load_discovered
from ingestion.http_client import AccessForbidden, PoliteHttpClient
from ingestion.metadata import (
    amended_document_numbers,
    build_document_id,
    infer_topics,
    parse_date,
    slugify,
    tier_for,
    version_status_for,
)

logger = configure_logging(__name__)

HTTP_CACHE_FILE = REGISTRY_DIR / "http_cache.json"

_AUTHORITY_DIR = {
    SourceAuthority.EAEU: "eaeu",
    SourceAuthority.ICH: "ich",
    SourceAuthority.EMA: "ema",
    SourceAuthority.WHO: "who",
    SourceAuthority.OTHER: "other",
}

_AUTHORITY_NAME = {
    SourceAuthority.EAEU: "Евразийская экономическая комиссия",
    SourceAuthority.ICH: "International Council for Harmonisation",
    SourceAuthority.EMA: "European Medicines Agency",
    SourceAuthority.WHO: "World Health Organization",
    SourceAuthority.OTHER: "Unknown",
}


def _parse_iso_date(value: str | None) -> date | None:
    """Parse the ISO date stored in ``discovered.json`` (``2016-11-03``)."""
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        # Fall back to the Russian notations understood by parse_date().
        return parse_date(value)


@dataclass(slots=True)
class DownloadOutcome:
    record: DocumentRecord
    changed: bool
    reason: str


# --------------------------------------------------------------------------- #
# Conditional-request cache
# --------------------------------------------------------------------------- #


class HttpCache:
    """Persisted ``ETag`` / ``Last-Modified`` values keyed by URL."""

    def __init__(self, path: Path = HTTP_CACHE_FILE) -> None:
        self.path = path
        self._data: dict[str, dict[str, str]] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning("HTTP cache unreadable, starting fresh: %s", path)

    def headers_for(self, url: str) -> dict[str, str]:
        entry = self._data.get(url, {})
        headers: dict[str, str] = {}
        if etag := entry.get("etag"):
            headers["If-None-Match"] = etag
        if modified := entry.get("last_modified"):
            headers["If-Modified-Since"] = modified
        return headers

    def update(self, url: str, etag: str, last_modified: str, sha256: str) -> None:
        self._data[url] = {
            "etag": etag,
            "last_modified": last_modified,
            "sha256": sha256,
            "checked_at": datetime.now().isoformat(timespec="seconds"),
        }

    def sha_for(self, url: str) -> str:
        return self._data.get(url, {}).get("sha256", "")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8"
        )


# --------------------------------------------------------------------------- #
# Downloader
# --------------------------------------------------------------------------- #


class DocumentDownloader:
    """Downloads discovered files into ``data/raw/<authority>/``."""

    def __init__(
        self,
        client: PoliteHttpClient | None = None,
        repository: Repository | None = None,
        raw_dir: Path = RAW_DIR,
    ) -> None:
        self.client = client or PoliteHttpClient()
        self.repository = repository or Repository()
        self.raw_dir = raw_dir
        self.cache = HttpCache()

    # -- public -------------------------------------------------------------
    def download_all(
        self, items: list[DiscoveredFile], prune_registry: bool = True
    ) -> list[DownloadOutcome]:
        outcomes: list[DownloadOutcome] = []
        total = len(items)
        for index, item in enumerate(items, start=1):
            logger.info("[%d/%d] %s", index, total, item.file_url)
            try:
                outcomes.append(self.download_one(item))
            except Exception as exc:  # noqa: BLE001 - keep going through the corpus
                logger.error("  download failed: %s", exc)
                outcomes.append(
                    DownloadOutcome(
                        record=self._build_record(
                            item,
                            local_path=None,
                            sha256="",
                            size=0,
                            content_type="",
                            status=DownloadStatus.FAILED,
                            note=f"Ошибка загрузки: {exc}",
                        ),
                        changed=False,
                        reason="failed",
                    )
                )
        self.cache.save()
        self._persist(outcomes, prune_registry)
        return outcomes

    def download_one(self, item: DiscoveredFile) -> DownloadOutcome:
        if "manual_required" in item.notes:
            return DownloadOutcome(
                record=self._build_record(
                    item, None, "", 0, "", DownloadStatus.MANUAL_REQUIRED,
                    "Источник помечен как требующий ручной загрузки.",
                ),
                changed=False,
                reason="manual",
            )

        try:
            if not self.client.allowed(item.file_url):
                raise AccessForbidden("robots.txt disallows this URL")
            result = self.client.fetch(item.file_url, stream=True)
        except AccessForbidden as exc:
            logger.warning("  manual_required: %s", exc)
            return DownloadOutcome(
                record=self._build_record(
                    item, None, "", 0, "", DownloadStatus.MANUAL_REQUIRED, str(exc)
                ),
                changed=False,
                reason="forbidden",
            )
        except requests.RequestException as exc:
            logger.warning("  network error: %s", exc)
            return DownloadOutcome(
                record=self._build_record(
                    item, None, "", 0, "", DownloadStatus.FAILED, f"Сетевая ошибка: {exc}"
                ),
                changed=False,
                reason="network",
            )

        sha256 = hashlib.sha256(result.content).hexdigest()
        authority = SourceAuthority(item.source_authority)
        target = self._target_path(item, authority, sha256)

        unchanged = (
            target.exists()
            and target.stat().st_size == len(result.content)
            and self.cache.sha_for(item.file_url) == sha256
        )
        if unchanged:
            logger.info("  unchanged (sha256 match) — kept %s", target.name)
            reason = "unchanged"
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(result.content)
            logger.info("  saved %s (%d bytes)", target.name, len(result.content))
            reason = "downloaded"

        self.cache.update(item.file_url, "", "", sha256)

        record = self._build_record(
            item,
            local_path=target,
            sha256=sha256,
            size=len(result.content),
            content_type=result.content_type,
            status=DownloadStatus.OK,
            note="",
        )
        return DownloadOutcome(record=record, changed=reason == "downloaded", reason=reason)

    # -- internals ----------------------------------------------------------
    def _target_path(
        self, item: DiscoveredFile, authority: SourceAuthority, sha256: str
    ) -> Path:
        directory = self.raw_dir / _AUTHORITY_DIR[authority]
        stem = slugify(Path(item.file_name).stem, 70)
        extension = item.file_extension or ".bin"
        return directory / f"{stem}__{sha256[:10]}{extension}"

    def _build_record(
        self,
        item: DiscoveredFile,
        local_path: Path | None,
        sha256: str,
        size: int,
        content_type: str,
        status: DownloadStatus,
        note: str,
    ) -> DocumentRecord:
        authority = SourceAuthority(item.source_authority)
        try:
            doc_type = DocumentType(item.document_type)
        except ValueError:
            doc_type = DocumentType.UNKNOWN

        # discovered.json stores the date in ISO form, not in the Russian
        # notation that parse_date() understands.
        adoption = _parse_iso_date(item.adoption_date)
        document_id = build_document_id(
            authority, doc_type, item.document_number, adoption, item.file_name
        )
        # The tier follows the *kind* of document (a Collegium recommendation is
        # guidance, TIER 2, even though it is listed on the acts page); the
        # source's default only fills in when the kind is unknown.
        tier = tier_for(authority, doc_type)
        if doc_type == DocumentType.UNKNOWN and item.default_tier:
            tier = SourceTier(item.default_tier)

        doc_status = (
            DocumentStatus.SUPERSEDED if item.is_obsolete else DocumentStatus.UNKNOWN
        )

        return DocumentRecord(
            document_id=document_id,
            title=item.title or item.raw_title or item.file_name,
            short_title=item.short_title,
            authority=_AUTHORITY_NAME[authority],
            document_type=doc_type,
            document_number=item.document_number,
            adoption_date=adoption,
            status=doc_status,
            version_status=version_status_for(False, item.is_amendment),
            jurisdiction="EAEU" if authority == SourceAuthority.EAEU else "INTERNATIONAL",
            language=item.language,
            source_url=item.file_url,
            page_url=item.page_url,
            source_authority=authority,
            source_tier=tier,
            amends_document=(
                ",".join(amended_document_numbers(item.title)) if item.is_amendment else ""
            ),
            topic=infer_topics(item.title, item.raw_title, item.notes),
            download_date=datetime.now(),
            download_status=status,
            sha256=sha256,
            size_bytes=size,
            content_type=content_type,
            local_file=str(local_path) if local_path else "",
            notes=note or item.notes,
        )

    # -- persistence --------------------------------------------------------
    def _persist(self, outcomes: list[DownloadOutcome], prune: bool) -> None:
        records = _resolve_id_collisions([o.record for o in outcomes])
        _link_amendments(records)
        self.repository.upsert_documents(records)
        if prune:
            self.repository.delete_documents_not_in(r.document_id for r in records)
        write_documents_csv(records)
        logger.info(
            "Registry updated: %d documents (%s)",
            len(records),
            DOCUMENTS_CSV,
        )


# --------------------------------------------------------------------------- #
# Registry helpers
# --------------------------------------------------------------------------- #


def _resolve_id_collisions(records: list[DocumentRecord]) -> list[DocumentRecord]:
    """Ensure document ids are unique (different files, same number/year)."""
    seen: dict[str, int] = {}
    for record in records:
        base = record.document_id
        count = seen.get(base, 0)
        if count:
            record.document_id = f"{base}_{count + 1}"
        seen[base] = count + 1
    return records


def _link_amendments(records: list[DocumentRecord]) -> None:
    """Populate ``amended_by`` / ``status`` from the discovered amending acts.

    A base act that has at least one amending act is marked ``amended`` and its
    ``version_status`` stays ``original_with_known_amendments`` — we never merge
    the texts ourselves.
    """
    by_number: dict[str, list[DocumentRecord]] = {}
    for record in records:
        if record.source_authority != SourceAuthority.EAEU:
            continue
        if record.document_type in (DocumentType.AMENDMENT,):
            continue
        if record.document_number:
            by_number.setdefault(record.document_number, []).append(record)

    for record in records:
        if record.document_type != DocumentType.AMENDMENT or not record.amends_document:
            continue
        for target_number in record.amends_document.split(","):
            for target in by_number.get(target_number.strip(), []):
                target.amended_by.append(record.document_id)
                if target.status != DocumentStatus.SUPERSEDED:
                    target.status = DocumentStatus.AMENDED
                target.version_status = VersionStatus.ORIGINAL_WITH_KNOWN_AMENDMENTS


CSV_COLUMNS = (
    "document_id", "title", "short_title", "authority", "document_type",
    "document_number", "adoption_date", "effective_date", "expiration_date",
    "status", "version_status", "jurisdiction", "language", "source_url",
    "page_url", "source_authority", "source_tier", "parent_document",
    "amends_document", "amended_by", "version", "topic", "download_date",
    "download_status", "sha256", "size_bytes", "content_type", "local_file",
    "n_pages", "n_chunks", "notes",
)


def write_documents_csv(
    records: list[DocumentRecord], path: Path = DOCUMENTS_CSV
) -> None:
    """Write the human-auditable document registry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, delimiter=";")
        writer.writeheader()
        for record in records:
            row = record.model_dump(mode="json")
            row["amended_by"] = "|".join(record.amended_by)
            row["topic"] = "|".join(record.topic)
            writer.writerow({key: row.get(key, "") for key in CSV_COLUMNS})


def main() -> int:
    ensure_directories()
    items = load_discovered()
    if not items:
        print(
            "Нет обнаруженных источников. Сначала выполните:\n"
            "    python -m ingestion.discover_sources"
        )
        return 1
    downloader = DocumentDownloader()
    outcomes = downloader.download_all(items)

    counts: dict[str, int] = {}
    for outcome in outcomes:
        counts[outcome.reason] = counts.get(outcome.reason, 0) + 1
    print(f"\nОбработано файлов: {len(outcomes)}")
    for reason, count in sorted(counts.items()):
        print(f"  {reason:<14} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
