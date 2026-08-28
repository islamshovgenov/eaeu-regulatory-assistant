"""Document parsing: PDF, DOCX, HTML and plain text -> ordered text blocks.

Design rules from the brief:

* PDF is read with the **text layer first** (PyMuPDF); OCR is an *optional*
  dependency used only when a PDF has no usable text layer;
* the original file is never modified;
* both ``raw_text`` and ``normalized_text`` are produced (normalisation lives
  in :mod:`ingestion.normalizer`);
* every block keeps its page number when the format provides one, so citations
  can point at a page.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from ingestion.normalizer import normalize_text

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".html", ".htm", ".txt", ".md"}

#: Below this many characters per page a PDF is considered to have no usable
#: text layer (i.e. it is a scan) and OCR is offered.
MIN_CHARS_PER_PAGE_FOR_TEXT_LAYER = 80


class UnsupportedFormat(RuntimeError):
    """The file extension has no parser."""


class OcrUnavailable(RuntimeError):
    """A scanned PDF was found but the optional OCR extras are not installed."""


@dataclass(slots=True)
class TextBlock:
    """One paragraph / table row / heading, in reading order."""

    text: str
    page: int | None = None
    is_table: bool = False
    style: str = ""


@dataclass(slots=True)
class ParsedDocument:
    """Result of parsing one file."""

    path: Path
    blocks: list[TextBlock] = field(default_factory=list)
    n_pages: int | None = None
    raw_text: str = ""
    normalized_text: str = ""
    parser_used: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return len(self.normalized_text)


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #


def _parse_pdf(path: Path, allow_ocr: bool) -> ParsedDocument:
    import pymupdf

    document = pymupdf.open(path)
    blocks: list[TextBlock] = []
    raw_parts: list[str] = []
    try:
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            page_text = page.get_text("text") or ""
            raw_parts.append(page_text)
            for paragraph in _split_pdf_page(page_text):
                blocks.append(TextBlock(text=paragraph, page=page_index + 1))
        n_pages = document.page_count
    finally:
        document.close()

    raw_text = "\n".join(raw_parts)
    parsed = ParsedDocument(
        path=path,
        blocks=blocks,
        n_pages=n_pages,
        raw_text=raw_text,
        parser_used="pymupdf",
    )

    density = len(raw_text.strip()) / max(n_pages, 1)
    if density < MIN_CHARS_PER_PAGE_FOR_TEXT_LAYER:
        parsed.warnings.append(
            f"Текстовый слой отсутствует или беден ({density:.0f} симв./стр.)."
        )
        if allow_ocr:
            ocr = _ocr_pdf(path)
            if ocr is not None:
                return ocr
            parsed.warnings.append("OCR недоступен (не установлены дополнительные зависимости).")
        else:
            parsed.warnings.append("OCR отключён (--ocr для включения).")
    return parsed


def _split_pdf_page(page_text: str) -> list[str]:
    """Join hard-wrapped PDF lines back into paragraphs.

    A new paragraph starts on a blank line, on an obvious structural marker
    (``1.``, ``I.``, ``ПРИЛОЖЕНИЕ``…) or after a line that ends a sentence and
    is noticeably shorter than the running line width.
    """
    from ingestion.structure_parser import starts_new_structural_unit

    paragraphs: list[str] = []
    buffer: list[str] = []
    lines = [line.rstrip() for line in page_text.splitlines()]
    widths = [len(line) for line in lines if line.strip()]
    typical_width = max(sorted(widths)[len(widths) // 2] if widths else 0, 40)

    def flush() -> None:
        if buffer:
            joined = " ".join(part.strip() for part in buffer if part.strip())
            joined = joined.replace("- ", "-") if joined.count("- ") > 6 else joined
            if joined.strip():
                paragraphs.append(joined.strip())
            buffer.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if starts_new_structural_unit(stripped):
            flush()
            buffer.append(stripped)
            continue
        buffer.append(stripped)
        if stripped.endswith((".", ";", ":")) and len(stripped) < typical_width * 0.75:
            flush()
    flush()
    return paragraphs


def _ocr_pdf(path: Path) -> ParsedDocument | None:
    """Optional OCR path — requires ``pytesseract`` + Tesseract binary."""
    try:
        import io

        import pymupdf
        import pytesseract  # type: ignore[import-not-found]
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError:
        return None

    from app.config import get_settings

    settings = get_settings()
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

    document = pymupdf.open(path)
    blocks: list[TextBlock] = []
    raw_parts: list[str] = []
    try:
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(dpi=settings.ocr_dpi)
            image = Image.open(io.BytesIO(pixmap.tobytes("png")))
            text = pytesseract.image_to_string(image, lang=settings.ocr_languages)
            raw_parts.append(text)
            for paragraph in _split_pdf_page(text):
                blocks.append(TextBlock(text=paragraph, page=page_index + 1))
        n_pages = document.page_count
    finally:
        document.close()

    return ParsedDocument(
        path=path,
        blocks=blocks,
        n_pages=n_pages,
        raw_text="\n".join(raw_parts),
        parser_used="pymupdf+ocr",
        warnings=["Текст получен OCR — возможны ошибки распознавания."],
    )


# --------------------------------------------------------------------------- #
# DOCX
# --------------------------------------------------------------------------- #


def _parse_docx(path: Path) -> ParsedDocument:
    import docx  # python-docx

    document = docx.Document(str(path))
    blocks: list[TextBlock] = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            blocks.append(TextBlock(text=text, style=paragraph.style.name or ""))

    for table in document.tables:
        rendered = _render_docx_table(table)
        if rendered:
            blocks.append(TextBlock(text=rendered, is_table=True, style="Table"))

    raw_text = "\n".join(block.text for block in blocks)
    return ParsedDocument(
        path=path, blocks=blocks, raw_text=raw_text, parser_used="python-docx"
    )


def _render_docx_table(table: object) -> str:
    """Flatten a DOCX table into pipe-separated rows (retrievable as text)."""
    rows: list[str] = []
    for row in getattr(table, "rows", []):
        cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
        if any(cells):
            rows.append(" | ".join(cells))
    if not rows:
        return ""
    return "[ТАБЛИЦА]\n" + "\n".join(rows)


# --------------------------------------------------------------------------- #
# HTML / text
# --------------------------------------------------------------------------- #


def _parse_html(path: Path) -> ParsedDocument:
    from bs4 import BeautifulSoup

    html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    blocks: list[TextBlock] = []
    for element in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "div"]):
        if element.find(["p", "li", "div"]):
            continue  # keep leaf-level text only
        text = element.get_text(" ", strip=True)
        if text:
            blocks.append(TextBlock(text=text, style=element.name))

    raw_text = soup.get_text("\n", strip=True)
    return ParsedDocument(
        path=path, blocks=blocks, raw_text=raw_text, parser_used="beautifulsoup"
    )


def _parse_txt(path: Path) -> ParsedDocument:
    raw_text = path.read_text(encoding="utf-8", errors="replace")
    # Reuse the PDF re-assembler: it splits on blank lines *and* on structural
    # markers, so a numbered пункт never gets glued to the previous paragraph.
    blocks = [TextBlock(text=part) for part in _split_pdf_page(raw_text)]
    return ParsedDocument(
        path=path, blocks=blocks, raw_text=raw_text, parser_used="plaintext"
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def parse_file(path: Path | str, allow_ocr: bool = False) -> ParsedDocument:
    """Parse *path* into ordered text blocks.

    ``allow_ocr`` only takes effect for PDFs without a usable text layer.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        parsed = _parse_pdf(path, allow_ocr)
    elif suffix == ".docx":
        parsed = _parse_docx(path)
    elif suffix in (".html", ".htm"):
        parsed = _parse_html(path)
    elif suffix in (".txt", ".md"):
        parsed = _parse_txt(path)
    else:
        raise UnsupportedFormat(
            f"Формат {suffix!r} не поддерживается (доступно: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))})"
        )

    parsed.blocks = [
        TextBlock(
            text=normalize_text(block.text),
            page=block.page,
            is_table=block.is_table,
            style=block.style,
        )
        for block in parsed.blocks
        if normalize_text(block.text)
    ]
    parsed.normalized_text = "\n\n".join(block.text for block in parsed.blocks)
    logger.debug(
        "Parsed %s with %s: %d blocks, %d chars",
        path.name,
        parsed.parser_used,
        len(parsed.blocks),
        len(parsed.normalized_text),
    )
    return parsed
