"""Text normalisation for Russian regulatory documents.

Normalisation is intentionally *conservative*: it repairs artefacts introduced
by PDF/DOCX extraction but never rewrites wording, numbers or punctuation that
could change the legal meaning of a provision.
"""

from __future__ import annotations

import re
import unicodedata

# --- character-level fixes -------------------------------------------------

#: Unicode spaces (NBSP, narrow NBSP, thin space, …) -> ordinary space.
_SPACE_CHARS = "          "
_SPACE_TRANSLATION = {ord(ch): " " for ch in _SPACE_CHARS}
#: Soft hyphen and zero-width characters -> removed.
_SPACE_TRANSLATION.update({ord(ch): None for ch in "­​‌‍﻿"})
#: Typographic dashes/quotes are kept, but the exotic minus signs are folded.
_SPACE_TRANSLATION.update({ord("−"): "-", ord("‐"): "-", ord("‑"): "-"})

_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
#: PDF hyphenation across a line break: "лекарствен-\nных" -> "лекарственных".
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\s*\n\s*(\w)")
#: Stray spaces before punctuation produced by column extraction.
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?%)\]])")
_SPACE_AFTER_OPEN_RE = re.compile(r"([(\[])\s+")
#: Page furniture such as a lone page number on its own line.
_PAGE_NUMBER_LINE_RE = re.compile(r"^\s*[-—]?\s*\d{1,4}\s*[-—]?\s*$")

def normalize_text(text: str) -> str:
    """Return a cleaned single-paragraph string.

    Steps: NFKC-lite unicode folding, unusual-space removal, de-hyphenation of
    line breaks, whitespace collapsing.  Content words are never altered.
    """
    if not text:
        return ""

    result = unicodedata.normalize("NFC", text)
    result = result.translate(_SPACE_TRANSLATION)
    result = _HYPHEN_BREAK_RE.sub(r"\1\2", result)
    result = result.replace("\r\n", "\n").replace("\r", "\n")

    lines = [
        line for line in result.split("\n") if not _PAGE_NUMBER_LINE_RE.match(line)
    ]
    result = "\n".join(lines)

    result = _MULTI_SPACE_RE.sub(" ", result)
    result = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", result)
    result = _SPACE_AFTER_OPEN_RE.sub(r"\1", result)
    result = _MULTI_NEWLINE_RE.sub("\n\n", result)
    return result.strip()


def normalize_for_search(text: str) -> str:
    """Aggressive normalisation used by BM25 tokenisation and INN matching."""
    result = normalize_text(text).lower()
    result = result.replace("ё", "е")
    result = re.sub(r"[«»\"'“”„]", " ", result)
    result = re.sub(r"\s+", " ", result)
    return result.strip()


#: Unicode blocks that identify the official language of an EAEU publication.
_SCRIPT_RANGES: tuple[tuple[str, int, int], ...] = (
    ("cyrillic", 0x0400, 0x04FF),
    ("armenian", 0x0530, 0x058F),
    ("georgian", 0x10A0, 0x10FF),
    ("latin", 0x0041, 0x024F),
)

#: Letters used by Kazakh/Kyrgyz/Belarusian but not by Russian.  The EAEU
#: publishes the same act in five languages, three of which are Cyrillic, so
#: script alone is not enough to isolate the Russian text.
_NON_RUSSIAN_CYRILLIC = set("әғқңөұүһіїєўјљњѓќѐ")


def detect_script(text: str) -> str:
    """Dominant script of *text* — ``cyrillic`` / ``armenian`` / ``latin`` / ``unknown``."""
    counts: dict[str, int] = {name: 0 for name, _, _ in _SCRIPT_RANGES}
    for char in text[:20000]:
        code = ord(char)
        for name, low, high in _SCRIPT_RANGES:
            if low <= code <= high:
                counts[name] += 1
                break
    total = sum(counts.values())
    if total < 50:
        return "unknown"
    name, count = max(counts.items(), key=lambda kv: kv[1])
    return name if count / total >= 0.5 else "unknown"


def is_russian_text(text: str, min_ratio: float = 0.005) -> bool:
    """``True`` when *text* is Russian rather than another EAEU language.

    Requires a Cyrillic-dominant script *and* a low share of the Cyrillic
    letters that Russian does not use (Kazakh ә/ғ/қ, Belarusian ў, …).
    """
    if detect_script(text) != "cyrillic":
        return False
    sample = text[:20000].lower()
    cyrillic = sum(1 for ch in sample if 0x0400 <= ord(ch) <= 0x04FF)
    if cyrillic < 50:
        return False
    foreign = sum(1 for ch in sample if ch in _NON_RUSSIAN_CYRILLIC)
    return (foreign / cyrillic) <= min_ratio


def estimate_tokens(text: str) -> int:
    """Rough token count for corpus statistics and context budgeting.

    Russian text averages ≈ 2.6 characters per token for modern BPE
    tokenisers; the estimate is used only for reporting, never for billing.
    """
    if not text:
        return 0
    return max(1, round(len(text) / 2.6))
