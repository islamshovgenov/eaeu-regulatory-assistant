"""INN (МНН) normalisation, transliteration and detection.

The corpus is Russian, but users write INNs in Russian, in English, or in a
mixed transliteration ("ибупрофен", "ibuprofen", "Ibuprofenum").  Expert
Committee recommendations are INN-specific, so matching an INN reliably is what
makes those Tier-3 documents retrievable at all.

The mapping is *rule-based* (Russian↔Latin transliteration plus the standard
INN stem endings), with a small curated table for names whose Russian and Latin
forms are not related by transliteration.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

# --------------------------------------------------------------------------- #
# Transliteration
# --------------------------------------------------------------------------- #

#: Cyrillic -> Latin, tuned for pharmaceutical nomenclature rather than for
#: general Russian (e.g. "ц" -> "c", not "ts", because INNs use "c").
_CYR_TO_LAT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "kh", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya", "-": "-",
    " ": " ",
}

#: Latin -> Cyrillic (longest-first so digraphs win).
_LAT_TO_CYR = [
    ("sch", "щ"), ("sh", "ш"), ("ch", "ч"), ("zh", "ж"), ("kh", "х"),
    ("yu", "ю"), ("ya", "я"), ("ph", "ф"), ("th", "т"), ("ck", "к"),
    ("y", "и"), ("a", "а"), ("b", "б"), ("c", "ц"), ("d", "д"), ("e", "е"),
    ("f", "ф"), ("g", "г"), ("h", "х"), ("i", "и"), ("j", "й"), ("k", "к"),
    ("l", "л"), ("m", "м"), ("n", "н"), ("o", "о"), ("p", "п"), ("q", "к"),
    ("r", "р"), ("s", "с"), ("t", "т"), ("u", "у"), ("v", "в"), ("w", "в"),
    ("x", "кс"), ("z", "з"), ("-", "-"), (" ", " "),
]

#: Latin INN suffixes that the Russian form drops or changes.
_LATIN_SUFFIXES = ("um", "us", "a", "e")

#: Names whose Russian and Latin forms are not a plain transliteration.
_MANUAL_ALIASES: dict[str, tuple[str, ...]] = {
    "ибупрофен": ("ibuprofen", "ibuprofenum"),
    "парацетамол": ("paracetamol", "acetaminophen", "paracetamolum"),
    "омепразол": ("omeprazole", "omeprazol", "omeprazolum"),
    "метформин": ("metformin", "metformini", "metforminum"),
    "ацетилсалициловая кислота": ("acetylsalicylic acid", "aspirin"),
    "левотироксин натрия": ("levothyroxine sodium", "levothyroxinum natricum"),
    "кетопрофен": ("ketoprofen", "ketoprofenum"),
    "торасемид": ("torasemide", "torsemide"),
    "адапален": ("adapalene", "adapalenum"),
    "папаверин": ("papaverine", "papaverinum"),
    "лизоцим": ("lysozyme", "lysozymum"),
    "циклоспорин": ("ciclosporin", "cyclosporine"),
    "такролимус": ("tacrolimus",),
    "амлодипин": ("amlodipine", "amlodipinum"),
    "аторвастатин": ("atorvastatin", "atorvastatinum"),
    "рифампицин": ("rifampicin", "rifampin"),
    "варфарин": ("warfarin", "warfarinum"),
    "дигоксин": ("digoxin", "digoxinum"),
    "карбамазепин": ("carbamazepine", "carbamazepinum"),
    "леводопа": ("levodopa",),
}

_WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z][А-Яа-яЁёA-Za-z\-]{4,40}")


def _strip(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower()).replace("ё", "е")


def to_latin(value: str) -> str:
    """Transliterate a Cyrillic INN to Latin script."""
    text = _strip(value)
    return "".join(_CYR_TO_LAT.get(ch, ch) for ch in text)


def to_cyrillic(value: str) -> str:
    """Transliterate a Latin INN to Cyrillic script."""
    text = _strip(value)
    for suffix in _LATIN_SUFFIXES:
        if text.endswith(suffix) and len(text) > len(suffix) + 4:
            text = text[: -len(suffix)]
            break
    result: list[str] = []
    index = 0
    while index < len(text):
        for latin, cyrillic in _LAT_TO_CYR:
            if text.startswith(latin, index):
                result.append(cyrillic)
                index += len(latin)
                break
        else:
            result.append(text[index])
            index += 1
    return "".join(result)


def is_cyrillic(value: str) -> bool:
    return bool(re.search(r"[А-Яа-яЁё]", value or ""))


@lru_cache(maxsize=2048)
def inn_variants(value: str) -> tuple[str, ...]:
    """All spellings of an INN worth searching for.

    Returns the canonical (lower-cased) input plus the transliterated form,
    curated aliases and a truncated stem that survives Russian case endings.
    """
    canonical = _strip(value)
    if not canonical:
        return ()

    variants: list[str] = [canonical]

    if is_cyrillic(canonical):
        variants.append(to_latin(canonical))
        variants.extend(_MANUAL_ALIASES.get(canonical, ()))
    else:
        cyrillic = to_cyrillic(canonical)
        variants.append(cyrillic)
        for russian, aliases in _MANUAL_ALIASES.items():
            if canonical in aliases:
                variants.append(russian)
                variants.extend(aliases)

    # Stems tolerate Russian declension ("ибупрофена", "ибупрофеном").
    for variant in list(variants):
        if len(variant) > 7 and " " not in variant:
            variants.append(variant[:-2])

    seen: set[str] = set()
    ordered: list[str] = []
    for variant in variants:
        variant = variant.strip()
        if variant and variant not in seen:
            seen.add(variant)
            ordered.append(variant)
    return tuple(ordered)


def canonical_inn(value: str) -> str:
    """A single canonical (Russian, lower-case) form used for grouping."""
    canonical = _strip(value)
    if not canonical:
        return ""
    if is_cyrillic(canonical):
        return canonical
    for russian, aliases in _MANUAL_ALIASES.items():
        if canonical in aliases:
            return russian
    return to_cyrillic(canonical)


#: INNs the corpus is known to discuss by name.  Seeded by the curated table
#: and extended at ingestion time from Expert Committee recommendation titles
#: (they are literally titled after an INN), see :func:`load_vocabulary`.
_KNOWN_INNS: list[str] = sorted(_MANUAL_ALIASES)

#: Expert Committee titles quote the INN inside guillemets:
#: «О выборе референтного ... (содержащих ... с МНН «эстриол» ...)».
_GUILLEMET_RE = re.compile(r"«([^«»]{3,80})»")

#: Titles of the recommendations themselves — not substance names.
_NOT_AN_INN = (
    "о выборе", "об особенностях", "об оценке", "рекомендац", "мнн",
    "лекарственн", "препарат", "исследован", "биоэквивалент", "группировочн",
)


def known_inns() -> tuple[str, ...]:
    """Current INN vocabulary (curated table + ingested Expert Committee INNs)."""
    return tuple(_KNOWN_INNS)


#: Backwards-compatible alias used as a default argument value.
KNOWN_INNS: tuple[str, ...] = tuple(_KNOWN_INNS)


def extract_inns_from_title(title: str) -> list[str]:
    """Pull INNs out of an Expert Committee recommendation title.

    Combination products list several substances, each in its own guillemets,
    so all of them are returned.
    """
    results: list[str] = []
    for candidate in _GUILLEMET_RE.findall(title or ""):
        cleaned = _strip(candidate.replace("+", " + "))
        if len(cleaned) < 4 or len(cleaned) > 70:
            continue
        if any(marker in cleaned for marker in _NOT_AN_INN):
            continue
        for part in re.split(r"\s*\+\s*", cleaned):
            part = part.strip(" .,;:()[]")
            if len(part) >= 4 and not any(m in part for m in _NOT_AN_INN):
                results.append(part)
    return sorted(set(results))


def register_inns(names: list[str]) -> int:
    """Add INNs to the in-process vocabulary. Returns the number of new names."""
    global KNOWN_INNS
    added = 0
    existing = set(_KNOWN_INNS)
    for name in names:
        canonical = _strip(name)
        if canonical and canonical not in existing:
            existing.add(canonical)
            _KNOWN_INNS.append(canonical)
            added += 1
    if added:
        _KNOWN_INNS.sort()
        KNOWN_INNS = tuple(_KNOWN_INNS)
        inn_variants.cache_clear()
    return added


def save_vocabulary(path: "Path") -> None:
    """Persist the vocabulary so the app does not need to re-scan the corpus."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sorted(_KNOWN_INNS), ensure_ascii=False, indent=1), encoding="utf-8"
    )


def load_vocabulary(path: "Path") -> int:
    """Load a previously persisted vocabulary; returns the number of new names."""
    if not path.exists():
        return 0
    try:
        names = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0
    return register_inns([n for n in names if isinstance(n, str)])


def detect_inns(text: str, known: tuple[str, ...] | None = None) -> list[str]:
    """Return canonical INNs mentioned in *text*.

    Only names from the vocabulary are reported — a general-purpose chemical NER
    is out of scope and would produce false positives that pollute retrieval
    filters.
    """
    lowered = _strip(text)
    found: list[str] = []
    for inn in known if known is not None else _KNOWN_INNS:
        for variant in inn_variants(inn):
            if len(variant) < 5:
                continue
            if variant in lowered:
                found.append(canonical_inn(inn))
                break
    return sorted(set(found))
