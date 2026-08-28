"""Query construction and retrieval orchestration.

Turns a user question or a :class:`ProductProfile` into a *set* of query
formulations, runs hybrid retrieval for each, fuses the results and applies the
tier-aware selection policy:

* binding EAEU acts (TIER 1–2) are always represented if any were retrieved;
* an INN-specific Expert Committee recommendation (TIER 3) is boosted into the
  final context, because it is the document a reviewer would actually apply;
* TIER 4 (ICH/EMA/WHO) is capped so supplementary material cannot crowd out the
  binding requirements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.config import Settings, get_settings
from app.db.models import DocumentType, SourceTier
from app.rag.hybrid_search import HybridSearcher, RetrievedChunk
from app.rag.reranker import Reranker, get_reranker
from app.regulatory.inn import canonical_inn, inn_variants
from app.regulatory.schemas import (
    PRODUCT_TYPE_LABELS,
    RELEASE_TYPE_LABELS,
    ProductProfile,
    ProductType,
    ReleaseType,
)

logger = logging.getLogger(__name__)

#: Maximum share of the final context that TIER-4 (ICH/EMA/WHO) may occupy.
_MAX_SUPPLEMENTARY_SHARE = 0.25


@dataclass(slots=True)
class CardDocument:
    """A relevant document whose text could not be indexed (a scanned PDF).

    It is deliberately kept out of the generation context — a registry card
    cannot support a normative statement — but it is still the document a
    reviewer would open, so it is reported to the user separately.
    """

    title: str
    url: str = ""
    page_url: str = ""


@dataclass(slots=True)
class RetrievalResult:
    """Everything the generator and the UI need to know about a retrieval."""

    items: list[RetrievedChunk] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    inn_specific_documents: list[str] = field(default_factory=list)
    card_only_documents: list[CardDocument] = field(default_factory=list)
    used_reranker: str = "none"

    @property
    def is_empty(self) -> bool:
        return not self.items

    def binding_items(self) -> list[RetrievedChunk]:
        return [
            item
            for item in self.items
            if item.chunk.source_tier in (SourceTier.TIER_1, SourceTier.TIER_2)
        ]


# --------------------------------------------------------------------------- #
# Query building
# --------------------------------------------------------------------------- #


def build_profile_queries(profile: ProductProfile) -> list[str]:
    """Query formulations for a structured regulatory assessment.

    Several targeted questions beat one long query: each retrieves the section
    of the corpus that actually answers it (registration rules, bioequivalence
    rules, biowaiver conditions, additional strengths, INN-specific advice).
    """
    inn = profile.inn.strip()
    form = profile.dosage_form
    strength = profile.strength
    route = profile.route_of_administration
    type_label = PRODUCT_TYPE_LABELS.get(profile.product_type, "")
    release_label = RELEASE_TYPE_LABELS.get(profile.release_type, "")

    queries: list[str] = []

    base = " ".join(
        part for part in (inn, form, strength, route, type_label, release_label) if part
    )
    if base:
        queries.append(base)

    if profile.product_type == ProductType.GENERIC:
        queries.append(
            "требования к регистрационному досье воспроизведённого лекарственного "
            "препарата, подтверждение эквивалентности референтному препарату"
        )
        queries.append(
            "исследование биоэквивалентности воспроизведённого лекарственного "
            "препарата, дизайн исследования, доверительный интервал 90%"
        )
        queries.append(
            "условия освобождения от исследования биоэквивалентности, биовейвер "
            "на основе биофармацевтической классификационной системы BCS"
        )
        queries.append(
            "выбор референтного лекарственного препарата для исследования "
            "биоэквивалентности"
        )
    elif profile.product_type == ProductType.HYBRID:
        queries.append(
            "гибридный лекарственный препарат, объём собственных исследований, "
            "отличия от референтного препарата"
        )
    elif profile.product_type in (ProductType.BIOLOGICAL, ProductType.BIOSIMILAR):
        queries.append(
            "требования к исследованиям биологических лекарственных препаратов, "
            "биоаналоговый (биоподобный) лекарственный препарат, сравнительные "
            "исследования, иммуногенность"
        )
    elif profile.product_type == ProductType.ORIGINAL:
        queries.append(
            "объём доклинических и клинических исследований оригинального "
            "лекарственного препарата, регистрационное досье, модуль 4 и модуль 5"
        )
    elif profile.product_type == ProductType.WELL_ESTABLISHED_USE:
        queries.append(
            "хорошо изученное медицинское применение, библиографические данные "
            "вместо собственных исследований"
        )
    elif profile.product_type == ProductType.COMBINATION:
        queries.append(
            "комбинированный лекарственный препарат с фиксированной комбинацией "
            "действующих веществ, обоснование комбинации, объём исследований"
        )
    else:
        queries.append(
            "порядок регистрации и экспертизы лекарственного препарата, состав "
            "регистрационного досье"
        )

    if profile.release_type in (
        ReleaseType.MODIFIED,
        ReleaseType.PROLONGED,
        ReleaseType.DELAYED,
    ):
        queries.append(
            "лекарственные препараты с модифицированным высвобождением, "
            "исследования биоэквивалентности, однократный и многократный приём"
        )
    if profile.additional_strengths:
        queries.append(
            "дополнительные дозировки лекарственного препарата, освобождение от "
            "исследования биоэквивалентности для дополнительных дозировок, "
            "пропорциональность состава"
        )
    if profile.highly_variable:
        queries.append(
            "высоковариабельные лекарственные препараты, расширение границ "
            "приемлемости, репликативный дизайн исследования"
        )
    if profile.narrow_therapeutic_index:
        queries.append(
            "лекарственные препараты с узким терапевтическим диапазоном, суженные "
            "границы биоэквивалентности"
        )
    if profile.new_indication:
        queries.append("регистрация нового показания к применению")
    if profile.new_dosage_form:
        queries.append("регистрация новой лекарственной формы")

    if inn:
        queries.extend(build_inn_queries(inn))

    # Deduplicate, preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for query in queries:
        if query and query not in seen:
            seen.add(query)
            unique.append(query)
    return unique


def build_inn_queries(inn: str) -> list[str]:
    """INN-specific formulations: Russian, Latin and Expert-Committee phrasing."""
    variants = [v for v in inn_variants(inn) if len(v) >= 5][:4]
    queries = [
        f"{variant} исследование биоэквивалентности референтный лекарственный препарат"
        for variant in variants[:2]
    ]
    canonical = canonical_inn(inn) or inn
    queries.append(
        f"рекомендация Экспертного комитета о выборе референтного лекарственного "
        f"препарата с МНН «{canonical}»"
    )
    return queries


def build_chat_queries(question: str, profile: ProductProfile | None = None) -> list[str]:
    """Query formulations for free-form chat."""
    queries = [question.strip()]
    if profile and profile.inn:
        queries.extend(build_inn_queries(profile.inn))
    return [q for q in dict.fromkeys(queries) if q]


# --------------------------------------------------------------------------- #
# Retriever
# --------------------------------------------------------------------------- #


#: How many scanned-only documents are worth reporting to the user.
_MAX_CARD_DOCUMENTS = 5


def _split_metadata_cards(
    candidates: list[RetrievedChunk],
) -> tuple[list[RetrievedChunk], list[CardDocument]]:
    """Separate real text fragments from registry cards of scanned documents.

    Cards used to compete for context slots and win on title similarity — an
    INN query would fill the whole context with "text not extracted" stubs, and
    the answer degenerated into a description of that fact.  They are now
    reported as documents to open manually instead.
    """
    usable: list[RetrievedChunk] = []
    cards: list[CardDocument] = []
    seen: set[str] = set()
    for item in candidates:
        chunk = item.chunk
        if not chunk.is_metadata_card:
            usable.append(item)
            continue
        if chunk.document_id in seen or len(cards) >= _MAX_CARD_DOCUMENTS:
            continue
        seen.add(chunk.document_id)
        cards.append(
            CardDocument(
                title=chunk.document_title or chunk.document_short_title,
                url=chunk.source_url,
                page_url=chunk.page_url,
            )
        )
    return usable, cards


class RegulatoryRetriever:
    """Hybrid retrieval + reranking + tier-aware selection."""

    def __init__(
        self,
        searcher: HybridSearcher | None = None,
        reranker: Reranker | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.searcher = searcher or HybridSearcher(self.settings)
        self.reranker = reranker or get_reranker(self.settings)

    def retrieve(
        self, queries: list[str], final_top_k: int | None = None
    ) -> RetrievalResult:
        final_top_k = final_top_k or self.settings.final_top_k
        if not queries:
            return RetrievalResult()

        candidates = self.searcher.multi_search(
            queries, final_top_k=max(self.settings.rerank_candidates, final_top_k * 3)
        )
        if not candidates:
            logger.info("Retrieval returned no candidates for %d queries", len(queries))
            return RetrievalResult(queries=queries)

        candidates, cards = _split_metadata_cards(candidates)
        if not candidates:
            logger.info("Only metadata cards matched %d queries", len(queries))
            return RetrievalResult(queries=queries, card_only_documents=cards)

        primary_query = queries[0]
        reranked = self.reranker.rerank(
            primary_query, candidates, top_k=max(final_top_k * 2, final_top_k)
        )
        selected = self._apply_tier_policy(reranked, final_top_k)

        inn_docs = sorted(
            {
                item.chunk.document_short_title or item.chunk.document_title[:80]
                for item in selected
                if item.chunk.document_type == DocumentType.EXPERT_COMMITTEE_RECOMMENDATION
            }
        )
        return RetrievalResult(
            items=selected,
            queries=queries,
            inn_specific_documents=inn_docs,
            card_only_documents=cards,
            used_reranker=self.reranker.name,
        )

    # -- selection policy ---------------------------------------------------
    def _apply_tier_policy(
        self, items: list[RetrievedChunk], final_top_k: int
    ) -> list[RetrievedChunk]:
        """Cap supplementary sources; guarantee binding acts keep their slots."""
        supplementary_cap = max(1, int(final_top_k * _MAX_SUPPLEMENTARY_SHARE))
        selected: list[RetrievedChunk] = []
        supplementary_used = 0

        for item in items:
            if len(selected) >= final_top_k:
                break
            if item.chunk.source_tier == SourceTier.TIER_4:
                if not self.settings.supplementary_sources_enabled:
                    continue
                if supplementary_used >= supplementary_cap:
                    continue
                supplementary_used += 1
            selected.append(item)

        # If nothing binding made it in, pull the best binding candidates up.
        if not any(
            item.chunk.source_tier in (SourceTier.TIER_1, SourceTier.TIER_2)
            for item in selected
        ):
            binding = [
                item
                for item in items
                if item.chunk.source_tier in (SourceTier.TIER_1, SourceTier.TIER_2)
            ][:2]
            if binding:
                selected = binding + selected[: max(0, final_top_k - len(binding))]
        return selected

    # -- health -------------------------------------------------------------
    def is_ready(self) -> bool:
        return self.searcher.is_ready()

    def stats(self) -> dict[str, int]:
        return self.searcher.stats()
