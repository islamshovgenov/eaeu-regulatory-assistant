"""Retrieval: BM25 tokenisation, RRF fusion, INN handling, tier policy."""

from __future__ import annotations

import pytest

from app.config import Settings
from app.db.models import SourceAuthority, SourceTier
from app.rag.hybrid_search import HybridSearcher, RetrievedChunk
from app.rag.reranker import LexicalOverlapReranker, NoOpReranker
from app.rag.retriever import (
    RegulatoryRetriever,
    build_inn_queries,
    build_profile_queries,
)
from app.regulatory.inn import (
    canonical_inn,
    detect_inns,
    extract_inns_from_title,
    inn_variants,
    register_inns,
    to_cyrillic,
    to_latin,
)
from app.regulatory.schemas import ProductProfile, ProductType, ReleaseType
from app.rag.indexes import BM25Index, tokenize
from tests.conftest import make_chunk


# --------------------------------------------------------------------------- #
# BM25
# --------------------------------------------------------------------------- #


def test_tokenizer_keeps_regulatory_tokens():
    tokens = tokenize("AUC(0-t) и Cmax, 90% доверительный интервал, пункт 42")
    assert "auc" in tokens
    assert "cmax" in tokens
    assert "90" in tokens
    assert "42" in tokens
    assert "и" not in tokens  # stop-word


def test_bm25_finds_literal_regulatory_phrase():
    chunks = [
        make_chunk("c1", "42. Биовейвер допускается при выполнении условий БКС."),
        make_chunk("c2", "11. Оценка проводится по AUC(0-t) и Cmax."),
        make_chunk("c3", "Требования к маркировке вторичной упаковки."),
    ]
    index = BM25Index.build(chunks)
    results = index.search("AUC Cmax доверительный интервал", top_k=3)

    assert results, "BM25 не вернул результатов"
    assert results[0][0] == "c2"


def test_bm25_empty_query_returns_nothing():
    index = BM25Index.build([make_chunk("c1", "текст")])
    assert index.search("и в на", top_k=5) == []


def test_bm25_roundtrip(tmp_path):
    # BM25 IDF needs more than one document: a term present in every document
    # of a one-document corpus gets a non-positive weight.
    index = BM25Index.build(
        [
            make_chunk("c1", "биовейвер условия БКС"),
            make_chunk("c2", "требования к маркировке упаковки"),
            make_chunk("c3", "клинические исследования эффективности"),
        ]
    )
    path = tmp_path / "bm25.pkl"
    index.save(path)
    loaded = BM25Index.load(path)

    assert loaded is not None
    assert loaded.chunk_ids == index.chunk_ids
    assert loaded.search("биовейвер", 3)


def test_bm25_load_missing_file_returns_none(tmp_path):
    assert BM25Index.load(tmp_path / "absent.pkl") is None


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #


def test_rrf_prefers_chunks_found_by_both_rankers():
    settings = Settings(rrf_k=60)
    searcher = HybridSearcher.__new__(HybridSearcher)  # no I/O
    searcher.settings = settings

    both = RetrievedChunk(chunk=make_chunk("both", "x"), vector_rank=3, bm25_rank=3)
    vector_only = RetrievedChunk(chunk=make_chunk("vec", "x"), vector_rank=1)
    bm25_only = RetrievedChunk(chunk=make_chunk("bm", "x"), bm25_rank=1)

    fused = searcher._fuse([vector_only, bm25_only, both])
    assert fused[0].chunk_id == "both"


def test_searcher_requires_at_least_one_channel():
    with pytest.raises(ValueError, match="At least one retrieval channel"):
        HybridSearcher(enable_vector=False, enable_bm25=False)


def test_disabled_channels_do_not_expose_injected_indexes():
    searcher = HybridSearcher.__new__(HybridSearcher)
    searcher.enable_vector = False
    searcher.enable_bm25 = False
    searcher._vector_index = object()
    searcher._bm25_index = object()
    searcher._bm25_loaded = True

    assert searcher.vector_index is None
    assert searcher.bm25_index is None


def test_lexical_reranker_promotes_literal_overlap():
    items = [
        RetrievedChunk(chunk=make_chunk("a", "Требования к маркировке упаковки"), score=0.05),
        RetrievedChunk(
            chunk=make_chunk("b", "42. Биовейвер допускается при выполнении условий"),
            score=0.04,
        ),
    ]
    reranked = LexicalOverlapReranker().rerank("биовейвер условия", items, top_k=2)
    assert reranked[0].chunk_id == "b"


def test_noop_reranker_preserves_order():
    items = [
        RetrievedChunk(chunk=make_chunk("a", "x"), score=0.1),
        RetrievedChunk(chunk=make_chunk("b", "y"), score=0.2),
    ]
    assert [i.chunk_id for i in NoOpReranker().rerank("q", items, 2)] == ["a", "b"]


# --------------------------------------------------------------------------- #
# INN handling
# --------------------------------------------------------------------------- #


def test_transliteration_roundtrip():
    assert to_latin("ибупрофен") == "ibuprofen"
    assert to_cyrillic("Ibuprofenum").startswith("ибупрофен")


def test_inn_variants_cover_both_scripts():
    variants = inn_variants("ибупрофен")
    assert "ибупрофен" in variants
    assert "ibuprofen" in variants


def test_canonical_inn_maps_english_to_russian():
    assert canonical_inn("Ibuprofen") == "ибупрофен"
    assert canonical_inn("acetaminophen") == "парацетамол"


def test_detect_inns_finds_declined_forms():
    assert "ибупрофен" in detect_inns("исследование препарата ибупрофена 400 мг")


def test_detect_inns_does_not_invent_substances():
    assert detect_inns("Общие требования к регистрации лекарственных препаратов") == []


def test_extract_inns_from_expert_committee_title():
    title = (
        "451. О выборе референтного лекарственного препарата для целей исследования "
        "биоэквивалентности лекарственных препаратов (содержащих действующее "
        "вещество с МНН «эстриол» в лекарственных формах таблетки, 2 мг)"
    )
    assert extract_inns_from_title(title) == ["эстриол"]


def test_extract_inns_handles_combinations():
    title = 'Рекомендация о комбинации («гвайфенезин» + «парацетамол»)'
    assert extract_inns_from_title(title) == ["гвайфенезин", "парацетамол"]


def test_register_inns_extends_vocabulary():
    added = register_inns(["тестовоевещество"])
    assert added == 1
    assert "тестовоевещество" in detect_inns("препарат тестовоевещество 10 мг")
    assert register_inns(["тестовоевещество"]) == 0  # idempotent


# --------------------------------------------------------------------------- #
# Query building
# --------------------------------------------------------------------------- #


def test_profile_queries_cover_generic_topics():
    profile = ProductProfile(
        inn="ибупрофен",
        dosage_form="таблетки",
        strength="400 мг",
        route_of_administration="пероральный",
        product_type=ProductType.GENERIC,
        release_type=ReleaseType.IMMEDIATE,
    )
    queries = build_profile_queries(profile)
    blob = " ".join(queries).lower()

    assert any("биоэквивалент" in q.lower() for q in queries)
    assert "биовейвер" in blob or "bcs" in blob
    assert "референтн" in blob
    assert len(queries) == len(set(queries))


def test_profile_queries_react_to_flags():
    profile = ProductProfile(
        inn="циклоспорин",
        product_type=ProductType.GENERIC,
        release_type=ReleaseType.MODIFIED,
        highly_variable=True,
        narrow_therapeutic_index=True,
        additional_strengths="50 мг",
    )
    blob = " ".join(build_profile_queries(profile)).lower()
    assert "высоковариабельн" in blob
    assert "узким терапевтическ" in blob
    assert "модифицированным высвобожден" in blob
    assert "дополнительные дозировки" in blob


def test_inn_queries_include_expert_committee_phrasing():
    queries = build_inn_queries("ибупрофен")
    assert any("Экспертного комитета" in q for q in queries)


# --------------------------------------------------------------------------- #
# Tier policy
# --------------------------------------------------------------------------- #


def _retriever() -> RegulatoryRetriever:
    retriever = RegulatoryRetriever.__new__(RegulatoryRetriever)
    retriever.settings = Settings(final_top_k=4, supplementary_sources_enabled=True)
    return retriever


def test_supplementary_sources_are_capped():
    items = [
        RetrievedChunk(
            chunk=make_chunk(
                f"s{i}", "supplementary", tier=SourceTier.TIER_4, authority=SourceAuthority.ICH
            ),
            score=1.0 - i * 0.01,
        )
        for i in range(4)
    ]
    items.append(
        RetrievedChunk(chunk=make_chunk("eaeu", "обязательный акт"), score=0.5)
    )
    selected = _retriever()._apply_tier_policy(items, final_top_k=4)

    tier4 = [i for i in selected if i.chunk.source_tier == SourceTier.TIER_4]
    assert len(tier4) <= 1
    assert any(i.chunk_id == "eaeu" for i in selected)


def test_binding_source_is_pulled_in_when_absent():
    items = [
        RetrievedChunk(
            chunk=make_chunk(
                f"s{i}", "x", tier=SourceTier.TIER_4, authority=SourceAuthority.ICH
            ),
            score=1.0,
        )
        for i in range(3)
    ]
    items.append(RetrievedChunk(chunk=make_chunk("binding", "акт"), score=0.001))
    selected = _retriever()._apply_tier_policy(items, final_top_k=4)
    assert any(i.chunk_id == "binding" for i in selected)
