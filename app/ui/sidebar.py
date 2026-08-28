"""Sidebar: mode selection and knowledge-base status."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from app.config import Settings
from app.db.repository import Repository
from app.rag.pipeline import RegulatoryPipeline

MODE_CHAT = "Чат"
MODE_ASSESSMENT = "Регуляторная оценка"
MODE_KNOWLEDGE_BASE = "База знаний"

MODES = (MODE_CHAT, MODE_ASSESSMENT, MODE_KNOWLEDGE_BASE)


def render_sidebar(
    settings: Settings, repository: Repository, pipeline: RegulatoryPipeline
) -> str:
    """Draw the sidebar and return the selected mode."""
    with st.sidebar:
        st.markdown("## EAEU Regulatory Assistant")
        st.caption(
            "Поддержка регуляторного анализа лекарственных препаратов "
            "в рамках ЕАЭС"
        )

        mode = st.radio("Режим работы", MODES, index=0, key="mode")

        st.divider()
        st.markdown("### Модель и поиск")
        st.markdown(f"**LLM:** {pipeline.llm_status()}")
        embedding = (
            settings.embedding_model
            if settings.embedding_provider == "local"
            else settings.openai_embedding_model
        )
        st.markdown(f"**Эмбеддинги:** `{embedding}`")
        st.markdown(
            f"**Поиск:** гибридный (вектор {settings.vector_top_k} + "
            f"BM25 {settings.bm25_top_k} → RRF → {settings.final_top_k})"
        )
        st.markdown(
            "**Реранкинг:** "
            + (f"`{settings.rerank_model}`" if settings.rerank_enabled else "лексический (по умолчанию)")
        )

        st.divider()
        st.markdown("### Статус базы знаний")
        stats = _kb_stats(repository, pipeline)
        if stats["chunks"] == 0:
            st.error("База знаний пуста. Выполните ingestion.", icon="🚫")
            st.code("python _архив_сборки_базы/ingestion/run_ingestion.py", language="powershell")
        else:
            column_left, column_right = st.columns(2)
            column_left.metric("Документов", stats["documents"])
            column_right.metric("Чанков", stats["chunks"])
            column_left.metric("Векторов", stats["vectors"])
            column_right.metric("Актов ЕАЭС", stats["eaeu"])
            st.caption(
                f"Рекомендаций Экспертного комитета: {stats['expert_committee']}"
            )
            st.caption(f"Последнее обновление: {stats['last_update']}")

        st.divider()
        st.caption(
            "Система носит вспомогательный характер и не заменяет решение "
            "уполномоченного регуляторного эксперта. Нормативные утверждения "
            "формируются только на основании проиндексированных документов."
        )
    return mode


@st.cache_data(ttl=60, show_spinner=False)
def _cached_document_counts(_repository: Repository) -> dict[str, int]:
    by_authority = _repository.counts_by("source_authority")
    by_type = _repository.counts_by("document_type")
    return {
        "documents": _repository.count_documents(),
        "chunks": _repository.count_chunks(),
        "eaeu": by_authority.get("EAEU", 0),
        "expert_committee": by_type.get("expert_committee_recommendation", 0),
    }


def _kb_stats(repository: Repository, pipeline: RegulatoryPipeline) -> dict[str, object]:
    counts = _cached_document_counts(repository)
    try:
        vectors = pipeline.retriever.stats().get("vectors", 0)
    except Exception:  # noqa: BLE001 - status must never break the UI
        vectors = 0
    last_update = repository.last_update()
    return {
        **counts,
        "vectors": vectors,
        "last_update": (
            last_update.strftime("%d.%m.%Y %H:%M") if last_update else "нет данных"
        ),
    }
