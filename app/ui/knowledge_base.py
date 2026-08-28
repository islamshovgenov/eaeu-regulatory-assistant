"""Knowledge Base page: corpus composition, provenance and document search."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.config import CORPUS_STATS_JSON
from app.db.models import DocumentStatus, DocumentType
from app.db.repository import Repository
from app.rag.pipeline import RegulatoryPipeline

_STATUS_LABELS = {
    DocumentStatus.ACTIVE: "действует",
    DocumentStatus.AMENDED: "изменён",
    DocumentStatus.SUPERSEDED: "утратил силу",
    DocumentStatus.EXPIRED: "истёк",
    DocumentStatus.UNKNOWN: "не подтверждён",
}

_TYPE_LABELS = {
    DocumentType.AGREEMENT: "Соглашение/Договор",
    DocumentType.COUNCIL_DECISION: "Решение Совета ЕЭК",
    DocumentType.COUNCIL_DISPOSITION: "Распоряжение Совета ЕЭК",
    DocumentType.COUNCIL_RECOMMENDATION: "Рекомендация Совета ЕЭК",
    DocumentType.COLLEGE_DECISION: "Решение Коллегии ЕЭК",
    DocumentType.COLLEGE_RECOMMENDATION: "Рекомендация Коллегии ЕЭК",
    DocumentType.EXPERT_COMMITTEE_RECOMMENDATION: "Рекомендация Экспертного комитета",
    DocumentType.GUIDELINE: "Руководство",
    DocumentType.AMENDMENT: "Акт о внесении изменений",
    DocumentType.SUPPLEMENTARY: "Международный справочный документ",
    DocumentType.UNKNOWN: "Не классифицирован",
}


def render_knowledge_base(repository: Repository, pipeline: RegulatoryPipeline) -> None:
    st.markdown("### База знаний")
    st.caption(
        "Состав проиндексированного корпуса, происхождение документов и поиск "
        "по реестру. Все документы загружены из официальных открытых источников."
    )

    documents = repository.list_documents()
    if not documents:
        st.error("Реестр документов пуст. Выполните ingestion.", icon="🚫")
        st.code("python _архив_сборки_базы/ingestion/run_ingestion.py", language="powershell")
        return

    by_authority = repository.counts_by("source_authority")
    by_status = repository.counts_by("status")
    by_type = repository.counts_by("document_type")
    by_version = repository.counts_by("version_status")
    by_download = repository.counts_by("download_status")

    metrics = st.columns(5)
    metrics[0].metric("Всего документов", repository.count_documents())
    metrics[1].metric("Документы ЕАЭС", by_authority.get("EAEU", 0))
    metrics[2].metric(
        "Экспертный комитет", by_type.get("expert_committee_recommendation", 0)
    )
    metrics[3].metric("Чанков", repository.count_chunks())
    try:
        vectors = pipeline.retriever.stats().get("vectors", 0)
    except Exception:  # noqa: BLE001
        vectors = 0
    metrics[4].metric("Векторов в индексе", vectors)

    last_update = repository.last_update()
    st.caption(
        "Дата последнего обновления базы: "
        + (last_update.strftime("%d.%m.%Y %H:%M") if last_update else "нет данных")
        + f" · Оценка объёма корпуса: ~{repository.total_tokens():,} токенов".replace(",", " ")
    )

    tab_composition, tab_registry, tab_provenance = st.tabs(
        ["Состав корпуса", "Реестр документов", "Происхождение и ограничения"]
    )

    # ---------------------------------------------------------------- состав
    with tab_composition:
        left, right = st.columns(2)
        with left:
            st.markdown("**По издающему органу**")
            st.dataframe(
                _as_frame(by_authority, "Орган"),
                use_container_width=True,
                hide_index=True,
            )
            st.markdown("**По статусу документа**")
            st.dataframe(
                _as_frame(
                    {
                        _STATUS_LABELS.get(DocumentStatus(k), k): v
                        for k, v in by_status.items()
                    },
                    "Статус",
                ),
                use_container_width=True,
                hide_index=True,
            )
        with right:
            st.markdown("**По типу документа**")
            st.dataframe(
                _as_frame(
                    {
                        _TYPE_LABELS.get(DocumentType(k), k): v
                        for k, v in by_type.items()
                    },
                    "Тип",
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.markdown("**По статусу редакции**")
            st.dataframe(
                _as_frame(by_version, "Редакция"),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("**По году принятия**")
        years = repository.counts_by_year()
        if years:
            frame = pd.DataFrame(
                {"Год": list(years), "Документов": list(years.values())}
            ).set_index("Год")
            st.bar_chart(frame)

        st.markdown("**Статус загрузки**")
        st.dataframe(
            _as_frame(by_download, "Загрузка"), use_container_width=True, hide_index=True
        )

        if CORPUS_STATS_JSON.exists():
            st.caption(f"Подробная статистика: `{CORPUS_STATS_JSON}`")

    # ---------------------------------------------------------------- реестр
    with tab_registry:
        search_columns = st.columns([3, 2, 2])
        query = search_columns[0].text_input("Поиск по названию или номеру", "")
        authority = search_columns[1].selectbox(
            "Орган", ["все", *sorted(by_authority)], index=0
        )
        doc_type = search_columns[2].selectbox(
            "Тип", ["все", *sorted(by_type)], index=0
        )

        filtered = repository.list_documents(
            source_authority=None if authority == "все" else authority,
            document_type=None if doc_type == "все" else doc_type,
            search=query or None,
        )
        st.caption(f"Найдено документов: {len(filtered)}")

        table = pd.DataFrame(
            [
                {
                    "ID": d.document_id,
                    "Тип": _TYPE_LABELS.get(d.document_type, str(d.document_type)),
                    "№": d.document_number,
                    "Дата": d.adoption_date.strftime("%d.%m.%Y") if d.adoption_date else "",
                    "Название": d.title[:140],
                    "Статус": _STATUS_LABELS.get(d.status, str(d.status)),
                    "Чанков": d.n_chunks,
                    "Орган": str(d.source_authority),
                    "URL": d.source_url,
                }
                for d in filtered[:600]
            ]
        )
        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            column_config={"URL": st.column_config.LinkColumn("Источник")},
        )

        if filtered:
            selected_id = st.selectbox(
                "Подробно о документе",
                [d.document_id for d in filtered[:600]],
                index=0,
            )
            document = repository.get_document(selected_id)
            if document:
                _render_document_details(document)

    # ------------------------------------------------------------ provenance
    with tab_provenance:
        st.markdown(
            """
Каждый документ в базе имеет полную цепочку происхождения:

- **source_url** — прямая ссылка на файл на официальном сайте;
- **page_url** — страница документа на `docs.eaeunion.org` / `eec.eaeunion.org`;
- **sha256**, **size_bytes**, **content_type** — контроль целостности файла;
- **download_date** — дата обращения к источнику;
- **local_file** — путь к неизменённому оригиналу в `data/raw/`.

Полный реестр выгружается в `data/registry/documents.csv`.
            """
        )
        st.warning(
            "Ограничение версионирования: система **не формирует** консолидированные "
            "редакции. Документы с изменениями хранятся как исходный акт плюс "
            "отдельные акты о внесении изменений; для таких документов "
            "`version_status` не подтверждён автоматически и требует экспертной "
            "проверки.",
            icon="⚠️",
        )
        manual = [d for d in documents if d.download_status.value == "manual_required"]
        if manual:
            st.markdown("**Требуют ручной загрузки:**")
            for document in manual:
                st.markdown(f"- {document.title} — {document.source_url}")


def _as_frame(counts: dict[str, int], label: str) -> pd.DataFrame:
    return pd.DataFrame(
        sorted(counts.items(), key=lambda kv: kv[1], reverse=True),
        columns=[label, "Документов"],
    )


def _render_document_details(document) -> None:  # noqa: ANN001 - DocumentRecord
    left, right = st.columns(2)
    with left:
        st.markdown(f"**Название:** {document.title}")
        st.markdown(f"**Орган:** {document.authority}")
        st.markdown(f"**Тип:** {_TYPE_LABELS.get(document.document_type, '')}")
        st.markdown(f"**Номер:** {document.document_number or '—'}")
        st.markdown(
            "**Дата принятия:** "
            + (document.adoption_date.strftime("%d.%m.%Y") if document.adoption_date else "—")
        )
        st.markdown(f"**Тематики:** {', '.join(document.topic) or '—'}")
    with right:
        st.markdown(f"**Статус:** {_STATUS_LABELS.get(document.status, '')}")
        st.markdown(f"**Редакция:** {document.version_status}")
        st.markdown(f"**Изменён актами:** {', '.join(document.amended_by) or '—'}")
        st.markdown(f"**SHA256:** `{document.sha256[:32]}…`")
        st.markdown(f"**Размер:** {document.size_bytes:,} байт".replace(",", " "))
        st.markdown(f"**Страниц:** {document.n_pages or '—'} · **Чанков:** {document.n_chunks}")
    if document.source_url:
        st.markdown(f"[Исходный файл]({document.source_url})")
    if document.page_url:
        st.markdown(f"[Страница документа]({document.page_url})")
    st.caption(f"Локальный файл: `{document.local_file}`")
    if document.notes:
        st.caption(f"Примечания: {document.notes}")
