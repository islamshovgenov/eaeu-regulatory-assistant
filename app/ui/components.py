"""Reusable Streamlit rendering components."""

from __future__ import annotations

import re

import streamlit as st

from app.db.models import SourceAuthority, SourceTier
from app.rag.citations import Citation, CitationRegistry
from app.rag.pipeline import PipelineResult
from app.regulatory.schemas import (
    DECISION_STATUS_LABELS,
    STATEMENT_KIND_LABELS,
    ChatAnswer,
    ConfidenceLevel,
    ConfidenceReport,
    Decision,
    ProductProfile,
    RegulatoryAssessment,
    Statement,
    StudyCategory,
)

_CONFIDENCE_COLOR = {
    ConfidenceLevel.HIGH: "#1a7f37",
    ConfidenceLevel.MEDIUM: "#9a6700",
    ConfidenceLevel.LOW: "#b42318",
}

_TIER_BADGE = {
    SourceTier.TIER_1: ("🔴", "Обязательный акт ЕАЭС"),
    SourceTier.TIER_2: ("🟠", "Руководство/рекомендация ЕЭК"),
    SourceTier.TIER_3: ("🟡", "Рекомендация Экспертного комитета ЕЭК"),
    SourceTier.TIER_4: ("🔵", "Международный справочный источник"),
    SourceTier.TIER_5: ("⚪", "Научный источник"),
}

_STUDY_CATEGORY_TITLES = {
    StudyCategory.PHARMACEUTICAL: "Фармацевтические",
    StudyCategory.NONCLINICAL: "Доклинические",
    StudyCategory.CLINICAL: "Клинические",
}


# --------------------------------------------------------------------------- #
# Small pieces
# --------------------------------------------------------------------------- #


def render_confidence_badge(confidence: ConfidenceReport, cited_count: int) -> None:
    """One-line status strip: confidence, score and how many sources back it."""
    color = _CONFIDENCE_COLOR.get(confidence.level, "#57606a")
    note = (
        f"обоснован {cited_count} нормативными фрагментами"
        if cited_count
        else "ссылок на нормативные фрагменты нет"
    )
    st.markdown(
        f"""<div class="answer-status">
        <span style="padding:3px 10px;border-radius:12px;background:{color}1A;
        color:{color};border:1px solid {color}55;font-weight:600;">
        Уверенность: {confidence.level} · {confidence.score:.2f}</span>
        <span class="answer-status-note">{note}</span></div>""",
        unsafe_allow_html=True,
    )


def render_statement(statement: Statement) -> None:
    label = STATEMENT_KIND_LABELS.get(statement.kind, str(statement.kind))
    refs = (
        " " + " ".join(f"`[{i}]`" for i in statement.citation_ids)
        if statement.citation_ids
        else ""
    )
    st.markdown(f"- **{label}:** {statement.text}{refs}")


def render_decision(title: str, decision: Decision) -> None:
    st.markdown(f"**{title}:** {DECISION_STATUS_LABELS.get(decision.status, decision.status)}")
    if decision.rationale:
        st.markdown(f"_Обоснование:_ {decision.rationale}")
    if decision.conditions:
        st.markdown("_Условия:_")
        for condition in decision.conditions:
            st.markdown(f"- {condition}")
    if decision.citation_ids:
        st.markdown(
            "_Нормативные основания:_ "
            + " ".join(f"`[{i}]`" for i in decision.citation_ids)
        )
    else:
        st.caption("Ссылки на нормативные фрагменты отсутствуют.")


def render_citation_card(citation: Citation, expanded: bool = False) -> None:
    """Expandable citation with the exact retrieved fragment."""
    emoji, tier_label = _TIER_BADGE.get(citation.tier, ("⚪", "Источник"))
    with st.expander(
        f"{emoji} **[{citation.citation_id}]** {citation.header()}", expanded=expanded
    ):
        st.markdown(
            f"""<div class="citation-quote">{
                _escape(citation.snippet(1400))
            }</div>""",
            unsafe_allow_html=True,
        )

        # Provenance on one line: everything a reader needs to judge the source
        # without a two-column metadata table.
        meta = [tier_label, citation.status_label]
        if citation.chunk.document_number:
            meta.append(f"№ {citation.chunk.document_number}")
        if citation.chunk.adoption_date:
            meta.append(f"от {citation.chunk.adoption_date.strftime('%d.%m.%Y')}")
        if pages := citation.pages():
            meta.append(pages)
        if citation.chunk.access_date:
            meta.append(
                f"загружен {citation.chunk.access_date.strftime('%d.%m.%Y')}"
            )
        st.caption(" · ".join(meta))
        if citation.version_label:
            st.caption(f"Редакция: {citation.version_label}")

        links = []
        if citation.chunk.source_url:
            links.append(f"[Исходный файл]({citation.chunk.source_url})")
        if (
            citation.chunk.page_url
            and citation.chunk.page_url != citation.chunk.source_url
        ):
            links.append(f"[Страница на сайте ЕАЭС]({citation.chunk.page_url})")
        if links:
            st.markdown(" · ".join(links))

        if citation.tier == SourceTier.TIER_4:
            st.caption(
                "ℹ️ Дополнительный международный источник — не является "
                "обязательным требованием ЕАЭС."
            )


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


#: When the answer cites nothing, the retrieved fragments are shown as context
#: only — a long unused list is noise, so it is trimmed.
_UNUSED_SOURCES_SHOWN = 5


def render_sources(registry: CitationRegistry, used_ids: set[int] | None = None) -> None:
    """The "Источники" block with clickable/expandable citations.

    Sources actually referenced by the answer are shown in full; if the answer
    referenced none, the block degrades to a short "what the search found"
    list so that an unsupported answer never looks well-sourced.
    """
    if used_ids:
        citations = registry.used(used_ids)
        heading = f"**Источники ({len(citations)})**"
        hint = "Нажмите на источник, чтобы увидеть точный фрагмент."
    else:
        citations = registry.citations[:_UNUSED_SOURCES_SHOWN]
        heading = f"**Найденные фрагменты ({len(citations)})**"
        hint = (
            "Ответ не опирается ни на один из них — это то, что вернул поиск."
        )
    if not citations:
        st.info("Нормативные источники по запросу не найдены.")
        return

    counts = {
        "обязательных актов ЕАЭС": sum(1 for c in citations if c.is_binding_eaeu),
        "рекомендаций Экспертного комитета": sum(
            1 for c in citations if c.tier == SourceTier.TIER_3
        ),
        "справочных международных": sum(
            1 for c in citations if c.chunk.source_authority != SourceAuthority.EAEU
        ),
    }
    st.markdown(heading)
    st.caption(
        " · ".join(f"{label}: {n}" for label, n in counts.items() if n) + f". {hint}"
    )
    for citation in citations:
        render_citation_card(citation)


def render_pipeline_notes(result: PipelineResult) -> None:
    """Surface pipeline failures (e.g. a rejected LLM request) to the user.

    Without this the assistant silently falls back to retrieval-only mode and
    the operator has no way to tell a configuration problem from an empty
    knowledge base.
    """
    for note in result.notes:
        st.error(note, icon="⚠️")


# --------------------------------------------------------------------------- #
# Composite renderers
# --------------------------------------------------------------------------- #


#: A markdown line that must keep its own line: list item, heading, quote, table.
_STRUCTURAL_LINE_RE = re.compile(r"^(?:[-*+]\s|\d+[.)]\s|#{1,6}\s|>|\|)")


def normalize_answer_markdown(text: str) -> str:
    """Make model output read as prose: real paragraphs, no runaway blank space.

    Models often emit sentence-per-line text, which markdown glues into one
    wall of text.  Consecutive prose lines are joined into a paragraph and
    paragraphs are separated by a blank line; list items, headings and quotes
    are left untouched.
    """
    lines = [line.rstrip() for line in (text or "").strip().splitlines()]
    blocks: list[str] = []
    paragraph: list[str] = []
    structural: list[str] = []

    def flush() -> None:
        if paragraph:
            blocks.append(" ".join(paragraph))
            paragraph.clear()
        if structural:
            # Keep a list as one block so markdown renders it tight.
            blocks.append("\n".join(structural))
            structural.clear()

    for line in lines:
        stripped = line.lstrip()
        if not stripped:
            flush()
        elif _STRUCTURAL_LINE_RE.match(stripped):
            if paragraph:
                blocks.append(" ".join(paragraph))
                paragraph.clear()
            structural.append(line)
        else:
            if structural:
                # A continuation line of the previous list item.
                structural[-1] = f"{structural[-1]} {stripped}"
            else:
                paragraph.append(stripped)
    flush()
    return "\n\n".join(blocks)


def _render_answer_details(result: PipelineResult, answer: ChatAnswer) -> None:
    """Everything that justifies the answer, folded away by default.

    The typology of statements, the validation report, the confidence factors
    and the retrieval trace are what the project is defended on — but they are
    evidence, not reading material, so they live behind one expander instead of
    five stacked blocks.
    """
    with st.expander("Как получен этот ответ", expanded=False):
        if answer.statements:
            st.markdown("**Разбор утверждений по типам**")
            for statement in answer.statements:
                render_statement(statement)

        st.markdown("**Уровень уверенности**")
        st.caption(
            "Оценка вычисляется программно по признакам поиска и источников, "
            "а не запрашивается у языковой модели."
        )
        for factor in answer.confidence.factors:
            st.markdown(f"- {factor}")

        if messages := result.validation.messages():
            st.markdown("**Проверка ссылок и утверждений**")
            for message in messages:
                st.markdown(f"- {message}")

        if answer.limitations:
            st.markdown("**Ограничения анализа**")
            for limitation in answer.limitations:
                st.markdown(f"- {limitation}")

        _render_retrieval_trace(result)


def render_chat_answer(result: PipelineResult) -> None:
    answer: ChatAnswer | None = result.chat_answer
    if answer is None:
        st.error("Ответ не сформирован.")
        return

    render_pipeline_notes(result)
    st.markdown(
        normalize_answer_markdown(answer.answer_markdown) or "_Пустой ответ._"
    )

    if answer.clarifying_questions:
        st.info(
            "**Чтобы дать вывод по конкретному препарату, уточните:**\n"
            + "\n".join(f"- {q}" for q in answer.clarifying_questions),
            icon="❓",
        )

    used_ids = answer.all_citation_ids() or None
    render_confidence_badge(
        answer.confidence, len(result.registry.used(used_ids) if used_ids else [])
    )
    _render_answer_details(result, answer)
    render_sources(result.registry, used_ids)


def render_assessment(result: PipelineResult) -> None:
    assessment: RegulatoryAssessment | None = result.assessment
    profile: ProductProfile | None = result.profile
    if assessment is None or profile is None:
        st.error("Оценка не сформирована.")
        return

    render_pipeline_notes(result)
    st.markdown("# Регуляторная оценка")

    st.markdown("## 1. Исходные данные")
    for line in profile.summary_lines():
        st.markdown(f"- {line}")
    if profile.additional_strengths:
        st.markdown(f"- Дополнительные дозировки: {profile.additional_strengths}")

    if not assessment.input_sufficient:
        st.warning(
            "Введённых данных недостаточно для однозначного определения программы "
            "исследований. Ниже приведены уточняющие вопросы; выводы носят "
            "предварительный характер.",
            icon="⚠️",
        )
        for index, question in enumerate(assessment.clarifying_questions, start=1):
            st.markdown(f"{index}. {question}")

    st.markdown("## 2. Предполагаемая регистрационная стратегия")
    if assessment.product_classification:
        st.markdown(f"_Классификация:_ {assessment.product_classification}")
    if assessment.registration_strategy:
        for statement in assessment.registration_strategy:
            render_statement(statement)
    else:
        st.info("Стратегия не сформирована — нет достаточного нормативного основания.")

    st.markdown("## 3. Необходимые исследования")
    for category in (
        StudyCategory.PHARMACEUTICAL,
        StudyCategory.NONCLINICAL,
        StudyCategory.CLINICAL,
    ):
        studies = [s for s in assessment.required_studies if s.category == category]
        st.markdown(f"### {_STUDY_CATEGORY_TITLES[category]}")
        if not studies:
            st.caption("Нет положений, подтверждённых найденными источниками.")
            continue
        for study in studies:
            refs = (
                " " + " ".join(f"`[{i}]`" for i in study.citation_ids)
                if study.citation_ids
                else ""
            )
            label = STATEMENT_KIND_LABELS.get(study.kind, "")
            st.markdown(f"- **{study.name}** — _{study.necessity}_{refs}")
            if study.rationale:
                st.markdown(f"  - {label}: {study.rationale}")

    st.markdown("## 4. Биоэквивалентность")
    render_decision("Исследование биоэквивалентности", assessment.bioequivalence)

    st.markdown("## 5. Возможность biowaiver")
    render_decision("Биовейвер", assessment.biowaiver)

    st.markdown("## 6. Дополнительные дозировки")
    render_decision("Дополнительные дозировки", assessment.additional_strengths)

    st.markdown("## 7. Основные регуляторные риски")
    if assessment.regulatory_risks:
        for statement in assessment.regulatory_risks:
            render_statement(statement)
    else:
        st.caption("Риски не выявлены на основании найденных источников.")

    st.markdown("## 8. Потенциальные вопросы эксперта")
    if assessment.potential_authority_questions:
        st.caption(
            "Это предположения о возможных вопросах регуляторного эксперта, "
            "а не нормативные требования."
        )
        for statement in assessment.potential_authority_questions:
            render_statement(statement)
    else:
        st.caption("Не сформулированы.")

    if assessment.conflicts:
        st.markdown("### Обнаружены потенциально различающиеся нормативные положения")
        for conflict in assessment.conflicts:
            with st.expander(conflict.description[:120], expanded=False):
                st.markdown(conflict.description)
                if conflict.resolution_note:
                    st.markdown(f"_Что проверить:_ {conflict.resolution_note}")
                if conflict.citation_ids:
                    st.markdown(
                        "_Источники:_ "
                        + " ".join(f"`[{i}]`" for i in conflict.citation_ids)
                    )

    used_ids = assessment.all_citation_ids() or None
    render_confidence_badge(
        assessment.confidence,
        len(result.registry.used(used_ids) if used_ids else []),
    )
    with st.expander("Как получена эта оценка", expanded=False):
        st.markdown("**Уровень уверенности**")
        st.caption(
            "Оценка вычисляется программно по признакам поиска и источников, "
            "а не запрашивается у языковой модели."
        )
        for factor in assessment.confidence.factors:
            st.markdown(f"- {factor}")
        if messages := result.validation.messages():
            st.markdown("**Проверка ссылок и утверждений**")
            for message in messages:
                st.markdown(f"- {message}")
        st.markdown("**Ограничения анализа**")
        for limitation in assessment.limitations:
            st.markdown(f"- {limitation}")
        _render_retrieval_trace(result)

    st.markdown("## 9. Нормативные основания")
    render_sources(result.registry, used_ids)


def _render_retrieval_trace(result: PipelineResult) -> None:
    """What the retriever actually did — rendered inside a parent expander."""
    st.markdown("**Поиск**")
    st.caption(
        f"Реранкер: {result.retrieval.used_reranker} · "
        f"LLM: {result.llm_model or 'не использовалась'}"
    )
    for query in result.retrieval.queries:
        st.markdown(f"- запрос: {query}")
    for index, item in enumerate(result.retrieval.items, start=1):
        st.markdown(
            f"- фрагмент [{index}]: score={item.score:.4f}, "
            f"методы: {', '.join(item.retrieval_methods()) or '—'}"
        )
