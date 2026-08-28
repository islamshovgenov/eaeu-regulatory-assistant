"""Chat mode and the structured regulatory assessment form."""

from __future__ import annotations

import streamlit as st

from app.rag.pipeline import PipelineResult, RegulatoryPipeline
from app.regulatory.schemas import (
    PRODUCT_TYPE_LABELS,
    RELEASE_TYPE_LABELS,
    ProductProfile,
    ProductType,
    ReleaseType,
)
from app.ui.components import render_assessment, render_chat_answer

EXAMPLE_QUESTIONS = (
    "Что требуется для регистрации воспроизведённого лекарственного препарата?",
    "Когда возможен BCS-based biowaiver?",
    "Какие особенности исследования высоковариабельного лекарственного препарата?",
    "Как выбирается референтный препарат?",
    "Ибупрофен таблетки 400 мг, воспроизведённый препарат, немедленное высвобождение",
)


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #


def render_chat(pipeline: RegulatoryPipeline) -> None:
    st.markdown("### Чат с ассистентом")
    st.caption(
        "Задайте вопрос по нормативным требованиям ЕАЭС. Ответ формируется только "
        "на основании найденных фрагментов официальных документов."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list[tuple[str, str]]
    if "chat_results" not in st.session_state:
        st.session_state.chat_results = {}  # index -> PipelineResult

    with st.expander("Примеры запросов", expanded=not st.session_state.chat_history):
        columns = st.columns(2)
        for index, example in enumerate(EXAMPLE_QUESTIONS):
            if columns[index % 2].button(example, key=f"example_{index}", use_container_width=True):
                st.session_state.pending_question = example
                st.rerun()

    for index, (role, text) in enumerate(st.session_state.chat_history):
        with st.chat_message(role):
            if role == "user":
                st.markdown(text)
            else:
                result: PipelineResult | None = st.session_state.chat_results.get(index)
                if result is not None:
                    render_chat_answer(result)
                else:
                    st.markdown(text)

    question = st.chat_input("Ваш вопрос по регуляторным требованиям ЕАЭС…")
    pending = st.session_state.pop("pending_question", None)
    question = question or pending

    if not question:
        return

    st.session_state.chat_history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Поиск в нормативной базе и подготовка ответа…"):
            result = pipeline.answer_question(
                question, history=st.session_state.chat_history[:-1]
            )
        render_chat_answer(result)

    answer_text = result.chat_answer.answer_markdown if result.chat_answer else ""
    st.session_state.chat_history.append(("assistant", answer_text))
    st.session_state.chat_results[len(st.session_state.chat_history) - 1] = result


# --------------------------------------------------------------------------- #
# Assessment form
# --------------------------------------------------------------------------- #

_PRODUCT_TYPE_OPTIONS = [
    ProductType.UNKNOWN,
    ProductType.GENERIC,
    ProductType.ORIGINAL,
    ProductType.HYBRID,
    ProductType.BIOLOGICAL,
    ProductType.BIOSIMILAR,
    ProductType.COMBINATION,
    ProductType.WELL_ESTABLISHED_USE,
]

_RELEASE_OPTIONS = [
    ReleaseType.UNKNOWN,
    ReleaseType.IMMEDIATE,
    ReleaseType.MODIFIED,
    ReleaseType.DELAYED,
    ReleaseType.PROLONGED,
]


def render_assessment_form(pipeline: RegulatoryPipeline) -> None:
    st.markdown("### Регуляторная оценка препарата")
    st.caption(
        "Заполните характеристики препарата. Незаполненные обязательные поля "
        "приведут к уточняющим вопросам, а не к категорическому выводу."
    )

    with st.form("assessment_form"):
        column_left, column_right = st.columns(2)
        with column_left:
            inn = st.text_input("МНН", placeholder="например, ибупрофен")
            dosage_form = st.text_input("Лекарственная форма", placeholder="таблетки")
            strength = st.text_input("Дозировка", placeholder="400 мг")
            route = st.text_input("Путь введения", placeholder="пероральный")
        with column_right:
            product_type = st.selectbox(
                "Тип препарата",
                _PRODUCT_TYPE_OPTIONS,
                format_func=lambda t: PRODUCT_TYPE_LABELS[t],
            )
            release_type = st.selectbox(
                "Характер высвобождения",
                _RELEASE_OPTIONS,
                format_func=lambda t: RELEASE_TYPE_LABELS[t],
            )
            reference_product = st.text_input(
                "Референтный препарат", placeholder="торговое наименование, если выбран"
            )
            additional_strengths = st.text_input(
                "Дополнительные дозировки", placeholder="например, 200 мг, 600 мг"
            )

        with st.expander("Дополнительные характеристики"):
            flag_columns = st.columns(3)
            highly_variable = flag_columns[0].checkbox("Высоковариабельный препарат")
            narrow_ti = flag_columns[1].checkbox("Узкий терапевтический диапазон")
            combination = flag_columns[2].checkbox("Комбинированный препарат")
            new_columns = st.columns(3)
            new_indication = new_columns[0].checkbox("Новое показание")
            new_route = new_columns[1].checkbox("Новый путь введения")
            new_form = new_columns[2].checkbox("Новая лекарственная форма")

        free_text = st.text_area(
            "Дополнительная информация",
            placeholder=(
                "Особенности разработки, планируемый дизайн исследований, вопросы "
                "к регуляторной стратегии…"
            ),
            height=110,
        )

        submitted = st.form_submit_button(
            "Провести регуляторный анализ", type="primary", use_container_width=True
        )

    if not submitted:
        if "assessment_result" in st.session_state:
            render_assessment(st.session_state.assessment_result)
        return

    profile = ProductProfile(
        inn=inn.strip(),
        dosage_form=dosage_form.strip(),
        strength=strength.strip(),
        route_of_administration=route.strip(),
        product_type=product_type,
        release_type=release_type,
        reference_product=reference_product.strip(),
        additional_strengths=additional_strengths.strip(),
        combination_product=combination or None,
        new_indication=new_indication or None,
        new_route=new_route or None,
        new_dosage_form=new_form or None,
        highly_variable=highly_variable or None,
        narrow_therapeutic_index=narrow_ti or None,
        free_text=free_text.strip(),
    )

    if not any(
        [profile.inn, profile.dosage_form, profile.strength, profile.free_text]
    ):
        st.error("Заполните хотя бы МНН или описание препарата.")
        return

    with st.spinner("Поиск нормативных оснований и формирование оценки…"):
        result = pipeline.assess_product(profile)
    st.session_state.assessment_result = result
    render_assessment(result)
