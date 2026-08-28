"""Streamlit entry point.

Run from the project root::

    streamlit run app/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit puts the script's own directory on sys.path, not the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from app.config import (  # noqa: E402
    INN_VOCABULARY_JSON,
    configure_logging,
    ensure_directories,
    get_settings,
)
from app.db.repository import Repository  # noqa: E402
from app.rag.pipeline import RegulatoryPipeline  # noqa: E402
from app.regulatory.inn import load_vocabulary  # noqa: E402
from app.ui.chat import render_assessment_form, render_chat  # noqa: E402
from app.ui.knowledge_base import render_knowledge_base  # noqa: E402
from app.ui.sidebar import (  # noqa: E402
    MODE_ASSESSMENT,
    MODE_CHAT,
    MODE_KNOWLEDGE_BASE,
    render_sidebar,
)

logger = configure_logging("app.ui")

#: Readability is a functional requirement here: the answers are regulatory
#: prose that a specialist reads end to end, so the page is set as a document
#: (measured line length, real paragraph spacing) rather than as a dashboard.
PAGE_STYLE = """
<style>
    .block-container { padding-top: 2.2rem; max-width: 980px; }
    h1, h2, h3 { letter-spacing: -0.01em; }
    h2 { margin-top: 1.6rem; }
    section[data-testid="stSidebar"] { width: 320px !important; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }

    /* Answer prose: comfortable measure, paragraphs that read as paragraphs. */
    div[data-testid="stChatMessage"] .stMarkdown p,
    div[data-testid="stMarkdownContainer"] > p {
        line-height: 1.7;
        margin-bottom: 0.95rem;
        font-size: 1.02rem;
    }
    div[data-testid="stMarkdownContainer"] li { line-height: 1.65; margin-bottom: 0.3rem; }
    .stMarkdown code { font-size: 0.85em; }

    /* Confidence strip under an answer. */
    .answer-status { display: flex; align-items: center; gap: 10px;
        font-size: 0.85rem; margin: 0.6rem 0 0.2rem 0; }
    .answer-status-note { opacity: 0.65; }

    /* The verbatim regulatory fragment inside a source card. */
    .citation-quote { background: rgba(127,127,127,0.10); padding: 12px 14px;
        border-left: 3px solid #888; border-radius: 4px; white-space: pre-wrap;
        font-size: 0.9rem; line-height: 1.55; margin-bottom: 0.6rem; }

    /* Technical detail must be present but quiet. */
    details, div[data-testid="stExpander"] { border-radius: 6px; }
    div[data-testid="stExpander"] summary { font-size: 0.9rem; }
</style>
"""


@st.cache_resource(show_spinner="Инициализация базы знаний и поисковых индексов…")
def _bootstrap() -> tuple[Repository, RegulatoryPipeline]:
    """Load heavy, process-wide resources exactly once."""
    ensure_directories()
    repository = Repository()
    load_vocabulary(INN_VOCABULARY_JSON)
    pipeline = RegulatoryPipeline(repository=repository)
    return repository, pipeline


def main() -> None:
    st.set_page_config(
        page_title="EAEU Regulatory Assistant",
        page_icon="⚕️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(PAGE_STYLE, unsafe_allow_html=True)

    settings = get_settings()
    repository, pipeline = _bootstrap()
    mode = render_sidebar(settings, repository, pipeline)

    st.title("EAEU Regulatory Assistant")
    st.caption(
        "Интеллектуальный ассистент регуляторной поддержки: планирование "
        "регистрации лекарственных препаратов в рамках Евразийского "
        "экономического союза"
    )

    if not pipeline.knowledge_base_ready() and mode != MODE_KNOWLEDGE_BASE:
        st.error(
            "Поисковые индексы не построены. Ответы формироваться не будут.\n\n"
            "Запустите загрузку нормативной базы и построение индекса:",
            icon="🚫",
        )
        st.code(
            "python _архив_сборки_базы/ingestion/run_ingestion.py",
            language="powershell",
        )
        return

    if mode == MODE_CHAT:
        render_chat(pipeline)
    elif mode == MODE_ASSESSMENT:
        render_assessment_form(pipeline)
    elif mode == MODE_KNOWLEDGE_BASE:
        render_knowledge_base(repository, pipeline)


# Streamlit executes this file top-to-bottom on every rerun, so `main()` is
# called unconditionally rather than behind an `if __name__ == "__main__"` guard.
main()
