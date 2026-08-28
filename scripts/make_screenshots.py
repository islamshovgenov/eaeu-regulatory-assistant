"""Screenshots of the running app for the defence materials.

Drives the live Streamlit app with Playwright and saves PNGs to
``защита/скриншоты``.  Reproducible: re-run it after any UI change and the
figures in the presentation stay in sync with the actual application.

    streamlit run app/main.py          # in one terminal
    python scripts/make_screenshots.py # in another

Each question starts from a freshly loaded page, so a screenshot shows exactly
one exchange instead of the whole accumulated conversation.

Requires ``pip install playwright && playwright install chromium``.
"""

from __future__ import annotations

import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "защита" / "скриншоты"
APP_URL = "http://localhost:8501"

#: A well-grounded general question — the everyday case.
GENERAL_QUESTION = "Что требуется для регистрации воспроизведённого лекарственного препарата?"
#: A concrete product — shows INN-specific Expert Committee recommendations.
INN_QUESTION = (
    "Нимесулид таблетки 100 мг, воспроизведённый препарат, "
    "немедленное высвобождение, пероральный"
)
#: A substance that is not in the corpus — shows the honest refusal.
REFUSAL_QUESTION = "а есть информация по препарату elagolix"

PLACEHOLDER = "Ваш вопрос по регуляторным требованиям ЕАЭС…"


def _wait_for_answer(page, timeout: int = 150) -> None:
    """Streamlit runs the pipeline asynchronously; wait until it settles."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        busy = page.locator('[data-testid="stStatusWidget"]').count()
        spinner = page.get_by_text("Поиск в нормативной базе").count()
        if not busy and not spinner:
            time.sleep(2.5)
            return
        time.sleep(1.0)
    raise TimeoutError("Ответ не получен за отведённое время")


def _fresh_page(page) -> None:
    page.goto(APP_URL, wait_until="networkidle", timeout=120_000)
    time.sleep(4)


def _ask(page, question: str) -> None:
    box = page.get_by_placeholder(PLACEHOLDER)
    box.click()
    box.fill(question)
    box.press("Enter")
    time.sleep(3)
    _wait_for_answer(page)


def _scroll_to_question(page) -> None:
    """Frame the shot from the user's question, not from mid-answer.

    Streamlit scrolls an inner container, so a full-page screenshot starts
    wherever the app happened to leave the viewport — usually cutting the
    answer in half.
    """
    messages = page.locator('[data-testid="stChatMessage"]')
    if messages.count() >= 2:
        messages.nth(messages.count() - 2).scroll_into_view_if_needed()
        time.sleep(1.5)


def _shot(page, name: str, full: bool = True) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(OUT_DIR / name), full_page=full)
    print(f"  сохранено: {name}")


def main() -> int:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        # Tall viewport on purpose: one shot must hold the question, the answer,
        # the confidence strip and the source list — that is the whole point of
        # the figure.  Streamlit scrolls an inner container, so a short viewport
        # cannot be compensated for with full_page.
        page = browser.new_page(
            viewport={"width": 1600, "height": 1500}, device_scale_factor=2
        )
        page.goto(APP_URL, wait_until="networkidle", timeout=180_000)
        time.sleep(8)  # bootstrap: embedding model + indexes

        print("1. Стартовый экран")
        _shot(page, "01_стартовый_экран.png", full=False)

        print("2. Ответ на общий вопрос со ссылками")
        _ask(page, GENERAL_QUESTION)
        _scroll_to_question(page)
        _shot(page, "02_ответ_со_ссылками.png")

        print("3. Раскрытый разбор «Как получен этот ответ»")
        page.get_by_text("Как получен этот ответ").last.click()
        time.sleep(2)
        _shot(page, "03_как_получен_ответ.png")

        print("4. Раскрытый источник с дословным фрагментом")
        page.locator('[data-testid="stExpander"] summary').last.click()
        time.sleep(2)
        _shot(page, "04_источник_с_фрагментом.png")

        print("5. Ответ по конкретному препарату")
        _fresh_page(page)
        _ask(page, INN_QUESTION)
        _scroll_to_question(page)
        _shot(page, "05_ответ_по_препарату.png")

        print("6. Честный отказ при отсутствии основания")
        _fresh_page(page)
        _ask(page, REFUSAL_QUESTION)
        _scroll_to_question(page)
        _shot(page, "06_честный_отказ.png")

        print("7. Режим регуляторной оценки")
        _fresh_page(page)
        page.get_by_text("Регуляторная оценка", exact=True).click()
        time.sleep(4)
        _shot(page, "07_регуляторная_оценка.png")

        print("8. База знаний")
        page.get_by_text("База знаний", exact=True).click()
        time.sleep(6)
        _shot(page, "08_база_знаний.png")

        browser.close()

    print(f"\nГотово. Скриншоты: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
