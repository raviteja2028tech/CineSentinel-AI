"""Headless smoke tests that run each Streamlit page end to end."""

from pathlib import Path

import pytest

from cinesentinel.config import METRICS_DIR, MODEL_DISPLAY_NAMES
from cinesentinel.predict import artifacts_available

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
LIVE = "pages/1_Live_Analyzer.py"
COMPARISON = "pages/2_Model_Comparison_Lab.py"
BATCH = "pages/3_Batch_Studio.py"
METHODOLOGY = "pages/4_Methodology.py"
PAGES = ["Home.py", LIVE, COMPARISON, BATCH, METHODOLOGY]

pytestmark = pytest.mark.skipif(
    not artifacts_available() or not (METRICS_DIR / "evaluation_results.json").exists(),
    reason="App tests need trained models and evaluation results.",
)


def run_page(page: str) -> "AppTest":
    at = AppTest.from_file(str(APP_DIR / page), default_timeout=120)
    at.run()
    return at


def verdict_text(at) -> str:
    return next(m.value for m in at.markdown if "class='cs-verdict " in m.value)


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(page):
    at = run_page(page)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_home_reports_all_models_verified():
    at = run_page("Home.py")
    assert not at.warning
    assert any("Verified models" in m.value and "6 / 6" in m.value for m in at.markdown)


@pytest.mark.parametrize(
    "example, expected", [("Positive example", "Positive"), ("Negative example", "Negative")]
)
def test_live_analyzer_flow(example, expected):
    at = run_page(LIVE)
    at.selectbox[0].select(example).run()
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.metric) == 6
    assert expected in verdict_text(at)


def test_live_analyzer_uses_the_selected_model():
    at = run_page(LIVE)
    at.selectbox[0].select("Positive example").run()
    at.selectbox[1].select("gradient_boosting").run()
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert MODEL_DISPLAY_NAMES["gradient_boosting"] in verdict_text(at)
    selected = [m.label for m in at.metric if "(selected)" in m.label]
    assert selected == [MODEL_DISPLAY_NAMES["gradient_boosting"] + " (selected)"]


def test_live_analyzer_reports_agreement_or_disagreement():
    at = run_page(LIVE)
    at.selectbox[0].select("Positive example").run()
    at.button[0].click().run()
    # Exactly one of the two banners is shown after a prediction.
    agree = [s for s in at.success if "models agree" in s.value]
    disagree = [w for w in at.warning if "disagree" in w.value]
    assert len(agree) + len(disagree) == 1


def test_live_analyzer_empty_input_warns():
    at = run_page(LIVE)
    at.button[0].click().run()
    assert not at.exception
    assert len(at.warning) == 1
    assert len(at.metric) == 0


def test_live_analyzer_stopword_only_input_warns():
    at = run_page(LIVE)
    at.text_area[0].set_value("the a an of to").run()
    at.button[0].click().run()
    assert not at.exception
    assert any("no usable words" in w.value for w in at.warning)
    assert len(at.metric) == 0


def test_comparison_lab_shows_six_confusion_matrices():
    at = run_page(COMPARISON)
    captions = [c.value for c in at.caption if c.value.startswith("TP ")]
    assert len(captions) == 6


def test_comparison_lab_names_best_model_from_results():
    import json

    with open(METRICS_DIR / "evaluation_results.json", "r", encoding="utf-8") as f:
        results = json.load(f)
    best = max(results, key=lambda k: results[k]["accuracy"])
    at = run_page(COMPARISON)
    note = next(m.value for m in at.markdown if "Best model by measured accuracy" in m.value)
    assert MODEL_DISPLAY_NAMES[best] in note


def test_batch_studio_prompts_for_upload():
    at = run_page(BATCH)
    assert any("Upload a CSV file" in i.value for i in at.info)
