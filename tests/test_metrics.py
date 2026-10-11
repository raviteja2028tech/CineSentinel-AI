"""Integrity checks on the saved evaluation results shown in the dashboard."""

import json

import pandas as pd
import pytest

from cinesentinel.app_utils import build_metrics_frame
from cinesentinel.config import METRICS_DIR, MODEL_DISPLAY_NAMES, MODEL_PATHS

RESULTS_PATH = METRICS_DIR / "evaluation_results.json"
COMPARISON_PATH = METRICS_DIR / "comparison.csv"
TEST_SET_SIZE = 25000

pytestmark = pytest.mark.skipif(
    not RESULTS_PATH.exists(),
    reason="No evaluation results - run `python -m cinesentinel.evaluate` first.",
)


@pytest.fixture(scope="module")
def results():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_results_cover_all_six_models(results):
    assert set(results) == set(MODEL_PATHS)


def test_confusion_matrices_cover_the_full_test_set(results):
    for key, metrics in results.items():
        cm = metrics["confusion_matrix"]
        assert cm["tp"] + cm["fp"] + cm["fn"] + cm["tn"] == TEST_SET_SIZE, key
        # Official split is balanced: 12,500 true positives and negatives.
        assert cm["tp"] + cm["fn"] == TEST_SET_SIZE // 2, key
        assert cm["tn"] + cm["fp"] == TEST_SET_SIZE // 2, key


def test_stored_metrics_match_confusion_matrix_counts(results):
    for key, metrics in results.items():
        cm = metrics["confusion_matrix"]
        tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
        assert metrics["accuracy"] == pytest.approx((tp + tn) / TEST_SET_SIZE), key
        assert metrics["positive_precision"] == pytest.approx(tp / (tp + fp)), key
        assert metrics["negative_precision"] == pytest.approx(tn / (tn + fn)), key
        assert metrics["recall"] == pytest.approx(tp / (tp + fn)), key
        assert metrics["specificity"] == pytest.approx(tn / (tn + fp)), key
        assert 0.5 < metrics["roc_auc"] <= 1.0, key


def test_every_model_beats_chance_clearly(results):
    for key, metrics in results.items():
        assert metrics["accuracy"] > 0.75, key


def test_comparison_csv_matches_json(results):
    comparison = pd.read_csv(COMPARISON_PATH).set_index("model_key")
    assert set(comparison.index) == set(results)
    for key, metrics in results.items():
        assert comparison.loc[key, "model"] == MODEL_DISPLAY_NAMES[key]
        assert comparison.loc[key, "accuracy"] == pytest.approx(metrics["accuracy"])
        assert comparison.loc[key, "tp"] == metrics["confusion_matrix"]["tp"]


def test_metrics_frame_is_sorted_and_complete(results):
    frame = build_metrics_frame(results)
    assert len(frame) == 6
    assert frame["accuracy"].is_monotonic_decreasing
    assert frame[["accuracy", "f1", "roc_auc", "recall", "specificity"]].notna().all().all()


def test_metrics_frame_handles_missing_results():
    assert build_metrics_frame(None).empty
