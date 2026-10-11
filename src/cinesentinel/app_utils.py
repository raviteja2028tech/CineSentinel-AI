"""Shared utilities for the Streamlit app (path setup, metrics loading)."""

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cinesentinel.config import (  # noqa: E402
    METRICS_DIR,
    MODEL_DISPLAY_NAMES,
    MODEL_PATHS,
    MODEL_SHORT_NAMES,
)


def load_comparison_table() -> pd.DataFrame | None:
    path = METRICS_DIR / "comparison.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_evaluation_results() -> dict | None:
    path = METRICS_DIR / "evaluation_results.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_training_summary() -> dict | None:
    path = METRICS_DIR / "training_summary.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_json(name: str) -> dict | None:
    path = METRICS_DIR / name
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset_stats() -> dict | None:
    """Output of scripts/dataset_stats.py."""
    return _load_json("dataset_stats.json")


def load_evaluation_metadata() -> dict | None:
    return _load_json("evaluation_metadata.json")


def load_gb_diagnostics() -> dict | None:
    """Output of experiments/gb_diagnostics.py."""
    return _load_json("gb_diagnostics.json")


def load_gb_paper_result() -> dict | None:
    """Test metrics of Gradient Boosting with the paper's 100-tree setting."""
    return _load_json("gb_paper_config_result.json")


def load_feature_comparison() -> pd.DataFrame | None:
    """Output of experiments/feature_comparison.py."""
    path = METRICS_DIR / "feature_comparison.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def best_model_key(eval_results: dict | None) -> str | None:
    """Key of the model with the highest measured test accuracy, if any."""
    if not eval_results:
        return None
    return max(eval_results, key=lambda k: eval_results[k]["accuracy"])


def display_name_for(model_key: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_key, model_key)


def build_metrics_frame(eval_results: dict | None, training_summary: dict | None = None) -> pd.DataFrame:
    """One tidy row per evaluated model, sorted by accuracy (best first).

    Built straight from evaluation_results.json so every number shown in the
    app traces back to a real evaluation run. Metrics missing from an older
    results file come through as NaN rather than being invented.
    """
    columns = [
        "model_key", "model", "short_name", "accuracy", "positive_precision",
        "negative_precision", "recall", "specificity", "f1", "macro_f1", "roc_auc",
        "tp", "fp", "fn", "tn", "train_seconds", "predict_seconds", "predict_ms_per_review",
    ]
    rows = []
    for key, metrics in (eval_results or {}).items():
        cm = metrics["confusion_matrix"]
        report = metrics.get("classification_report", {})
        rows.append(
            {
                "model_key": key,
                "model": MODEL_DISPLAY_NAMES.get(key, key),
                "short_name": MODEL_SHORT_NAMES.get(key, key),
                "accuracy": metrics["accuracy"],
                "positive_precision": metrics["positive_precision"],
                "negative_precision": metrics["negative_precision"],
                "recall": metrics.get("recall", report.get("positive", {}).get("recall")),
                "specificity": metrics.get("specificity", report.get("negative", {}).get("recall")),
                "f1": metrics.get("f1", report.get("positive", {}).get("f1-score")),
                "macro_f1": metrics.get("macro_f1", report.get("macro avg", {}).get("f1-score")),
                "roc_auc": metrics.get("roc_auc"),
                "tp": cm["tp"],
                "fp": cm["fp"],
                "fn": cm["fn"],
                "tn": cm["tn"],
                "train_seconds": (training_summary or {}).get(key, {}).get("train_seconds"),
                "predict_seconds": metrics.get("predict_seconds"),
                "predict_ms_per_review": metrics.get("predict_ms_per_review"),
            }
        )
    frame = pd.DataFrame(rows, columns=columns)
    return frame.sort_values("accuracy", ascending=False).reset_index(drop=True)


def missing_models() -> list:
    """Display names of models that have no trained artifact on disk."""
    return [MODEL_DISPLAY_NAMES[k] for k, p in MODEL_PATHS.items() if not p.exists()]
