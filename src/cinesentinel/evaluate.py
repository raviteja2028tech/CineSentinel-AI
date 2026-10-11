"""
Evaluate trained models on the official aclImdb test split and generate
genuine metrics (confusion matrix, positive/negative precision, accuracy,
plus standard precision/recall/F1 and ROC-AUC) mirroring the reference
paper's §4.

All numbers here come from actual model predictions on held-out test data —
nothing in this module is hand-authored or fabricated.

Conventions: "positive" is the positive class. precision / recall / f1 are
for the positive class; macro_f1 is the unweighted mean of the positive- and
negative-class F1 scores.
"""

import json
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from cinesentinel.config import (
    METRICS_DIR,
    MODEL_DISPLAY_NAMES,
    MODEL_PATHS,
    VECTORIZER_PATH,
)
from cinesentinel.train import CLEAN_TEST_CSV, load_datasets


ROC_CURVE_POINTS = 200


def evaluate_model(model, X_test, y_test, positive_label="positive", negative_label="negative"):
    start = time.time()
    y_pred = model.predict(X_test)
    predict_seconds = time.time() - start

    labels = [positive_label, negative_label]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    # cm layout: rows=true, cols=pred, given labels=[positive, negative]
    # cm[0,0]=TP, cm[0,1]=FN (true positive row), cm[1,0]=FP, cm[1,1]=TN
    tp = int(cm[0, 0])
    fn = int(cm[0, 1])
    fp = int(cm[1, 0])
    tn = int(cm[1, 1])

    positive_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    negative_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred, labels=labels, output_dict=True)

    # ROC from the positive-class probability; the curve is thinned to a fixed
    # number of points so the JSON stays small.
    pos_idx = [str(c) for c in model.classes_].index(positive_label)
    y_score = model.predict_proba(X_test)[:, pos_idx]
    y_true = (np.asarray(y_test) == positive_label).astype(int)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    keep = np.unique(np.linspace(0, len(fpr) - 1, ROC_CURVE_POINTS).astype(int))

    return {
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.tolist(),
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
        },
        "positive_precision": positive_precision,
        "negative_precision": negative_precision,
        "accuracy": accuracy,
        "classification_report": report,
        "recall": report[positive_label]["recall"],
        "specificity": report[negative_label]["recall"],
        "f1": report[positive_label]["f1-score"],
        "macro_f1": report["macro avg"]["f1-score"],
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "roc_curve": {"fpr": fpr[keep].tolist(), "tpr": tpr[keep].tolist()},
        "predict_seconds": predict_seconds,
        "predict_ms_per_review": 1000.0 * predict_seconds / X_test.shape[0],
    }


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading vectorizer and trained models...")
    vectorizer = joblib.load(VECTORIZER_PATH)

    if not CLEAN_TEST_CSV.exists():
        print("Cleaned test set not found; re-running data loading/cleaning...")
        _, test_df = load_datasets()
    else:
        test_df = pd.read_csv(CLEAN_TEST_CSV)
        test_df = test_df[test_df["clean_review"].str.len() > 0].reset_index(drop=True)

    X_test = vectorizer.transform(test_df["clean_review"])
    y_test = test_df["sentiment"]

    results = {}
    comparison_rows = []

    for model_key, model_path in MODEL_PATHS.items():
        display_name = MODEL_DISPLAY_NAMES[model_key]
        if not model_path.exists():
            print(f"\nSkipping {display_name}: {model_path.name} not found (not trained yet).")
            continue
        print(f"\nEvaluating {display_name}...")
        model = joblib.load(model_path)
        metrics = evaluate_model(model, X_test, y_test)
        results[model_key] = metrics

        print(
            f"  Accuracy={metrics['accuracy']:.4f} | "
            f"Positive precision={metrics['positive_precision']:.4f} | "
            f"Negative precision={metrics['negative_precision']:.4f} | "
            f"F1={metrics['f1']:.4f} | ROC-AUC={metrics['roc_auc']:.4f}"
        )

        comparison_rows.append(
            {
                "model": display_name,
                "accuracy": metrics["accuracy"],
                "positive_precision": metrics["positive_precision"],
                "negative_precision": metrics["negative_precision"],
                "tp": metrics["confusion_matrix"]["tp"],
                "fp": metrics["confusion_matrix"]["fp"],
                "fn": metrics["confusion_matrix"]["fn"],
                "tn": metrics["confusion_matrix"]["tn"],
                "recall": metrics["recall"],
                "specificity": metrics["specificity"],
                "f1": metrics["f1"],
                "macro_f1": metrics["macro_f1"],
                "predict_ms_per_review": metrics["predict_ms_per_review"],
                "roc_auc": metrics["roc_auc"],
                "model_key": model_key,
            }
        )

    metrics_path = METRICS_DIR / "evaluation_results.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull evaluation results saved -> {metrics_path}")

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        "accuracy", ascending=False
    )
    comparison_path = METRICS_DIR / "comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Comparison table saved -> {comparison_path}")

    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_rows": int(X_test.shape[0]),
        "n_features": int(X_test.shape[1]),
        "positive_label": "positive",
        "negative_label": "negative",
        "averaging": (
            "precision, recall and f1 are for the positive class; negative_precision "
            "and specificity are for the negative class; macro_f1 is the unweighted "
            "mean of the two per-class F1 scores"
        ),
        "timing_note": (
            "predict_seconds is one batch prediction over the whole test matrix on "
            "this machine; it excludes text cleaning and vectorization"
        ),
        "models_evaluated": list(results),
        "best_model_by_accuracy": comparison_df.iloc[0]["model_key"] if len(comparison_df) else None,
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
    }
    metadata_path = METRICS_DIR / "evaluation_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Evaluation metadata saved -> {metadata_path}")
    print("\n" + comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
