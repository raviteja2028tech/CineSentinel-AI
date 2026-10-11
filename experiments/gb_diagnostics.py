"""
Diagnose Gradient Boosting's low accuracy and choose its number of trees.

Question: is the ~81% test accuracy of the paper-configured model (100
trees) a bug, or is the model simply too small for 50,000 sparse features?

Method (training split only - the test split is never loaded):
  1. Split the training reviews 80/20 into fit / validation parts.
  2. Fit the TF-IDF vectorizer on the fit part only.
  3. Sanity check: fit the same model on shuffled labels. If the pipeline
     were leaking or misaligned, this control would not sit at chance.
  4. Fit one boosting run with MAX_TREES trees and record accuracy on the
     fit and validation parts after every STAGE_STEP trees.
  5. Select the tree count by a rule fixed before looking at any result:
     the smallest count whose validation accuracy is within TOLERANCE of
     the best validation accuracy seen.

Output: reports/metrics/gb_diagnostics.json

Usage:  python experiments\\gb_diagnostics.py
"""

import json
import time

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

from common import load_fit_validation_split

from cinesentinel.config import GB_PARAMS, METRICS_DIR, RANDOM_STATE
from cinesentinel.features import build_vectorizer

MAX_TREES = 300
STAGE_STEP = 10
TOLERANCE = 0.002
CONTROL_TREES = 20


def main() -> None:
    fit_df, val_df = load_fit_validation_split()
    print(f"fit rows: {len(fit_df)} | validation rows: {len(val_df)}")

    vectorizer = build_vectorizer()
    X_fit = vectorizer.fit_transform(fit_df["clean_review"])
    X_val = vectorizer.transform(val_df["clean_review"])
    y_fit = fit_df["sentiment"].to_numpy()
    y_val = val_df["sentiment"].to_numpy()
    print(f"feature matrix: {X_fit.shape}")

    base_params = {k: v for k, v in GB_PARAMS.items() if k != "n_estimators"}

    # Control: labels shuffled, so there is nothing real to learn.
    rng = np.random.default_rng(RANDOM_STATE)
    control = GradientBoostingClassifier(n_estimators=CONTROL_TREES, **base_params)
    control.fit(X_fit, rng.permutation(y_fit))
    control_accuracy = accuracy_score(y_val, control.predict(X_val))
    print(f"shuffled-label control ({CONTROL_TREES} trees): validation accuracy {control_accuracy:.4f}")

    print(f"fitting {MAX_TREES} trees...")
    start = time.time()
    model = GradientBoostingClassifier(n_estimators=MAX_TREES, **base_params)
    model.fit(X_fit, y_fit)
    fit_seconds = time.time() - start
    print(f"  done in {fit_seconds:.1f}s")

    stages = []
    staged = zip(model.staged_predict(X_fit), model.staged_predict(X_val))
    for n_trees, (pred_fit, pred_val) in enumerate(staged, start=1):
        if n_trees % STAGE_STEP == 0:
            stages.append(
                {
                    "n_estimators": n_trees,
                    "fit_accuracy": float(accuracy_score(y_fit, pred_fit)),
                    "validation_accuracy": float(accuracy_score(y_val, pred_val)),
                }
            )
            print(
                f"  {n_trees:4d} trees | fit {stages[-1]['fit_accuracy']:.4f} "
                f"| validation {stages[-1]['validation_accuracy']:.4f}"
            )

    best_val = max(s["validation_accuracy"] for s in stages)
    selected = next(s for s in stages if s["validation_accuracy"] >= best_val - TOLERANCE)

    result = {
        "description": (
            "Gradient Boosting accuracy by number of trees, measured on an 80/20 "
            "fit/validation split of the TRAINING reviews. The test split is not used."
        ),
        "fit_rows": len(fit_df),
        "validation_rows": len(val_df),
        "n_features": int(X_fit.shape[1]),
        "params": {k: v for k, v in base_params.items()},
        "max_trees": MAX_TREES,
        "fit_seconds": fit_seconds,
        "shuffled_label_control_accuracy": float(control_accuracy),
        "features_used": int((model.feature_importances_ > 0).sum()),
        "selection_rule": (
            f"smallest n_estimators with validation accuracy within {TOLERANCE} "
            "of the best validation accuracy"
        ),
        "best_validation_accuracy": best_val,
        "selected_n_estimators": selected["n_estimators"],
        "selected_validation_accuracy": selected["validation_accuracy"],
        "still_improving_at_max": stages[-1]["validation_accuracy"] >= best_val - 1e-12,
        "stages": stages,
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / "gb_diagnostics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nselected n_estimators = {selected['n_estimators']} (validation {selected['validation_accuracy']:.4f})")
    print(f"features used by the {MAX_TREES}-tree model: {result['features_used']} of {X_fit.shape[1]}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
