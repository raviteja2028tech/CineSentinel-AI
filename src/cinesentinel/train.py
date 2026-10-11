"""
Train all six classifiers from the reference paper's §3.3 algorithm sweep:
Logistic Regression, Linear SVM, Multinomial Naive Bayes, Random Forest,
Gradient Boosting, and a deep neural network (MLP).

Hyperparameters follow the reference paper:
  - Logistic Regression: L2 penalty, tol=1e-4, C selected from
    [0.01, 0.05, 0.25, 0.5, 1] via cross-validation.
  - Linear SVM: L2 penalty, tol=1e-4, C selected from the same grid.
  - Multinomial Naive Bayes: Laplace smoothing alpha=1 (fixed).
  - Random Forest: 100 trees, Gini criterion, no max depth,
    min_samples_split=2.
  - Gradient Boosting: MSE criterion, learning_rate=0.1,
    min_samples_split=2. The number of trees is the one exception to the
    paper's settings; see config.GB_PARAMS for how it was chosen.
  - Deep neural network (MLPClassifier): hidden layers (30,30,20,10,10),
    logistic activation, alpha=0.0001, early stopping, Adam solver.

All six models are fit on the same TF-IDF matrix (one shared vectorizer).

Usage:
    python -m cinesentinel.train                  # train only missing models
    python -m cinesentinel.train --force          # retrain everything
    python -m cinesentinel.train --models random_forest mlp_neural_net
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from cinesentinel.config import (
    C_GRID,
    CV_FOLDS,
    GB_PARAMS,
    METRICS_DIR,
    MLP_PARAMS,
    MODEL_PATHS,
    MODELS_DIR,
    NB_ALPHA,
    RANDOM_STATE,
    RF_PARAMS,
    TEST_CSV,
    TRAIN_CSV,
    VECTORIZER_PATH,
)
from cinesentinel.features import build_vectorizer
from cinesentinel.preprocessing import clean_texts, ensure_nltk_resources

CLEAN_TRAIN_CSV = TRAIN_CSV.parent / "train_clean.csv"
CLEAN_TEST_CSV = TEST_CSV.parent / "test_clean.csv"


def _load_or_clean(raw_csv: Path, clean_csv: Path) -> pd.DataFrame:
    if clean_csv.exists():
        print(f"  Loading cached cleaned data from {clean_csv}")
        return pd.read_csv(clean_csv)

    print(f"  Cleaning {raw_csv.name} (this may take a few minutes)...")
    df = pd.read_csv(raw_csv)
    start = time.time()
    df["clean_review"] = clean_texts(df["review"].tolist())
    elapsed = time.time() - start
    print(f"  Cleaned {len(df)} reviews in {elapsed:.1f}s")
    df.to_csv(clean_csv, index=False)
    return df


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Loading and cleaning datasets...")
    train_df = _load_or_clean(TRAIN_CSV, CLEAN_TRAIN_CSV)
    test_df = _load_or_clean(TEST_CSV, CLEAN_TEST_CSV)
    # Drop any rows that cleaned down to empty strings (keeps X/y aligned).
    # This never touches the held-out test labels/text beyond this cleanup,
    # and is applied identically to train/test -- no information crosses
    # the split boundary.
    train_df = train_df[train_df["clean_review"].str.len() > 0].reset_index(drop=True)
    test_df = test_df[test_df["clean_review"].str.len() > 0].reset_index(drop=True)
    return train_df, test_df


def train_logistic_regression(X_train, y_train) -> tuple[LogisticRegression, dict]:
    print("\nTraining Logistic Regression (grid search over C)...")
    base = LogisticRegression(
        penalty="l2", tol=1e-4, max_iter=1000, random_state=RANDOM_STATE
    )
    grid = GridSearchCV(
        base,
        param_grid={"C": C_GRID},
        cv=CV_FOLDS,
        scoring="accuracy",
        n_jobs=-1,
    )
    start = time.time()
    grid.fit(X_train, y_train)
    elapsed = time.time() - start
    print(
        f"  Best C={grid.best_params_['C']} | CV accuracy={grid.best_score_:.4f} "
        f"| trained in {elapsed:.1f}s"
    )
    return grid.best_estimator_, {
        "best_C": grid.best_params_["C"],
        "cv_accuracy": grid.best_score_,
        "train_seconds": elapsed,
    }


def train_linear_svm(X_train, y_train) -> tuple[CalibratedClassifierCV, dict]:
    print("\nTraining Linear SVM (grid search over C)...")
    base = LinearSVC(penalty="l2", tol=1e-4, max_iter=5000, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        base,
        param_grid={"C": C_GRID},
        cv=CV_FOLDS,
        scoring="accuracy",
        n_jobs=-1,
    )
    start = time.time()
    grid.fit(X_train, y_train)
    search_elapsed = time.time() - start
    best_C = grid.best_params_["C"]
    print(
        f"  Best C={best_C} | CV accuracy={grid.best_score_:.4f} "
        f"| grid search in {search_elapsed:.1f}s"
    )

    # LinearSVC has no predict_proba; calibrate the best estimator (Platt
    # scaling) so the app can display a genuine calibrated probability
    # rather than mislabeling the raw decision_function as one.
    print("  Calibrating best SVM for probability estimates (Platt scaling)...")
    calib_start = time.time()
    calibrated = CalibratedClassifierCV(
        LinearSVC(C=best_C, penalty="l2", tol=1e-4, max_iter=5000, random_state=RANDOM_STATE),
        cv=CV_FOLDS,
        method="sigmoid",
    )
    calibrated.fit(X_train, y_train)
    calib_elapsed = time.time() - calib_start
    print(f"  Calibration done in {calib_elapsed:.1f}s")

    return calibrated, {
        "best_C": best_C,
        "cv_accuracy": grid.best_score_,
        "train_seconds": search_elapsed + calib_elapsed,
    }


def train_multinomial_nb(X_train, y_train) -> tuple[MultinomialNB, dict]:
    print(f"\nTraining Multinomial Naive Bayes (alpha={NB_ALPHA})...")
    model = MultinomialNB(alpha=NB_ALPHA)
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"  Trained in {elapsed:.1f}s")
    return model, {"alpha": NB_ALPHA, "train_seconds": elapsed}


def train_random_forest(X_train, y_train) -> tuple[RandomForestClassifier, dict]:
    print(f"\nTraining Random Forest ({RF_PARAMS['n_estimators']} trees)...")
    model = RandomForestClassifier(**RF_PARAMS)
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"  Trained in {elapsed:.1f}s")
    return model, {**RF_PARAMS, "train_seconds": elapsed}


def train_gradient_boosting(X_train, y_train) -> tuple[GradientBoostingClassifier, dict]:
    print(f"\nTraining Gradient Boosting ({GB_PARAMS['n_estimators']} trees)...")
    model = GradientBoostingClassifier(**GB_PARAMS)
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"  Trained in {elapsed:.1f}s")
    return model, {**GB_PARAMS, "train_seconds": elapsed}


def train_mlp(X_train, y_train) -> tuple[MLPClassifier, dict]:
    print(f"\nTraining MLP neural network (hidden layers={MLP_PARAMS['hidden_layer_sizes']})...")
    model = MLPClassifier(**MLP_PARAMS)
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"  Trained in {elapsed:.1f}s | stopped after {model.n_iter_} epochs")
    return model, {
        **{k: (list(v) if isinstance(v, tuple) else v) for k, v in MLP_PARAMS.items()},
        "train_seconds": elapsed,
        "n_iter": model.n_iter_,
        "best_validation_score": model.best_validation_score_,
    }


TRAINERS = {
    "logistic_regression": train_logistic_regression,
    "linear_svm": train_linear_svm,
    "multinomial_nb": train_multinomial_nb,
    "random_forest": train_random_forest,
    "gradient_boosting": train_gradient_boosting,
    "mlp_neural_net": train_mlp,
}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train CineSentinel AI classifiers.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(TRAINERS),
        help="Train exactly these models (default: only models with no saved artifact).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refit the TF-IDF vectorizer and retrain all six models from scratch.",
    )
    args = parser.parse_args(argv)

    if args.force:
        to_train = list(TRAINERS)
    elif args.models:
        to_train = args.models
    else:
        to_train = [key for key in TRAINERS if not MODEL_PATHS[key].exists()]

    if not to_train:
        print("All six model artifacts already exist. Use --force to retrain.")
        return

    ensure_nltk_resources()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    train_df, _ = load_datasets()
    print(f"\nTrain rows: {len(train_df)}")

    # Existing models are tied to the vocabulary they were fit on, so the
    # saved vectorizer is reused unless everything is being retrained.
    if VECTORIZER_PATH.exists() and not args.force:
        print(f"\nReusing fitted TF-IDF vectorizer <- {VECTORIZER_PATH}")
        vectorizer = joblib.load(VECTORIZER_PATH)
        X_train = vectorizer.transform(train_df["clean_review"])
    else:
        if not args.force:
            # No vectorizer on disk: any saved models are unusable without it.
            to_train = list(TRAINERS)
        print("\nFitting TF-IDF vectorizer on training data...")
        vectorizer = build_vectorizer()
        start = time.time()
        X_train = vectorizer.fit_transform(train_df["clean_review"])
        print(f"  fit+transform in {time.time() - start:.1f}s")
        joblib.dump(vectorizer, VECTORIZER_PATH)
        print(f"  Saved vectorizer -> {VECTORIZER_PATH}")
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)} | matrix shape: {X_train.shape}")

    y_train = train_df["sentiment"]

    summary_path = METRICS_DIR / "training_summary.json"
    training_summary = {}
    if summary_path.exists() and not args.force:
        with open(summary_path, "r", encoding="utf-8") as f:
            training_summary = json.load(f)

    for key in to_train:
        model, info = TRAINERS[key](X_train, y_train)
        joblib.dump(model, MODEL_PATHS[key], compress=3)
        training_summary[key] = info
        # Written after every model so a crash mid-run loses nothing.
        ordered = {k: training_summary[k] for k in TRAINERS if k in training_summary}
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(ordered, f, indent=2, default=str)

    print(f"\nTraining summary saved -> {summary_path}")
    print(f"Trained: {', '.join(to_train)}. Run `python -m cinesentinel.evaluate` next.")


if __name__ == "__main__":
    main()
