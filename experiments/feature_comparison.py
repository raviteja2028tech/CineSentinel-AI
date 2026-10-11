"""
Compare feature representations, as a side experiment to the main pipeline.

The reference paper compares binary, word-count and TF-IDF vectorization.
This script measures all three, each with unigrams, 1-2-grams and
1-3-grams, using the three classifiers that train in seconds.

It does NOT change the application's pipeline (TF-IDF, 1-2-grams) and does
NOT touch the test split: accuracy is measured on a 20% validation part
held out from the training reviews.

Classifier settings are fixed (not tuned per representation), so this
compares representations under one configuration, not their best case.

Output: reports/metrics/feature_comparison.csv

Usage:  python experiments\\feature_comparison.py
"""

import time
import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from common import load_fit_validation_split

from cinesentinel.config import METRICS_DIR, NB_ALPHA, RANDOM_STATE, TFIDF_PARAMS

SHARED = {
    "max_features": TFIDF_PARAMS["max_features"],
    "min_df": TFIDF_PARAMS["min_df"],
    "max_df": TFIDF_PARAMS["max_df"],
}
NGRAM_RANGES = {"1-gram": (1, 1), "1-2-gram": (1, 2), "1-3-gram": (1, 3)}


def build_vectorizer(representation: str, ngram_range: tuple):
    if representation == "binary":
        return CountVectorizer(binary=True, ngram_range=ngram_range, **SHARED)
    if representation == "count":
        return CountVectorizer(ngram_range=ngram_range, **SHARED)
    if representation == "tfidf":
        return TfidfVectorizer(
            ngram_range=ngram_range, sublinear_tf=TFIDF_PARAMS["sublinear_tf"], **SHARED
        )
    raise ValueError(representation)


def build_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE),
        "Linear SVM": LinearSVC(C=1.0, max_iter=5000, random_state=RANDOM_STATE),
        "Multinomial Naive Bayes": MultinomialNB(alpha=NB_ALPHA),
    }


def main() -> None:
    fit_df, val_df = load_fit_validation_split()
    y_fit, y_val = fit_df["sentiment"], val_df["sentiment"]
    print(f"fit rows: {len(fit_df)} | validation rows: {len(val_df)}")

    rows = []
    for representation in ("binary", "count", "tfidf"):
        for ngram_name, ngram_range in NGRAM_RANGES.items():
            vectorizer = build_vectorizer(representation, ngram_range)
            X_fit = vectorizer.fit_transform(fit_df["clean_review"])
            X_val = vectorizer.transform(val_df["clean_review"])
            for model_name, model in build_models().items():
                start = time.time()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(X_fit, y_fit)
                converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
                pred = model.predict(X_val)
                row = {
                    "representation": representation,
                    "ngrams": ngram_name,
                    "model": model_name,
                    "n_features": X_fit.shape[1],
                    "validation_accuracy": accuracy_score(y_val, pred),
                    "validation_macro_f1": f1_score(y_val, pred, average="macro"),
                    "fit_seconds": time.time() - start,
                    "converged": converged,
                }
                rows.append(row)
                print(
                    f"{representation:6s} {ngram_name:8s} {model_name:24s} "
                    f"acc={row['validation_accuracy']:.4f} "
                    f"({row['fit_seconds']:.1f}s{'' if converged else ', not converged'})"
                )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / "feature_comparison.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
