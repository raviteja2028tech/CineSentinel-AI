"""Shared inference helpers for live prediction and batch scoring."""

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from cinesentinel.config import MODEL_DISPLAY_NAMES, MODEL_PATHS, VECTORIZER_PATH
from cinesentinel.preprocessing import clean_text, ensure_nltk_resources

# Columns appended to a scored dataframe by score_dataframe().
PREDICTION_COLUMNS = ("predicted_sentiment", "confidence")


def available_models() -> list:
    """Keys of the models whose trained artifact exists on disk."""
    if not VECTORIZER_PATH.exists():
        return []
    return [key for key, path in MODEL_PATHS.items() if path.exists()]


def artifacts_available() -> bool:
    """True if the vectorizer and all six trained models exist on disk."""
    return len(available_models()) == len(MODEL_PATHS)


@lru_cache(maxsize=1)
def load_artifacts():
    """Load (and cache) the vectorizer and every trained model on disk."""
    ensure_nltk_resources()
    vectorizer = joblib.load(VECTORIZER_PATH)
    models = {key: joblib.load(MODEL_PATHS[key]) for key in available_models()}
    return vectorizer, models


def predict_single(review_text: str) -> dict:
    """Predict sentiment for one raw review string across all trained models.

    Returns a dict keyed by model_key -> {label, confidence, display_name}.
    """
    vectorizer, models = load_artifacts()
    cleaned = clean_text(review_text)

    if not cleaned:
        return {
            key: {
                "display_name": MODEL_DISPLAY_NAMES[key],
                "label": "neutral",
                "confidence": 0.0,
                "error": "Review text had no usable content after cleaning.",
            }
            for key in models
        }

    X = vectorizer.transform([cleaned])

    results = {}
    for key, model in models.items():
        proba = model.predict_proba(X)[0]
        classes = [str(c) for c in model.classes_]
        pred_idx = int(proba.argmax())
        results[key] = {
            "display_name": MODEL_DISPLAY_NAMES[key],
            "label": classes[pred_idx],
            "confidence": float(proba[pred_idx]),
            "probabilities": {cls: float(p) for cls, p in zip(classes, proba)},
        }
    return results


def summarize_consensus(results: dict) -> dict:
    """Majority vote + mean positive probability over predict_single() output."""
    scored = [r for r in results.values() if "probabilities" in r]
    if not scored:
        return {
            "label": "neutral",
            "votes": {"positive": 0, "negative": 0},
            "mean_positive_probability": None,
            "unanimous": False,
            "n_models": 0,
        }

    votes = {"positive": 0, "negative": 0}
    for r in scored:
        votes[r["label"]] += 1
    mean_pos = float(np.mean([r["probabilities"]["positive"] for r in scored]))

    if votes["positive"] != votes["negative"]:
        label = max(votes, key=votes.get)
    else:
        # Tied vote: fall back to the averaged probability.
        label = "positive" if mean_pos >= 0.5 else "negative"

    return {
        "label": label,
        "votes": votes,
        "mean_positive_probability": mean_pos,
        "unanimous": max(votes.values()) == len(scored),
        "n_models": len(scored),
    }


def explain_review(review_text: str, top_n: int = 10) -> list:
    """Top TF-IDF terms driving the Logistic Regression score for one review.

    Each term's contribution is its TF-IDF weight times the model coefficient
    (positive pushes toward "positive", negative toward "negative"). Returns
    [] if the Logistic Regression model is unavailable or the text is empty.
    """
    vectorizer, models = load_artifacts()
    model = models.get("logistic_regression")
    cleaned = clean_text(review_text)
    if model is None or not cleaned:
        return []

    row = vectorizer.transform([cleaned])
    if row.nnz == 0:
        return []

    # coef_ is oriented toward classes_[1]; flip if that is not "positive".
    sign = 1.0 if str(model.classes_[1]) == "positive" else -1.0
    contributions = row.data * model.coef_[0][row.indices] * sign
    terms = vectorizer.get_feature_names_out()[row.indices]

    order = np.argsort(-np.abs(contributions))[:top_n]
    return [
        {"term": str(terms[i]), "contribution": float(contributions[i])} for i in order
    ]


def predict_batch(review_texts: list, model_key: str = "logistic_regression") -> list:
    """Predict sentiment for a batch of raw review strings using one model."""
    vectorizer, models = load_artifacts()
    model = models[model_key]
    classes = [str(c) for c in model.classes_]

    cleaned_texts = [clean_text(t) for t in review_texts]
    results = [
        {"review": raw, "sentiment": "neutral", "confidence": 0.0} for raw in review_texts
    ]

    scorable = [i for i, cleaned in enumerate(cleaned_texts) if cleaned]
    if scorable:
        X = vectorizer.transform([cleaned_texts[i] for i in scorable])
        probas = model.predict_proba(X)
        for i, proba in zip(scorable, probas):
            pred_idx = int(proba.argmax())
            results[i]["sentiment"] = classes[pred_idx]
            results[i]["confidence"] = float(proba[pred_idx])
    return results


def validate_review_column(df: pd.DataFrame, text_column: str) -> dict:
    """Describe a candidate review column and flag anything suspicious.

    Returns counts plus a list of human-readable warnings. Nothing here
    blocks scoring; rows without usable text are simply labelled "neutral".
    """
    if text_column not in df.columns:
        raise KeyError(f"Column '{text_column}' not found in the uploaded data.")

    as_text = df[text_column].fillna("").astype(str).str.strip()
    n_rows = len(df)
    n_missing = int((as_text == "").sum())
    non_empty = as_text[as_text != ""]
    n_numeric = int(pd.to_numeric(non_empty, errors="coerce").notna().sum())
    median_words = float(non_empty.str.split().str.len().median()) if len(non_empty) else 0.0

    warnings = []
    if n_rows and n_missing == n_rows:
        warnings.append("Every row in this column is empty.")
    elif n_missing:
        warnings.append(
            f"{n_missing:,} row(s) are missing or blank and will be labelled neutral."
        )
    if len(non_empty) and n_numeric / len(non_empty) > 0.5:
        warnings.append(
            "Most values in this column are numbers. Check that this is the review text column."
        )
    elif len(non_empty) and median_words < 3:
        warnings.append(
            "Most values are fewer than three words. The models were trained on "
            "full reviews and are less reliable on very short text."
        )

    return {
        "n_rows": n_rows,
        "n_missing": n_missing,
        "n_numeric": n_numeric,
        "median_words": median_words,
        "warnings": warnings,
    }


def score_dataframe(
    df: pd.DataFrame,
    text_column: str,
    model_key: str = "logistic_regression",
    progress_callback=None,
    chunk_size: int = 250,
) -> tuple[pd.DataFrame, list]:
    """Append predicted_sentiment / confidence columns to a dataframe of reviews.

    Any input column already named like a prediction column is renamed to
    "<name>_original" so the output never has duplicate headers. Returns the
    scored dataframe and the list of columns that were renamed.

    progress_callback(done, total), if given, is called after every chunk.
    """
    if text_column not in df.columns:
        raise KeyError(f"Column '{text_column}' not found in the uploaded data.")

    out = df.reset_index(drop=True).copy()
    renamed = [c for c in out.columns if c in PREDICTION_COLUMNS]
    if renamed:
        out = out.rename(columns={c: f"{c}_original" for c in renamed})
        if text_column in renamed:
            text_column = f"{text_column}_original"

    texts = out[text_column].fillna("").astype(str).tolist()
    predictions = []
    for start in range(0, len(texts), chunk_size):
        predictions.extend(predict_batch(texts[start : start + chunk_size], model_key=model_key))
        if progress_callback is not None:
            progress_callback(len(predictions), len(texts))
    out["predicted_sentiment"] = [p["sentiment"] for p in predictions]
    out["confidence"] = [p["confidence"] for p in predictions]
    return out, renamed
