import pandas as pd
import pytest

from cinesentinel.config import MODEL_PATHS
from cinesentinel.predict import (
    artifacts_available,
    available_models,
    explain_review,
    predict_batch,
    predict_single,
    score_dataframe,
    summarize_consensus,
)

pytestmark = pytest.mark.skipif(
    not artifacts_available(),
    reason="Trained model artifacts not found under models/ - run the training pipeline first.",
)

ALL_MODELS = set(MODEL_PATHS)

POSITIVE_REVIEW = (
    "An absolute masterpiece. Brilliant performances, stunning visuals, "
    "a perfect, emotionally powerful story. I loved it."
)
NEGATIVE_REVIEW = (
    "A horrible, boring, poorly acted disaster. Total waste of time. "
    "I hated every single minute of this awful film."
)


def test_all_six_models_are_available():
    assert set(available_models()) == ALL_MODELS
    assert len(ALL_MODELS) == 6


def test_predict_single_returns_all_six_models():
    results = predict_single("This movie was absolutely fantastic, I loved every minute!")
    assert set(results.keys()) == ALL_MODELS


def test_predict_single_labels_and_confidence_are_valid():
    results = predict_single("A wonderful, touching, beautifully acted film.")
    for result in results.values():
        assert result["label"] in {"positive", "negative"}
        assert isinstance(result["label"], str)
        assert 0.5 <= result["confidence"] <= 1.0
        assert sum(result["probabilities"].values()) == pytest.approx(1.0)


def test_predict_single_clearly_positive_review():
    results = predict_single(POSITIVE_REVIEW)
    for key, result in results.items():
        assert result["label"] == "positive", key


def test_predict_single_clearly_negative_review():
    results = predict_single(NEGATIVE_REVIEW)
    for key, result in results.items():
        assert result["label"] == "negative", key


def test_predict_single_empty_text_handled_gracefully():
    results = predict_single("")
    assert set(results.keys()) == ALL_MODELS
    for result in results.values():
        assert result["label"] == "neutral"
        assert result["confidence"] == 0.0


def test_consensus_on_clear_reviews():
    positive = summarize_consensus(predict_single(POSITIVE_REVIEW))
    assert positive["label"] == "positive"
    assert positive["unanimous"]
    assert positive["n_models"] == 6
    assert positive["mean_positive_probability"] > 0.5

    negative = summarize_consensus(predict_single(NEGATIVE_REVIEW))
    assert negative["label"] == "negative"
    assert negative["mean_positive_probability"] < 0.5


def test_consensus_on_unscorable_text():
    consensus = summarize_consensus(predict_single("the a an of to"))
    assert consensus["label"] == "neutral"
    assert consensus["n_models"] == 0


def test_explain_review_signs_match_sentiment_words():
    contributions = {c["term"]: c["contribution"] for c in explain_review("great movie but awful ending", top_n=20)}
    assert contributions["great"] > 0
    assert contributions["awful"] < 0
    assert explain_review("") == []


@pytest.mark.parametrize("model_key", sorted(ALL_MODELS))
def test_predict_batch_works_for_every_model(model_key):
    reviews = ["Great movie, loved it!", "Terrible, I hated it.", ""]
    results = predict_batch(reviews, model_key=model_key)
    assert len(results) == 3
    assert [r["review"] for r in results] == reviews
    assert results[0]["sentiment"] in {"positive", "negative"}
    assert results[2]["sentiment"] == "neutral"
    assert results[2]["confidence"] == 0.0


def test_predict_batch_matches_predict_single():
    reviews = [POSITIVE_REVIEW, NEGATIVE_REVIEW]
    batch = predict_batch(reviews, model_key="linear_svm")
    for review, row in zip(reviews, batch):
        single = predict_single(review)["linear_svm"]
        assert row["sentiment"] == single["label"]
        assert row["confidence"] == pytest.approx(single["confidence"])


def test_score_dataframe_appends_prediction_columns():
    df = pd.DataFrame({"id": [1, 2, 3], "review": [POSITIVE_REVIEW, NEGATIVE_REVIEW, None]})
    out, renamed = score_dataframe(df, "review", "logistic_regression")
    assert renamed == []
    assert list(out.columns) == ["id", "review", "predicted_sentiment", "confidence"]
    assert out["predicted_sentiment"].tolist() == ["positive", "negative", "neutral"]
    assert out["id"].tolist() == [1, 2, 3]
    # The caller's dataframe must not be mutated.
    assert list(df.columns) == ["id", "review"]


def test_score_dataframe_renames_colliding_columns():
    # Regression: an uploaded CSV that already had a "confidence" column used
    # to produce duplicate headers in the output.
    df = pd.DataFrame({"review": [POSITIVE_REVIEW], "confidence": ["high"], "predicted_sentiment": ["?"]})
    out, renamed = score_dataframe(df, "review", "multinomial_nb")
    assert set(renamed) == {"confidence", "predicted_sentiment"}
    assert out.columns.is_unique
    assert out.loc[0, "confidence_original"] == "high"
    assert out.loc[0, "predicted_sentiment_original"] == "?"
    assert out.loc[0, "predicted_sentiment"] == "positive"
    assert isinstance(out.loc[0, "confidence"], float)


def test_score_dataframe_csv_round_trip(tmp_path):
    fixture = pd.read_csv("tests/fixtures/sample_batch.csv")
    out, _ = score_dataframe(fixture, "review", "random_forest")
    path = tmp_path / "out.csv"
    out.to_csv(path, index=False)
    reloaded = pd.read_csv(path)
    assert len(reloaded) == len(fixture)
    assert reloaded["predicted_sentiment"].tolist() == out["predicted_sentiment"].tolist()
    assert reloaded["confidence"].tolist() == pytest.approx(out["confidence"].tolist())


def test_score_dataframe_unknown_column_raises():
    with pytest.raises(KeyError):
        score_dataframe(pd.DataFrame({"a": ["x"]}), "review")
