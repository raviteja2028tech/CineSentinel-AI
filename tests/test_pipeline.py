"""Checks on the six-model configuration, trainers and evaluation maths.

These run without the trained artifacts or the dataset: models are fit on a
tiny synthetic corpus, so they exercise the code paths, not model quality.
Nothing measured here is an evaluation result.
"""

import json

import joblib
import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

from cinesentinel import config, train
from cinesentinel.evaluate import evaluate_model
from cinesentinel.features import build_vectorizer

EXPECTED_MODELS = {
    "logistic_regression",
    "linear_svm",
    "multinomial_nb",
    "random_forest",
    "gradient_boosting",
    "mlp_neural_net",
}

EXPECTED_TYPES = {
    "logistic_regression": LogisticRegression,
    "linear_svm": CalibratedClassifierCV,
    "multinomial_nb": MultinomialNB,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "mlp_neural_net": MLPClassifier,
}


@pytest.fixture(scope="module")
def tiny_dataset():
    positive = ["great wonderful loved brilliant film", "loved great acting superb story", "wonderful brilliant superb movie"]
    negative = ["awful boring hated terrible film", "terrible awful plot waste time", "boring hated waste dreadful movie"]
    texts = (positive + negative) * 10
    labels = np.array((["positive"] * 3 + ["negative"] * 3) * 10)
    X = build_vectorizer(min_df=1, max_df=1.0).fit_transform(texts)
    return X, labels


def test_all_six_models_registered_everywhere():
    for mapping in (
        config.MODEL_PATHS,
        config.MODEL_DISPLAY_NAMES,
        config.MODEL_SHORT_NAMES,
        config.MODEL_DESCRIPTIONS,
        config.MODEL_COLORS,
        config.MODEL_CONFIDENCE_NOTES,
        config.MODEL_LIMITATIONS,
        config.MODEL_FAMILIES,
        config.PAPER_TFIDF_RESULTS,
        train.TRAINERS,
    ):
        assert set(mapping) == EXPECTED_MODELS


def test_model_paths_are_distinct_joblib_files_in_models_dir():
    paths = list(config.MODEL_PATHS.values())
    assert len(set(paths)) == 6
    for path in paths + [config.VECTORIZER_PATH]:
        assert path.parent == config.MODELS_DIR
        assert path.suffix == ".joblib"


def test_hyperparameters_follow_reference_paper():
    assert config.C_GRID == [0.01, 0.05, 0.25, 0.5, 1.0]
    assert config.NB_ALPHA == 1.0
    assert config.RF_PARAMS["n_estimators"] == 100
    assert config.RF_PARAMS["criterion"] == "gini"
    assert config.RF_PARAMS["max_depth"] is None
    assert config.RF_PARAMS["min_samples_split"] == 2
    assert config.GB_PARAMS["criterion"] == "squared_error"
    assert config.GB_PARAMS["learning_rate"] == 0.1
    assert config.GB_PARAMS["min_samples_split"] == 2
    assert config.MLP_PARAMS["hidden_layer_sizes"] == (30, 30, 20, 10, 10)
    assert config.MLP_PARAMS["activation"] == "logistic"
    assert config.MLP_PARAMS["alpha"] == 0.0001
    assert config.MLP_PARAMS["early_stopping"] is True
    assert config.MLP_PARAMS["solver"] == "adam"


def test_gradient_boosting_tree_count_matches_validation_selection():
    # The one deliberate departure from the paper: the tree count comes from
    # the validation experiment, and config must agree with what it selected.
    assert config.GB_PAPER_N_ESTIMATORS == 100
    diagnostics_path = config.METRICS_DIR / "gb_diagnostics.json"
    if not diagnostics_path.exists():
        pytest.skip("gb_diagnostics.json not generated")
    with open(diagnostics_path, "r", encoding="utf-8") as f:
        selected = json.load(f)["selected_n_estimators"]
    assert config.GB_PARAMS["n_estimators"] == selected


def test_tfidf_configuration_is_unigram_bigram():
    assert config.TFIDF_PARAMS["ngram_range"] == (1, 2)
    assert config.TFIDF_PARAMS["max_features"] == 50000


@pytest.mark.parametrize("model_key", sorted(EXPECTED_MODELS))
def test_every_trainer_fits_saves_and_reloads(tiny_dataset, tmp_path, model_key):
    X, y = tiny_dataset
    model, info = train.TRAINERS[model_key](X, y)
    assert isinstance(model, EXPECTED_TYPES[model_key])
    assert info["train_seconds"] >= 0
    assert [str(c) for c in model.classes_] == ["negative", "positive"]

    proba = model.predict_proba(X[:6])
    assert proba.shape == (6, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)

    # Round trip through joblib exactly as train.py saves it.
    path = tmp_path / f"{model_key}.joblib"
    joblib.dump(model, path, compress=3)
    reloaded = joblib.load(path)
    assert np.allclose(reloaded.predict_proba(X[:6]), proba)
    assert list(reloaded.predict(X[:6])) == list(model.predict(X[:6]))


def test_evaluate_model_confusion_matrix_orientation(tiny_dataset):
    X, y = tiny_dataset
    model = MultinomialNB().fit(X, y)

    # Flip three true-positive rows to "negative": the model still predicts
    # positive for them, so they must be counted as false positives.
    y_true = y.copy()
    flipped = np.where(y == "positive")[0][:3]
    y_true[flipped] = "negative"

    metrics = evaluate_model(model, X, y_true)
    cm = metrics["confusion_matrix"]
    assert (cm["tp"], cm["fp"], cm["fn"], cm["tn"]) == (27, 3, 0, 30)
    assert cm["matrix"] == [[27, 0], [3, 30]]
    assert metrics["positive_precision"] == pytest.approx(27 / 30)
    assert metrics["negative_precision"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(57 / 60)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["specificity"] == pytest.approx(30 / 33)
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert len(metrics["roc_curve"]["fpr"]) == len(metrics["roc_curve"]["tpr"])


def test_evaluate_model_f1_and_macro_f1(tiny_dataset):
    X, y = tiny_dataset
    model = MultinomialNB().fit(X, y)
    y_true = y.copy()
    y_true[np.where(y == "positive")[0][:3]] = "negative"

    metrics = evaluate_model(model, X, y_true)
    # tp=27 fp=3 fn=0 tn=30
    f1_positive = 2 * 27 / (2 * 27 + 3 + 0)
    f1_negative = 2 * 30 / (2 * 30 + 0 + 3)
    assert metrics["f1"] == pytest.approx(f1_positive)
    assert metrics["macro_f1"] == pytest.approx((f1_positive + f1_negative) / 2)
    assert metrics["predict_seconds"] >= 0
    assert metrics["predict_ms_per_review"] == pytest.approx(1000 * metrics["predict_seconds"] / 60)
