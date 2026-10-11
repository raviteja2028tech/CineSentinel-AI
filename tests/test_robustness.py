"""Missing artifacts, CSV validation, dataset tooling and import checks.

None of these need the trained models or the real dataset.
"""

import importlib
import importlib.util
import py_compile
import sys
from pathlib import Path

import pandas as pd
import pytest

from cinesentinel import predict
from cinesentinel.app_utils import best_model_key, build_metrics_frame
from cinesentinel.predict import summarize_consensus, validate_review_column

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script(relative_path: str):
    """Import a file from scripts/ or experiments/ as a module."""
    path = PROJECT_ROOT / relative_path
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(path.parent))


# ------------------------------------------------------- missing artifacts


def test_no_models_available_when_vectorizer_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(predict, "VECTORIZER_PATH", tmp_path / "missing.joblib")
    assert predict.available_models() == []
    assert predict.artifacts_available() is False


def test_missing_model_files_are_reported_not_assumed(monkeypatch, tmp_path):
    vectorizer = tmp_path / "tfidf_vectorizer.joblib"
    vectorizer.write_bytes(b"placeholder")
    paths = {key: tmp_path / f"{key}.joblib" for key in predict.MODEL_PATHS}
    (tmp_path / "logistic_regression.joblib").write_bytes(b"placeholder")
    monkeypatch.setattr(predict, "VECTORIZER_PATH", vectorizer)
    monkeypatch.setattr(predict, "MODEL_PATHS", paths)

    assert predict.available_models() == ["logistic_regression"]
    assert predict.artifacts_available() is False


def test_metrics_frame_and_best_model_with_no_results():
    assert build_metrics_frame(None).empty
    assert build_metrics_frame({}).empty
    assert best_model_key(None) is None
    assert best_model_key({}) is None


def test_best_model_is_derived_from_results():
    results = {"a": {"accuracy": 0.7}, "b": {"accuracy": 0.9}, "c": {"accuracy": 0.8}}
    assert best_model_key(results) == "b"


# ---------------------------------------------------------------- consensus


def _result(label: str, positive_probability: float) -> dict:
    return {
        "display_name": "m",
        "label": label,
        "confidence": max(positive_probability, 1 - positive_probability),
        "probabilities": {"positive": positive_probability, "negative": 1 - positive_probability},
    }


def test_consensus_flags_disagreement():
    results = {
        "a": _result("positive", 0.9),
        "b": _result("positive", 0.6),
        "c": _result("negative", 0.4),
    }
    consensus = summarize_consensus(results)
    assert consensus["label"] == "positive"
    assert consensus["votes"] == {"positive": 2, "negative": 1}
    assert consensus["unanimous"] is False


def test_consensus_tie_falls_back_to_mean_probability():
    results = {"a": _result("positive", 0.55), "b": _result("negative", 0.1)}
    consensus = summarize_consensus(results)
    assert consensus["votes"] == {"positive": 1, "negative": 1}
    assert consensus["label"] == "negative"


# ------------------------------------------------------------ CSV validation


def test_validate_review_column_clean_input():
    df = pd.DataFrame({"review": ["a really great film overall", "not a good movie at all"]})
    check = validate_review_column(df, "review")
    assert check["n_rows"] == 2
    assert check["n_missing"] == 0
    assert check["warnings"] == []


def test_validate_review_column_counts_missing_and_blank():
    df = pd.DataFrame({"review": ["a really great film overall", None, "   ", ""]})
    check = validate_review_column(df, "review")
    assert check["n_missing"] == 3
    assert any("missing or blank" in w for w in check["warnings"])


def test_validate_review_column_all_empty():
    df = pd.DataFrame({"review": [None, "", "  "]})
    check = validate_review_column(df, "review")
    assert check["n_missing"] == check["n_rows"] == 3
    assert any("Every row" in w for w in check["warnings"])


def test_validate_review_column_flags_numeric_column():
    df = pd.DataFrame({"id": [1, 2, 3, 4], "review": ["good film here", "bad film here", "ok film here", "fine film here"]})
    assert any("numbers" in w for w in validate_review_column(df, "id")["warnings"])
    assert validate_review_column(df, "review")["warnings"] == []


def test_validate_review_column_flags_very_short_text():
    df = pd.DataFrame({"review": ["good", "bad", "fine"]})
    assert any("fewer than three words" in w for w in validate_review_column(df, "review")["warnings"])


def test_validate_review_column_unknown_column():
    with pytest.raises(KeyError):
        validate_review_column(pd.DataFrame({"a": ["x"]}), "review")


# ----------------------------------------------------------- dataset tooling


def test_tarball_loader_reads_official_directory_layout(tmp_path):
    build_dataset = load_script("scripts/build_dataset.py")
    for folder, texts in (("pos", ["loved it", "great"]), ("neg", ["hated it"])):
        directory = tmp_path / "train" / folder
        directory.mkdir(parents=True)
        for i, text in enumerate(texts):
            (directory / f"{i}_7.txt").write_text(text, encoding="utf-8")

    df = build_dataset.load_split_from_tarball_dir("train", tmp_path)
    assert list(df.columns) == ["review", "sentiment", "split"]
    assert df["sentiment"].value_counts().to_dict() == {"positive": 2, "negative": 1}
    assert set(df["review"]) == {"loved it", "great", "hated it"}
    assert (df["split"] == "train").all()


def test_dataset_stats_counts_duplicates_and_overlap():
    dataset_stats = load_script("scripts/dataset_stats.py")
    train = pd.DataFrame(
        {"review": ["a b c", "a b c", "d e", "f g h i"], "sentiment": ["positive", "positive", "negative", "negative"]}
    )
    test = pd.DataFrame({"review": ["d e", "x y z"], "sentiment": ["positive", "negative"]})

    stats = dataset_stats.compute_stats(train, test)
    assert stats["total_rows"] == 6
    assert stats["train"]["duplicate_rows"] == 1
    assert stats["train"]["labels"] == {"positive": 2, "negative": 2}
    assert stats["test"]["words_max"] == 3
    overlap = stats["cross_split_overlap"]
    assert overlap["test_rows_also_in_train"] == 1
    assert overlap["share_of_test"] == pytest.approx(0.5)
    assert overlap["label_conflicts"] == 1


# ------------------------------------------------------------------ imports


@pytest.mark.parametrize(
    "module",
    [
        "cinesentinel.config",
        "cinesentinel.preprocessing",
        "cinesentinel.features",
        "cinesentinel.train",
        "cinesentinel.evaluate",
        "cinesentinel.predict",
        "cinesentinel.app_utils",
        "cinesentinel.ui",
    ],
)
def test_package_modules_import(module):
    importlib.import_module(module)


def test_all_app_and_script_files_compile():
    files = [
        *sorted((PROJECT_ROOT / "app").rglob("*.py")),
        *sorted((PROJECT_ROOT / "scripts").glob("*.py")),
        *sorted((PROJECT_ROOT / "experiments").glob("*.py")),
    ]
    assert len(files) >= 10
    for path in files:
        py_compile.compile(str(path), doraise=True)


def test_app_has_the_five_expected_pages():
    pages = sorted(p.name for p in (PROJECT_ROOT / "app" / "pages").glob("*.py"))
    assert (PROJECT_ROOT / "app" / "Home.py").exists()
    assert pages == [
        "1_Live_Analyzer.py",
        "2_Model_Comparison_Lab.py",
        "3_Batch_Studio.py",
        "4_Methodology.py",
    ]


# ------------------------------------------------------------- visual theme


def test_supplied_background_images_are_found_and_inlined():
    from cinesentinel import ui

    for name in ui.IMAGES:
        path = ui.image_path(name)
        if path is None:
            pytest.skip(f"background image '{name}' not present")
        before = (path.stat().st_size, path.stat().st_mtime_ns)
        uri = ui.image_data_uri(name)
        assert uri.startswith("data:image/jpeg;base64,")
        # Inlined copy is far smaller than the PNG, and the file is untouched.
        assert len(uri) < path.stat().st_size
        assert (path.stat().st_size, path.stat().st_mtime_ns) == before


def test_missing_background_images_do_not_break_the_theme(monkeypatch, tmp_path):
    from cinesentinel import ui

    monkeypatch.setattr(ui, "IMAGE_DIRS", (tmp_path,))
    ui.image_data_uri.cache_clear()
    try:
        assert ui.image_path("emblem") is None
        assert ui.image_data_uri("emblem") == ""
        assert ui._css_url("") == "none"
    finally:
        ui.image_data_uri.cache_clear()


def test_theme_respects_reduced_motion_and_avoids_inline_variables():
    from cinesentinel import ui

    assert "@media (prefers-reduced-motion: reduce)" in ui._CSS
    assert "prefers-reduced-motion" in ui._FX_SCRIPT
    # Streamlit strips custom properties from inline styles, so the ambient
    # markup must not rely on them.
    assert "style=" not in ui._ambient_html()
    variables = ui._variable_css()
    assert variables.count(".cs-embers>span:nth-child(") == len(ui._EMBERS)
    for label in ("positive", "negative", "neutral"):
        assert f".cs-verdict-{label}" in variables
