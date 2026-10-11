"""CineSentinel AI - Methodology / about page."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cinesentinel import ui  # noqa: E402
from cinesentinel.app_utils import (  # noqa: E402
    build_metrics_frame,
    load_dataset_stats,
    load_evaluation_results,
    load_training_summary,
)
from cinesentinel.config import (  # noqa: E402
    APP_NAME,
    C_GRID,
    CV_FOLDS,
    GB_PARAMS,
    MLP_PARAMS,
    MODEL_DESCRIPTIONS,
    MODEL_DISPLAY_NAMES,
    MODEL_FAMILIES,
    MODEL_LIMITATIONS,
    NB_ALPHA,
    RF_PARAMS,
    TFIDF_PARAMS,
)
from cinesentinel.preprocessing import clean_text  # noqa: E402

st.set_page_config(page_title="Methodology - CineSentinel AI", page_icon="🎬", layout="wide")
ui.inject_theme()
ui.sidebar_brand()

ui.hero(
    "Methodology",
    "How the project works, from raw reviews to evaluated models: the problem, the "
    "data, the processing steps, the six algorithms, the metrics, and the limits of "
    "what the results show.",
    eyebrow="About the project",
    image="emblem",
    position="30% 60%",
    opacity=0.34,
)

dataset = load_dataset_stats()
eval_results = load_evaluation_results()
training_summary = load_training_summary() or {}
metrics = build_metrics_frame(eval_results, training_summary)

# ---------------------------------------------------------------- problem
ui.section("Problem statement")
st.markdown(
    "Given the text of an IMDb movie review, decide whether the reviewer's opinion "
    "is **positive** or **negative**. This is document-level binary sentiment "
    "classification: one label for the whole review.\n\n"
    "The project follows the methodology of Wu & Shin, *Machine Learning based "
    "classification for Sentimental analysis of IMDb reviews* (Stanford CS229), "
    "which compares six classifier families on this task."
)

# ---------------------------------------------------------------- dataset
ui.section("Dataset", "Stanford Large Movie Review Dataset (Maas et al., 2011).")
if dataset is None:
    st.info("Dataset statistics not found. Run `python scripts\\dataset_stats.py`.")
else:
    overlap = dataset["cross_split_overlap"]
    table = pd.DataFrame(
        [
            {
                "Split": split.title(),
                "Reviews": f"{dataset[split]['rows']:,}",
                "Positive": f"{dataset[split]['labels'].get('positive', 0):,}",
                "Negative": f"{dataset[split]['labels'].get('negative', 0):,}",
                "Median words": f"{dataset[split]['words_median']:.0f}",
                "Mean words": f"{dataset[split]['words_mean']:.0f}",
                "Duplicate rows": dataset[split]["duplicate_rows"],
            }
            for split in ("train", "test")
        ]
    )
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.markdown(
        f"- **Labels:** positive for a rating of 7 or more, negative for 4 or less; "
        "reviews rated 5–6 are not in the dataset.\n"
        f"- **Split:** the official train/test partition is used exactly as published. "
        "The test split is used only for the final evaluation.\n"
        f"- **Source:** official release at `{dataset['source']['official']}`; the files "
        f"used here were downloaded from the mirror at `{dataset['source']['downloaded_from']}`. "
        + (
            "All 50,000 review–label pairs were compared with the official archive and are identical.\n"
            if dataset["source"].get("verified_against_official")
            else "They were not compared with the official archive.\n"
        )
        +
        f"- **Known overlap:** {overlap['test_rows_also_in_train']} test reviews "
        f"({overlap['share_of_test'] * 100:.2f}%) also appear in the training split, with "
        f"{overlap['label_conflicts']} label conflicts. They were left in place, so test "
        "accuracy may be optimistic by a small fraction of a percentage point."
    )

# ---------------------------------------------------------- preprocessing
ui.section("Preprocessing", "Each review is cleaned before it is turned into features.")
steps_col, demo_col = st.columns(2, gap="large")
with steps_col:
    st.markdown(
        "1. Remove HTML line breaks and tags\n"
        "2. Convert to lower case\n"
        "3. Remove punctuation and digits\n"
        "4. Split into words and drop English stop words\n"
        "5. Lemmatize each word to its root form (WordNet)\n\n"
        "The same function cleans training reviews, test reviews, and any text "
        "typed or uploaded in this app, so the models always see text prepared "
        "the same way."
    )
with demo_col:
    sample = "I LOVED this movie!<br />The actors were amazing, and it's one of the 10 best films I have seen."
    st.markdown("**Example**")
    st.code(sample, language=None)
    st.markdown("becomes")
    st.code(clean_text(sample), language=None)

# ------------------------------------------------------------------ tfidf
ui.section("Feature extraction: TF-IDF")
st.markdown(
    "Each cleaned review becomes a vector with one entry per vocabulary term. An entry "
    "is large when the term is frequent in that review (term frequency) and rare "
    "across reviews (inverse document frequency), so distinctive words count for more "
    "than common ones.\n\n"
    f"- Terms are single words and two-word phrases (n-gram range {TFIDF_PARAMS['ngram_range']}), "
    "so a phrase like *not good* is a feature of its own.\n"
    f"- The vocabulary is capped at the {TFIDF_PARAMS['max_features']:,} most frequent terms; "
    f"terms in fewer than {TFIDF_PARAMS['min_df']} reviews or more than "
    f"{TFIDF_PARAMS['max_df']:.0%} of reviews are dropped.\n"
    "- Term frequency is log-scaled (sublinear TF).\n"
    "- The vectorizer is fitted on the **training split only** and then applied to the "
    "test split, so no test-set statistics reach the models.\n"
    "- All six models are trained on this one matrix, which keeps the comparison like-for-like."
)

# ------------------------------------------------------------- algorithms
ui.section("The six algorithms")
SETTINGS = {
    "logistic_regression": f"L2 penalty, tolerance 1e-4, C chosen from {C_GRID} by {CV_FOLDS}-fold cross-validation",
    "linear_svm": f"L2 penalty, tolerance 1e-4, C chosen from {C_GRID} by {CV_FOLDS}-fold cross-validation; Platt-calibrated",
    "multinomial_nb": f"Laplace smoothing = {NB_ALPHA:g}",
    "random_forest": (
        f"{RF_PARAMS['n_estimators']} trees, {RF_PARAMS['criterion'].title()} criterion, "
        f"no depth limit, min. samples to split = {RF_PARAMS['min_samples_split']}"
    ),
    "gradient_boosting": (
        f"{GB_PARAMS['n_estimators']} trees, learning rate {GB_PARAMS['learning_rate']}, "
        f"MSE split criterion, min. samples to split = {GB_PARAMS['min_samples_split']}"
    ),
    "mlp_neural_net": (
        f"hidden layers {MLP_PARAMS['hidden_layer_sizes']}, {MLP_PARAMS['activation']} activation, "
        f"L2 = {MLP_PARAMS['alpha']}, Adam, early stopping"
    ),
}
accuracy_by_key = dict(zip(metrics["model_key"], metrics["accuracy"]))
for key, name in MODEL_DISPLAY_NAMES.items():
    acc = accuracy_by_key.get(key)
    suffix = f" · test accuracy {ui.pct(acc)}" if acc is not None else ""
    with st.expander(f"{name} · {MODEL_FAMILIES[key]}{suffix}"):
        st.markdown(MODEL_DESCRIPTIONS[key])
        st.markdown(f"**Settings:** {SETTINGS[key]}")
        st.markdown(f"**Limitation:** {MODEL_LIMITATIONS[key]}")
st.caption(
    "Settings follow §3.3 of the reference paper, with one documented exception: the "
    "Gradient Boosting tree count was chosen on validation data (see the Experiments tab "
    "of the Model Comparison Lab). Model selection never uses the test split."
)

# ---------------------------------------------------------------- metrics
ui.section("Evaluation metrics", "The positive class is “positive”.")
st.dataframe(
    pd.DataFrame(
        [
            {"Metric": "Accuracy", "Definition": "(TP + TN) / all reviews", "Meaning": "Share of reviews classified correctly"},
            {"Metric": "Precision", "Definition": "TP / (TP + FP)", "Meaning": "Of reviews predicted positive, the share that are positive"},
            {"Metric": "Recall", "Definition": "TP / (TP + FN)", "Meaning": "Of positive reviews, the share that were found"},
            {"Metric": "F1", "Definition": "Harmonic mean of precision and recall", "Meaning": "Single score for the positive class"},
            {"Metric": "Macro-F1", "Definition": "Mean of positive-class and negative-class F1", "Meaning": "Treats both classes equally"},
            {"Metric": "Negative precision", "Definition": "TN / (TN + FN)", "Meaning": "The reference paper's second precision measure"},
            {"Metric": "ROC-AUC", "Definition": "Area under the ROC curve", "Meaning": "Ranking quality across all thresholds"},
        ]
    ),
    use_container_width=True,
    hide_index=True,
)
st.markdown(
    "TP, FP, FN and TN are the confusion-matrix counts: true positives, false "
    "positives, false negatives and true negatives. Because the test split is exactly "
    "balanced, accuracy is a fair summary here."
)

# ----------------------------------------------------------- architecture
ui.section("System architecture", "Offline training produces files; the dashboard only reads them.")
st.graphviz_chart(
    """
    digraph {
        rankdir=LR; bgcolor="transparent"; pad=0.2; nodesep=0.35; ranksep=0.5;
        node [shape=box, style="rounded,filled", fillcolor="#17171B", color="#E11D2E",
              fontcolor="#ECEDEF", fontname="Segoe UI", fontsize=11, margin="0.18,0.1"];
        edge [color="#7C8798", fontcolor="#9BA1AC", fontname="Segoe UI", fontsize=9];

        data  [label="Stanford IMDb dataset\\n25k train / 25k test"];
        clean [label="preprocessing.py\\nclean + lemmatize"];
        feat  [label="features.py\\nTF-IDF vectorizer"];
        train [label="train.py\\nsix classifiers"];
        eval  [label="evaluate.py\\ntest-split metrics"];
        files [label="models/*.joblib\\nreports/metrics/*", color="#C9CED6"];
        pred  [label="predict.py\\ninference helpers"];
        app   [label="Streamlit dashboard\\n5 pages", color="#C9CED6"];

        data -> clean -> feat -> train -> files;
        train -> eval -> files;
        files -> pred -> app;
        files -> app [label="metrics"];
    }
    """,
    use_container_width=True,
)
st.markdown(
    "- **Offline:** `scripts/` download and build the dataset; `cinesentinel.train` fits "
    "the vectorizer and models; `cinesentinel.evaluate` writes the metrics files.\n"
    "- **Online:** the dashboard loads the saved vectorizer and models for prediction and "
    "reads the metrics files for every chart. No figure is typed into the app.\n"
    "- Everything runs locally on a CPU. No GPU, paid API or network service is used at run time."
)

# ------------------------------------------------------------ limitations
ui.section("Limitations")
st.markdown(
    "- **Not an exact reproduction of the paper.** The paper uses 3-gram features and "
    "does not publish its vocabulary size, preprocessing code or random seeds.\n"
    "- **One training run per model.** There are no repeated seeds or confidence "
    "intervals, so small differences between models are not statistically established.\n"
    "- **Bag-of-words features ignore word order.** Sarcasm, negation across several "
    "words, and mixed reviews are handled poorly by every model here.\n"
    "- **Binary labels only.** The dataset has no neutral class; the app's “neutral” is "
    "used only for text with no usable words.\n"
    "- **Confidence values are model estimates**, not calibrated except for the SVM, and "
    "are not comparable between models.\n"
    "- **Domain.** The models were trained on English IMDb reviews and may not transfer "
    "to other languages, short comments, or other kinds of text.\n"
    "- **Dataset overlap.** A small number of test reviews also occur in the training split."
)

ui.section("Future work")
st.markdown(
    "- Repeat training over several random seeds and report confidence intervals.\n"
    "- Tune the tree ensembles further (more and deeper trees) on validation data.\n"
    "- Evaluate the binary and word-count representations for all six models.\n"
    "- Try models that use word order, such as recurrent networks or transformers, as "
    "the reference paper suggests."
)

st.caption(f"{APP_NAME} is a course mini-project for UE24CS352A — Machine Learning.")
ui.footer()
