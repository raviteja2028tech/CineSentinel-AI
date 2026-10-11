"""CineSentinel AI - Home / overview page."""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cinesentinel import ui  # noqa: E402
from cinesentinel.app_utils import (  # noqa: E402
    build_metrics_frame,
    load_dataset_stats,
    load_evaluation_results,
    load_training_summary,
    missing_models,
)
from cinesentinel.config import (  # noqa: E402
    APP_NAME,
    APP_TAGLINE,
    MODEL_DESCRIPTIONS,
    MODEL_DISPLAY_NAMES,
    MODEL_FAMILIES,
    SENTIMENT_COLORS,
    TFIDF_PARAMS,
)
from cinesentinel.predict import available_models  # noqa: E402

st.set_page_config(
    page_title=f"{APP_NAME} — {APP_TAGLINE}",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui.inject_theme()
ui.sidebar_brand()

ui.hero(
    "Sentiment intelligence for movie reviews",
    "Machine-learning classification of IMDb reviews as positive or negative. Six "
    "classifiers are trained on the Stanford Large Movie Review Dataset, compared on "
    "its official held-out test split, and served for live and batch prediction.",
    badges=["TF-IDF · 1–2 grams", *MODEL_DISPLAY_NAMES.values()],
    eyebrow=f"{APP_NAME} · {APP_TAGLINE}",
    image="emblem",
    alt_image="mask",
    large=True,
    opacity=0.62,
)

trained = available_models()
not_trained = missing_models()
if not trained:
    st.warning(
        "Trained model artifacts were not found under `models/`. Run the pipeline first:\n\n"
        "```\npython scripts\\download_data.py\npython scripts\\build_dataset.py\n"
        "python -m cinesentinel.train\npython -m cinesentinel.evaluate\n```"
    )
elif not_trained:
    st.warning(
        f"{len(trained)} of {len(MODEL_DISPLAY_NAMES)} models are trained. Missing: "
        f"{', '.join(not_trained)}. Run `python -m cinesentinel.train` then "
        "`python -m cinesentinel.evaluate` to add them."
    )

eval_results = load_evaluation_results()
training_summary = load_training_summary()
metrics = build_metrics_frame(eval_results, training_summary)
dataset = load_dataset_stats()

# A model counts as verified when it is both loadable from disk and has
# evaluation results on the test split.
verified = [k for k in trained if eval_results and k in eval_results]

cards = [
    ui.kpi_card(
        "Reviews",
        f"{dataset['total_rows']:,}" if dataset else "—",
        f"{dataset['train']['rows']:,} train · {dataset['test']['rows']:,} test"
        if dataset
        else "Run scripts\\dataset_stats.py",
    ),
    ui.kpi_card(
        "Verified models",
        f"{len(verified)} / {len(MODEL_DISPLAY_NAMES)}",
        "Trained, loadable and evaluated",
        ui.ACCENT_2,
    ),
]
if not metrics.empty:
    best = metrics.iloc[0]
    cards += [
        ui.kpi_card("Best accuracy", ui.pct(best["accuracy"]), best["model"], ui.CRIMSON),
        ui.kpi_card("Best model F1", ui.pct(best["f1"]), "Positive class, same model", ui.EMBER),
    ]
else:
    cards += [ui.kpi_card("Best accuracy", "—", "Run the evaluation step", ui.CRIMSON)]
ui.card_grid(cards, columns=len(cards))

left, right = st.columns([3, 2], gap="large")

with left:
    ui.section("Leaderboard", "Accuracy on the 25,000 held-out test reviews.")
    if metrics.empty:
        st.info("No evaluation results yet — run `python -m cinesentinel.evaluate`.")
    else:
        ordered = metrics.sort_values("accuracy")
        fig = go.Figure(
            go.Bar(
                x=ordered["accuracy"],
                y=ordered["model"],
                orientation="h",
                marker=dict(color=[ui.model_color(k) for k in ordered["model_key"]]),
                text=[ui.pct(v) for v in ordered["accuracy"]],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="<b>%{y}</b><br>Accuracy: %{x:.4f}<extra></extra>",
            )
        )
        low = max(0.0, float(ordered["accuracy"].min()) - 0.06)
        fig.update_xaxes(range=[low, min(1.0, float(ordered["accuracy"].max()) + 0.025)], tickformat=".0%")
        ui.style_fig(fig, height=360)
        st.plotly_chart(fig, use_container_width=True, config=ui.PLOTLY_CONFIG)
        st.caption("The accuracy axis is zoomed in to make differences visible; it does not start at zero.")

with right:
    ui.section("Class distribution", "Positive and negative reviews in each official split.")
    if dataset is None:
        st.info("Dataset statistics not found. Run `python scripts\\dataset_stats.py`.")
    else:
        fig = go.Figure()
        for sentiment in ["positive", "negative"]:
            counts = [dataset[split]["labels"].get(sentiment, 0) for split in ("train", "test")]
            fig.add_trace(
                go.Bar(
                    x=["Train", "Test"],
                    y=counts,
                    name=sentiment.title(),
                    marker_color=SENTIMENT_COLORS[sentiment],
                    text=[f"{c:,}" for c in counts],
                    textposition="inside",
                    hovertemplate="%{x} · " + sentiment.title() + ": %{y:,}<extra></extra>",
                )
            )
        fig.update_layout(barmode="group")
        ui.style_fig(fig, height=360, legend_bottom=True)
        st.plotly_chart(fig, use_container_width=True, config=ui.PLOTLY_CONFIG)

if dataset is not None:
    ui.section("Dataset at a glance", "Stanford Large Movie Review Dataset (Maas et al., 2011).")
    overlap = dataset["cross_split_overlap"]
    ui.card_grid(
        [
            ui.kpi_card(
                "Median review length",
                f"{dataset['train']['words_median']:.0f} words",
                f"Mean {dataset['train']['words_mean']:.0f} · longest {dataset['train']['words_max']:,}",
            ),
            ui.kpi_card("Class balance", "50% / 50%", "Positive / negative in both splits", ui.ACCENT_2),
            ui.kpi_card(
                "Duplicate reviews",
                f"{dataset['train']['duplicate_rows']} / {dataset['test']['duplicate_rows']}",
                "Within train / within test",
                ui.EMBER,
            ),
            ui.kpi_card(
                "Train–test overlap",
                f"{overlap['test_rows_also_in_train']} reviews",
                f"{overlap['share_of_test'] * 100:.2f}% of test · {overlap['label_conflicts']} label conflicts",
                ui.STEEL,
            ),
        ],
        columns=4,
    )
    st.caption(
        "Duplicates and overlap are properties of the public dataset. They are reported "
        "here, not removed, so the official split is used exactly as published."
    )

ui.section("Methodology workflow", "The pipeline follows the reference paper, step by step.")
steps = [
    ("Dataset", "50,000 labelled IMDb reviews with the official train/test split kept intact."),
    ("Cleaning", "HTML and punctuation stripped, lowercased, stop words removed, words lemmatized."),
    (
        "Features",
        f"TF-IDF over unigrams and bigrams, {TFIDF_PARAMS['max_features']:,} terms, "
        "fitted on the training split only.",
    ),
    ("Training", "Six classifiers fitted on the same feature matrix; tuning uses training data only."),
    ("Evaluation", "Accuracy, precision, recall, F1 and confusion matrix on the 25,000 test reviews."),
]
ui.card_grid(
    [
        "<div class='cs-card'>"
        f"<div class='cs-title'><span class='cs-step'>{i}</span>{title}</div>"
        f"<div class='cs-body'>{body}</div></div>"
        for i, (title, body) in enumerate(steps, start=1)
    ],
    columns=5,
)

ui.section("Explore the lab")
NAV = [
    ("pages/1_Live_Analyzer.py", "Live Sentiment Analyzer", "Classify a single review with any model and see where the models disagree."),
    ("pages/2_Model_Comparison_Lab.py", "Model Comparison Lab", "Metrics, confusion matrices, ROC curves and training cost for all six models."),
    ("pages/3_Batch_Studio.py", "Batch Prediction Studio", "Score a CSV of reviews and download the predictions."),
    ("pages/4_Methodology.py", "Methodology", "Dataset, preprocessing, algorithms, metrics, architecture and limitations."),
]
for col, (page, title, body) in zip(st.columns(len(NAV)), NAV):
    with col:
        st.markdown(ui.info_card(title, body), unsafe_allow_html=True)
        st.page_link(page, label=f"Open {title}")

ui.section("The six classifiers", "Each card shows the model's measured test accuracy.")
accuracy_by_key = dict(zip(metrics["model_key"], metrics["accuracy"]))
model_cards = []
for key, name in MODEL_DISPLAY_NAMES.items():
    acc = accuracy_by_key.get(key)
    model_cards.append(
        ui.info_card(
            f"{name} · {MODEL_FAMILIES[key]}",
            MODEL_DESCRIPTIONS[key],
            accent=ui.model_color(key),
            footer=f"Test accuracy {ui.pct(acc)}" if acc is not None else "Not evaluated yet",
            bar=acc,
        )
    )
ui.card_grid(model_cards, columns=3)

if training_summary:
    with st.expander("Training details and selected hyperparameters"):
        rows = []
        for key, info in training_summary.items():
            params = {k: v for k, v in info.items() if k not in {"train_seconds", "cv_accuracy"}}
            rows.append(
                {
                    "Model": MODEL_DISPLAY_NAMES.get(key, key),
                    "Training time (s)": round(float(info.get("train_seconds", 0.0)), 1),
                    "Settings": ", ".join(f"{k}={v}" for k, v in params.items()),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

ui.footer()
