"""CineSentinel AI - Live sentiment analyzer (single review)."""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cinesentinel import ui  # noqa: E402
from cinesentinel.app_utils import best_model_key, load_evaluation_results  # noqa: E402
from cinesentinel.config import (  # noqa: E402
    MODEL_CONFIDENCE_NOTES,
    MODEL_DISPLAY_NAMES,
    SENTIMENT_COLORS,
)
from cinesentinel.predict import (  # noqa: E402
    available_models,
    explain_review,
    predict_single,
    summarize_consensus,
)
from cinesentinel.preprocessing import clean_text  # noqa: E402

st.set_page_config(page_title="Live Sentiment Analyzer - CineSentinel AI", page_icon="🎬", layout="wide")
ui.inject_theme()
ui.sidebar_brand()

ui.hero(
    "Live sentiment analyzer",
    "Paste a movie review, choose a model, and see its verdict alongside the other "
    "five models, including where they disagree.",
    eyebrow="Live analysis",
    image="mask",
    # Zoomed onto the mask's web texture at the left of the frame.
    position="0% 30%",
    size="235% auto",
    opacity=0.5,
)

models = available_models()
if not models:
    st.error(
        "Trained models not found. Run the training pipeline "
        "(`scripts/download_data.py` → `scripts/build_dataset.py` → "
        "`python -m cinesentinel.train` → `python -m cinesentinel.evaluate`) first."
    )
    st.stop()

EXAMPLES = {
    "— pick an example —": "",
    "Positive example": (
        "This film completely blew me away. The performances were nuanced, the "
        "cinematography was stunning, and the story kept me engaged from start to "
        "finish. Easily one of the best movies I've seen this year."
    ),
    "Negative example": (
        "What a disappointment. The plot made no sense, the pacing dragged on "
        "forever, and the acting felt wooden throughout. I regret spending two "
        "hours on this."
    ),
    "Mixed example": (
        "The visuals are gorgeous and the lead actor is excellent, but the script "
        "is a mess and the second half is painfully slow. Not terrible, not great."
    ),
}

# Default to the model with the best measured test accuracy, if evaluated.
default_model = best_model_key(load_evaluation_results())
default_index = models.index(default_model) if default_model in models else 0

example_col, model_col = st.columns(2)
with example_col:
    choice = st.selectbox("Load an example (optional)", list(EXAMPLES.keys()))
with model_col:
    selected_model = st.selectbox(
        "Model",
        options=models,
        index=default_index,
        format_func=lambda k: MODEL_DISPLAY_NAMES[k],
        help="The headline verdict comes from this model. The other models are shown below for comparison.",
    )

review_text = st.text_area(
    "Movie review text",
    value=EXAMPLES[choice],
    height=180,
    placeholder="Type or paste a movie review here...",
)
st.caption(f"{len(review_text.split())} words · {len(review_text)} characters")

predict_clicked = st.button("Analyze sentiment", type="primary")

if predict_clicked:
    if not review_text or not review_text.strip():
        st.warning("Please enter some review text first.")
        st.stop()

    with st.spinner("Running inference across all models..."):
        results = predict_single(review_text)
        consensus = summarize_consensus(results)
        contributions = explain_review(review_text, top_n=12)

    if consensus["n_models"] == 0:
        st.warning(
            "This text has no usable words after cleaning (only stop words, digits "
            "or punctuation), so the models cannot score it."
        )
        st.stop()

    chosen = results[selected_model]
    label = chosen["label"]
    votes = consensus["votes"]
    n_models = consensus["n_models"]

    ui.verdict_banner(
        label,
        label.title(),
        f"{chosen['display_name']} estimates a {chosen['confidence'] * 100:.1f}% "
        f"probability that this review is {label}.",
        eyebrow=f"Verdict · {chosen['display_name']}",
    )

    cleaned = clean_text(review_text)
    ui.card_grid(
        [
            ui.kpi_card("Review length", f"{len(review_text.split())} words", f"{len(cleaned.split())} after cleaning"),
            ui.kpi_card(
                "Selected model",
                f"{chosen['confidence'] * 100:.1f}%",
                f"Estimated probability of “{label}”",
                SENTIMENT_COLORS[label],
            ),
            ui.kpi_card(
                "Model votes",
                f"{votes['positive']} positive · {votes['negative']} negative",
                f"Across {n_models} models",
                ui.ACCENT_2,
            ),
            ui.kpi_card(
                "Agreement",
                "Unanimous" if consensus["unanimous"] else "Split",
                f"Majority: {consensus['label']}",
                "#34D399" if consensus["unanimous"] else "#F59E0B",
            ),
        ],
        columns=4,
    )

    if consensus["unanimous"]:
        st.success(f"All {n_models} models agree: {consensus['label']}.")
    else:
        dissenting = [r["display_name"] for r in results.values() if r["label"] != consensus["label"]]
        st.warning(
            f"The models disagree on this review. {votes[consensus['label']]} of {n_models} "
            f"predict {consensus['label']}; dissenting: {', '.join(dissenting)}. "
            "Treat the verdict with caution."
        )

    gauge_col, chart_col = st.columns([2, 3], gap="large")

    with gauge_col:
        ui.section("Selected model", f"{chosen['display_name']}: estimated probability of positive.")
        positive_probability = chosen["probabilities"]["positive"]
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=positive_probability * 100,
                number={"suffix": "%", "font": {"size": 44, "color": SENTIMENT_COLORS[label]}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": ui.MUTED},
                    "bar": {"color": SENTIMENT_COLORS[label], "thickness": 0.3},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 50], "color": "rgba(244,63,94,0.16)"},
                        {"range": [50, 100], "color": "rgba(34,197,94,0.16)"},
                    ],
                    "threshold": {"line": {"color": "#F8FAFC", "width": 2}, "thickness": 0.8, "value": 50},
                },
            )
        )
        ui.style_fig(fig, height=300)
        fig.update_layout(margin=dict(l=40, r=40))
        st.plotly_chart(fig, use_container_width=True, config=ui.PLOTLY_CONFIG)

    with chart_col:
        ui.section("All models", "Estimated class probabilities from each classifier.")
        names = [r["display_name"] for r in results.values()]
        fig = go.Figure()
        for sentiment in ["negative", "positive"]:
            values = [r["probabilities"][sentiment] for r in results.values()]
            fig.add_trace(
                go.Bar(
                    y=names,
                    x=values,
                    name=sentiment.title(),
                    orientation="h",
                    marker_color=SENTIMENT_COLORS[sentiment],
                    # Slivers too narrow to hold a label are left blank (hover still works).
                    text=[f"{v * 100:.1f}%" if v >= 0.08 else "" for v in values],
                    textposition="inside",
                    insidetextanchor="middle",
                    hovertemplate="<b>%{y}</b><br>" + sentiment.title() + ": %{x:.1%}<extra></extra>",
                )
            )
        fig.update_layout(barmode="stack")
        fig.update_xaxes(range=[0, 1], tickformat=".0%")
        fig.update_yaxes(autorange="reversed")
        ui.style_fig(fig, height=300, legend_bottom=True)
        st.plotly_chart(fig, use_container_width=True, config=ui.PLOTLY_CONFIG)

    ui.section("Model verdicts")
    cols = st.columns(3)
    for i, (key, result) in enumerate(results.items()):
        with cols[i % 3]:
            st.metric(
                label=result["display_name"] + (" (selected)" if key == selected_model else ""),
                value=result["label"].title(),
            )
            st.caption(f"{result['confidence'] * 100:.1f}% estimated probability")

    with st.expander("How to read these confidence values"):
        st.markdown(
            "Each percentage is the model's **estimated probability** for the class it "
            "predicted. It is a model output, not a measured chance of being right, and "
            "values from different models are not directly comparable:"
        )
        for key in results:
            st.markdown(f"- **{MODEL_DISPLAY_NAMES[key]}** — {MODEL_CONFIDENCE_NOTES[key]}")

    if contributions:
        ui.section(
            "Influential words",
            "Words and phrases that moved the Logistic Regression score the most "
            "(TF-IDF weight × model coefficient).",
        )
        ordered = sorted(contributions, key=lambda c: c["contribution"])
        fig = go.Figure(
            go.Bar(
                x=[c["contribution"] for c in ordered],
                y=[c["term"] for c in ordered],
                orientation="h",
                marker_color=[
                    SENTIMENT_COLORS["positive"] if c["contribution"] >= 0 else SENTIMENT_COLORS["negative"]
                    for c in ordered
                ],
                hovertemplate="<b>%{y}</b><br>Contribution: %{x:+.3f}<extra></extra>",
            )
        )
        fig.update_xaxes(title_text="← pushes negative · pushes positive →")
        ui.style_fig(fig, height=max(280, 30 * len(ordered) + 90))
        st.plotly_chart(fig, use_container_width=True, config=ui.PLOTLY_CONFIG)
        st.caption(
            "This explanation applies to Logistic Regression only, whichever model is "
            "selected above; the other models weigh the same words differently."
        )

    with st.expander("Show cleaned text fed to the models"):
        st.code(cleaned or "(empty after cleaning)")

ui.footer()
