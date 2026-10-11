"""CineSentinel AI - Batch prediction studio (CSV in, CSV out)."""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cinesentinel import ui  # noqa: E402
from cinesentinel.app_utils import best_model_key, load_evaluation_results  # noqa: E402
from cinesentinel.config import MODEL_DISPLAY_NAMES, SENTIMENT_COLORS  # noqa: E402
from cinesentinel.predict import (  # noqa: E402
    available_models,
    score_dataframe,
    validate_review_column,
)

st.set_page_config(page_title="Batch Prediction Studio - CineSentinel AI", page_icon="🎬", layout="wide")
ui.inject_theme()
ui.sidebar_brand()

ui.hero(
    "Batch prediction studio",
    "Upload a CSV of reviews, choose a model, and download the same file with a "
    "predicted sentiment and confidence value added to every row.",
    eyebrow="Batch scoring",
    image="descent",
    opacity=0.75,
)

models = available_models()
if not models:
    st.error(
        "Trained models not found. Run the training pipeline first "
        "(see the Home page for exact commands)."
    )
    st.stop()

MAX_ROWS = 5000
SAMPLE_CSV = PROJECT_ROOT / "tests" / "fixtures" / "sample_batch.csv"

upload_col, help_col = st.columns([3, 2], gap="large")
with upload_col:
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
with help_col:
    st.markdown(
        "**What the file needs**\n\n"
        "- One column of review text (any column name)\n"
        "- Other columns are kept untouched in the output\n"
        f"- Up to {MAX_ROWS:,} rows are scored per run"
    )
    if SAMPLE_CSV.exists():
        st.download_button(
            "Download a sample CSV",
            data=SAMPLE_CSV.read_bytes(),
            file_name="sample_reviews.csv",
            mime="text/csv",
        )

if uploaded_file is None:
    st.info("Upload a CSV file with a column of review text to get started.")
    ui.footer()
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except pd.errors.EmptyDataError:
    st.error("This file is empty. Upload a CSV with a header row and at least one review.")
    st.stop()
except UnicodeDecodeError:
    st.error("This file is not UTF-8 text. Re-save it as a UTF-8 CSV and upload it again.")
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not read this file as a CSV: {exc}")
    st.stop()

if df.empty:
    st.warning("The uploaded CSV has a header but no data rows.")
    st.stop()

df.columns = [str(c) for c in df.columns]
if not df.columns.is_unique:
    st.error(
        "This CSV has repeated column names, so the review column cannot be picked "
        "unambiguously. Rename the duplicated columns and upload it again."
    )
    st.stop()

st.success(f"Loaded {len(df):,} rows and {len(df.columns)} column(s): {', '.join(df.columns)}")

ui.section("Data preview", "First five rows of the uploaded file.")
st.dataframe(df.head(5), use_container_width=True, hide_index=True)

# Prefer a column literally named "review", then the text column with the
# longest values, so the default is usually right.
text_like = [c for c in df.columns if df[c].dtype == object]
if "review" in df.columns:
    default_column = "review"
elif text_like:
    default_column = max(text_like, key=lambda c: df[c].fillna("").astype(str).str.len().mean())
else:
    default_column = df.columns[0]

default_model = best_model_key(load_evaluation_results())

pick_col, model_col = st.columns(2)
with pick_col:
    text_column = st.selectbox(
        "Which column contains the review text?",
        options=df.columns.tolist(),
        index=df.columns.tolist().index(default_column),
    )
with model_col:
    model_key = st.selectbox(
        "Model to use for batch scoring",
        options=models,
        index=models.index(default_model) if default_model in models else 0,
        format_func=lambda k: MODEL_DISPLAY_NAMES[k],
    )

if len(df) > MAX_ROWS:
    st.warning(
        f"This file has {len(df):,} rows; only the first {MAX_ROWS:,} "
        "will be scored to keep the app responsive."
    )
    df = df.head(MAX_ROWS)

check = validate_review_column(df, text_column)
for message in check["warnings"]:
    st.warning(message)
all_empty = check["n_missing"] == check["n_rows"]

if st.button("Run batch prediction", type="primary", disabled=all_empty):
    progress = st.progress(0.0, text="Scoring reviews...")

    def report_progress(done: int, total: int) -> None:
        progress.progress(done / total, text=f"Scored {done:,} of {total:,} reviews")

    try:
        output_df, renamed = score_dataframe(
            df, text_column, model_key, progress_callback=report_progress
        )
    except Exception as exc:  # noqa: BLE001
        progress.empty()
        st.error(f"Batch scoring failed: {exc}")
        st.stop()
    progress.empty()
    st.session_state["batch_result"] = {
        "output": output_df,
        "renamed": renamed,
        "model_key": model_key,
        "source": uploaded_file.name,
    }

# Results live in session state so they survive the rerun triggered by the
# download button.
result = st.session_state.get("batch_result")
if result and result["source"] == uploaded_file.name:
    output_df = result["output"]
    model_name = MODEL_DISPLAY_NAMES[result["model_key"]]

    if result["renamed"]:
        st.info(
            f"Renamed existing column(s) {result['renamed']} to `<name>_original` to avoid "
            "clashing with the prediction output columns."
        )

    counts = output_df["predicted_sentiment"].value_counts()
    scored = output_df[output_df["predicted_sentiment"] != "neutral"]
    total = len(output_df)
    ui.section("Results", f"Scored with {model_name}.")
    ui.card_grid(
        [
            ui.kpi_card("Rows processed", f"{total:,}", result["source"]),
            ui.kpi_card(
                "Positive",
                f"{int(counts.get('positive', 0)):,}",
                ui.pct(counts.get("positive", 0) / total, 1),
                SENTIMENT_COLORS["positive"],
            ),
            ui.kpi_card(
                "Negative",
                f"{int(counts.get('negative', 0)):,}",
                ui.pct(counts.get("negative", 0) / total, 1),
                SENTIMENT_COLORS["negative"],
            ),
            ui.kpi_card(
                "Not scored",
                f"{int(counts.get('neutral', 0)):,}",
                "Missing or no usable text; labelled neutral",
                SENTIMENT_COLORS["neutral"],
            ),
        ],
        columns=4,
    )

    donut_col, hist_col = st.columns(2, gap="large")
    with donut_col:
        fig = go.Figure(
            go.Pie(
                labels=[s.title() for s in counts.index],
                values=counts.values,
                hole=0.62,
                marker=dict(colors=[SENTIMENT_COLORS.get(s, ui.MUTED) for s in counts.index]),
                textinfo="label+percent",
                insidetextorientation="horizontal",
                hovertemplate="%{label}: %{value:,} rows<extra></extra>",
            )
        )
        fig.update_layout(showlegend=False)
        ui.style_fig(fig, height=360, title="Sentiment distribution")
        st.plotly_chart(fig, use_container_width=True, config=ui.PLOTLY_CONFIG)
    with hist_col:
        fig = go.Figure()
        for sentiment in ["positive", "negative"]:
            subset = scored[scored["predicted_sentiment"] == sentiment]
            if len(subset):
                fig.add_trace(
                    go.Histogram(
                        x=subset["confidence"],
                        name=sentiment.title(),
                        marker_color=SENTIMENT_COLORS[sentiment],
                        xbins=dict(start=0.5, end=1.0, size=0.025),
                        opacity=0.85,
                    )
                )
        fig.update_layout(barmode="stack")
        fig.update_xaxes(title_text="Estimated probability of the predicted class", tickformat=".0%", range=[0.5, 1.0])
        fig.update_yaxes(title_text="Reviews")
        ui.style_fig(fig, height=360, title="Confidence distribution")
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.0, x=1, xanchor="right"))
        st.plotly_chart(fig, use_container_width=True, config=ui.PLOTLY_CONFIG)

    ui.section("Scored rows", "First 100 rows of the output file.")
    st.dataframe(
        output_df.head(100),
        use_container_width=True,
        hide_index=True,
        column_config={
            "confidence": st.column_config.ProgressColumn(
                "confidence", min_value=0.0, max_value=1.0, format="%.3f"
            )
        },
    )
    st.caption(
        "`confidence` is the model's estimated probability for the predicted class "
        "(0 for rows that could not be scored). It is a model output, not a guarantee."
    )

    st.download_button(
        label="Download predictions as CSV",
        data=output_df.to_csv(index=False).encode("utf-8"),
        file_name="cinesentinel_batch_predictions.csv",
        mime="text/csv",
        type="primary",
    )

ui.footer()
