"""CineSentinel AI - Model comparison lab."""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cinesentinel import ui  # noqa: E402
from cinesentinel.app_utils import (  # noqa: E402
    build_metrics_frame,
    display_name_for,
    load_evaluation_metadata,
    load_evaluation_results,
    load_feature_comparison,
    load_gb_diagnostics,
    load_gb_paper_result,
    load_training_summary,
)
from cinesentinel.config import (  # noqa: E402
    MODEL_DESCRIPTIONS,
    MODEL_FAMILIES,
    MODEL_LIMITATIONS,
    PAPER_TFIDF_RESULTS,
)

st.set_page_config(page_title="Model Comparison Lab - CineSentinel AI", page_icon="🎬", layout="wide")
ui.inject_theme()
ui.sidebar_brand()

ui.hero(
    "Model comparison lab",
    "Every number on this page is computed from real predictions on the official "
    "25,000-review Stanford IMDb test split. Nothing is hand-entered.",
    eyebrow="Evaluation",
    image="emblem",
    position="70% 40%",
    opacity=0.38,
)

eval_results = load_evaluation_results()
if not eval_results:
    st.error(
        "Evaluation results not found. Run `python -m cinesentinel.evaluate` "
        "after training the models."
    )
    st.stop()

training_summary = load_training_summary() or {}
metrics = build_metrics_frame(eval_results, training_summary)
metadata = load_evaluation_metadata()
has_roc = metrics["roc_auc"].notna().all()

REPRESENTATION_NAMES = {"binary": "binary", "count": "word-count", "tfidf": "TF-IDF"}

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "positive_precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "macro_f1": "Macro-F1",
    "negative_precision": "Negative precision",
    "specificity": "Specificity",
    "roc_auc": "ROC-AUC",
}
METRIC_LABELS = {k: v for k, v in METRIC_LABELS.items() if metrics[k].notna().all()}

best = metrics.iloc[0]
runner_up_gap = (best["accuracy"] - metrics.iloc[1]["accuracy"]) if len(metrics) > 1 else None
fastest = metrics.dropna(subset=["train_seconds"]).sort_values("train_seconds").head(1)
cards = [
    ui.kpi_card("Best accuracy", ui.pct(best["accuracy"]), best["model"], ui.CRIMSON),
    ui.kpi_card("Best F1", ui.pct(metrics["f1"].max()), metrics.loc[metrics["f1"].idxmax(), "model"], ui.SILVER),
    ui.kpi_card(
        "Best precision",
        ui.pct(metrics["positive_precision"].max()),
        metrics.loc[metrics["positive_precision"].idxmax(), "model"],
        ui.STEEL,
    ),
]
if not fastest.empty:
    cards.append(
        ui.kpi_card(
            "Fastest to train",
            f"{fastest.iloc[0]['train_seconds']:.1f} s",
            fastest.iloc[0]["model"],
            ui.EMBER,
        )
    )
ui.card_grid(cards, columns=len(cards))

if runner_up_gap is not None:
    n_test = int(best["tp"] + best["fp"] + best["fn"] + best["tn"])
    ui.note(
        f"<b>Best model by measured accuracy: {best['model']}</b>, ahead of "
        f"{metrics.iloc[1]['model']} by {runner_up_gap * 100:.2f} percentage points "
        f"(about {round(runner_up_gap * n_test):,} of {n_test:,} test reviews). "
        "Each model was trained once, so gaps this small should not be read as a reliable ranking."
    )

st.caption(
    "Positive class: “positive”. Precision, recall and F1 are for the positive class; "
    "Macro-F1 is the unweighted mean of the positive- and negative-class F1."
)

tab_overview, tab_cm, tab_curves, tab_models, tab_paper, tab_experiments = st.tabs(
    ["Overview", "Confusion matrices", "ROC and cost", "Model details", "Reference paper", "Experiments"]
)


def grouped_bar(frame: pd.DataFrame, metric_keys: list, height: int = 420) -> go.Figure:
    """One group per model, one bar per metric, zoomed y-axis."""
    palette = [ui.CRIMSON, ui.SILVER, ui.STEEL, ui.EMBER]
    fig = go.Figure()
    for metric_key, color in zip(metric_keys, palette):
        fig.add_trace(
            go.Bar(
                x=frame["short_name"],
                y=frame[metric_key],
                name=METRIC_LABELS[metric_key],
                marker_color=color,
                text=[f"{v * 100:.1f}" for v in frame[metric_key]],
                textposition="outside",
                cliponaxis=False,
                customdata=frame["model"],
                hovertemplate="<b>%{customdata}</b><br>" + METRIC_LABELS[metric_key] + ": %{y:.4f}<extra></extra>",
            )
        )
    low = max(0.0, float(frame[metric_keys].min().min()) - 0.05)
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[low, 1.0], tickformat=".0%")
    ui.style_fig(fig, height=height, legend_bottom=True)
    return fig


with tab_overview:
    ui.section("Leaderboard", "Sorted by test accuracy. Click a column header to re-sort.")
    table = metrics[["model", *METRIC_LABELS.keys(), "train_seconds", "predict_ms_per_review"]].rename(
        columns={
            "model": "Model",
            **METRIC_LABELS,
            "train_seconds": "Train time (s)",
            "predict_ms_per_review": "Inference (ms/review)",
        }
    )
    st.dataframe(
        table.style.format(
            {
                **{label: "{:.4f}" for label in METRIC_LABELS.values()},
                "Train time (s)": "{:.1f}",
                "Inference (ms/review)": "{:.4f}",
            },
            na_rep="—",
        ),
        use_container_width=True,
        hide_index=True,
    )

    ui.section("Accuracy and F1")
    st.plotly_chart(grouped_bar(metrics, ["accuracy", "f1"]), use_container_width=True, config=ui.PLOTLY_CONFIG)

    ui.section("Precision and recall", "Positive class. A gap between the two shows a bias toward one label.")
    st.plotly_chart(
        grouped_bar(metrics, ["positive_precision", "recall"]),
        use_container_width=True,
        config=ui.PLOTLY_CONFIG,
    )
    st.caption("Vertical axes on this tab are zoomed in and do not start at zero.")

with tab_cm:
    ui.section(
        "Confusion matrices",
        "Rows are the true label, columns the predicted label, on 25,000 test reviews.",
    )
    normalize = st.toggle("Show row percentages instead of counts", value=False)
    keys = metrics["model_key"].tolist()
    for start in range(0, len(keys), 3):
        cols = st.columns(3)
        for col, key in zip(cols, keys[start : start + 3]):
            cm_info = eval_results[key]["confusion_matrix"]
            labels = cm_info["labels"]
            matrix = cm_info["matrix"]
            if normalize:
                z = [[v / sum(row) if sum(row) else 0 for v in row] for row in matrix]
                text = [[f"{v * 100:.1f}%" for v in row] for row in z]
            else:
                z = matrix
                text = [[f"{v:,}" for v in row] for row in matrix]
            fig = go.Figure(
                go.Heatmap(
                    z=z,
                    x=[f"Pred. {label.title()}" for label in labels],
                    y=[f"True {label.title()}" for label in labels],
                    text=text,
                    texttemplate="%{text}",
                    textfont=dict(size=17),
                    colorscale=[[0, "#1B1B20"], [1, ui.model_color(key)]],
                    showscale=False,
                    xgap=4,
                    ygap=4,
                    hovertemplate="%{y} · %{x}<br>%{text}<extra></extra>",
                )
            )
            fig.update_yaxes(autorange="reversed")
            ui.style_fig(fig, height=310, title=display_name_for(key))
            with col:
                st.plotly_chart(fig, use_container_width=True, config=ui.PLOTLY_CONFIG)
                st.caption(
                    f"TP {cm_info['tp']:,} · FP {cm_info['fp']:,} · "
                    f"FN {cm_info['fn']:,} · TN {cm_info['tn']:,}"
                )

with tab_curves:
    if has_roc:
        ui.section("ROC curves", "True-positive rate against false-positive rate across all thresholds.")
        fig = go.Figure()
        for _, row in metrics.sort_values("roc_auc", ascending=False).iterrows():
            curve = eval_results[row["model_key"]]["roc_curve"]
            fig.add_trace(
                go.Scatter(
                    x=curve["fpr"],
                    y=curve["tpr"],
                    mode="lines",
                    name=f"{row['model']} (AUC {row['roc_auc']:.3f})",
                    line=dict(color=ui.model_color(row["model_key"]), width=2.5),
                    hovertemplate="FPR %{x:.3f} · TPR %{y:.3f}<extra>" + row["model"] + "</extra>",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                line=dict(color=ui.MUTED, dash="dash", width=1), hoverinfo="skip",
            )
        )
        fig.update_xaxes(title_text="False-positive rate", range=[0, 1])
        fig.update_yaxes(title_text="True-positive rate", range=[0, 1.01])
        ui.style_fig(fig, height=480)
        st.plotly_chart(fig, use_container_width=True, config=ui.PLOTLY_CONFIG)

    timed = metrics.dropna(subset=["train_seconds"]).sort_values("train_seconds")
    if not timed.empty:
        time_col, scatter_col = st.columns(2, gap="large")
        with time_col:
            ui.section("Training time", "Seconds on this machine, log scale.")
            fig = go.Figure(
                go.Bar(
                    x=timed["train_seconds"],
                    y=timed["short_name"],
                    orientation="h",
                    marker_color=[ui.model_color(k) for k in timed["model_key"]],
                    text=[f"{v:.1f} s" for v in timed["train_seconds"]],
                    textposition="outside",
                    cliponaxis=False,
                    customdata=timed["model"],
                    hovertemplate="<b>%{customdata}</b><br>%{x:.1f} s<extra></extra>",
                )
            )
            fig.update_xaxes(type="log", title_text="Training time (seconds)")
            ui.style_fig(fig, height=400)
            fig.update_layout(margin=dict(r=70))
            st.plotly_chart(fig, use_container_width=True, config=ui.PLOTLY_CONFIG)
        with scatter_col:
            ui.section("Accuracy against training cost", "Up and to the left is better.")
            fig = go.Figure()
            for _, row in timed.iterrows():
                fig.add_trace(
                    go.Scatter(
                        x=[row["train_seconds"]],
                        y=[row["accuracy"]],
                        mode="markers+text",
                        text=[row["short_name"]],
                        textposition="top center",
                        cliponaxis=False,
                        marker=dict(size=18, color=ui.model_color(row["model_key"]), line=dict(color="#060607", width=2)),
                        hovertemplate="<b>" + row["model"] + "</b><br>Train time: %{x:.1f} s<br>Accuracy: %{y:.4f}<extra></extra>",
                    )
                )
            fig.update_xaxes(type="log", title_text="Training time (seconds)")
            fig.update_yaxes(title_text="Test accuracy", tickformat=".1%")
            fig.update_layout(showlegend=False)
            ui.style_fig(fig, height=400)
            st.plotly_chart(fig, use_container_width=True, config=ui.PLOTLY_CONFIG)
        st.caption(
            "Timings are from a single run. Logistic Regression and Linear SVM include their "
            "cross-validated grid search over C. Inference time, in the Overview table, is one "
            "batch prediction over the test matrix and excludes text cleaning and vectorization."
        )

with tab_models:
    ui.section("Model details", "What each model is, how it was configured, and where it falls short.")
    HIDDEN = {"train_seconds", "cv_accuracy", "random_state", "n_jobs"}
    for key in metrics["model_key"]:
        row = metrics.set_index("model_key").loc[key]
        info = training_summary.get(key, {})
        settings = ", ".join(f"{k}={v}" for k, v in info.items() if k not in HIDDEN) or "—"
        with st.expander(f"{display_name_for(key)} · {MODEL_FAMILIES[key]} · accuracy {ui.pct(row['accuracy'])}"):
            st.markdown(MODEL_DESCRIPTIONS[key])
            st.markdown(f"**Settings:** `{settings}`")
            if "cv_accuracy" in info:
                st.markdown(
                    f"**Model selection:** C chosen by 3-fold cross-validation on the training "
                    f"split (CV accuracy {float(info['cv_accuracy']):.4f})."
                )
            st.markdown(f"**Limitation:** {MODEL_LIMITATIONS[key]}")
            report = eval_results[key]["classification_report"]
            rows = [
                {
                    "class": cls,
                    "precision": report[cls]["precision"],
                    "recall": report[cls]["recall"],
                    "f1-score": report[cls]["f1-score"],
                    "support": int(report[cls]["support"]),
                }
                for cls in ["positive", "negative"]
                if cls in report
            ]
            st.dataframe(
                pd.DataFrame(rows).style.format(
                    {"precision": "{:.4f}", "recall": "{:.4f}", "f1-score": "{:.4f}"}
                ),
                use_container_width=True,
                hide_index=True,
            )

with tab_paper:
    ui.section(
        "This project against the reference paper",
        "Paper figures are the TF-IDF rows of Table 2 in Wu & Shin, transcribed from the PDF. "
        "Bars labelled “This project” are measured here.",
    )
    PAPER_METRICS = {
        "accuracy": "Accuracy",
        "positive_precision": "Positive precision",
        "negative_precision": "Negative precision",
    }
    paper_metric = st.radio(
        "Metric to compare", list(PAPER_METRICS), horizontal=True, format_func=PAPER_METRICS.get
    )
    compared = metrics[metrics["model_key"].isin(PAPER_TFIDF_RESULTS)]
    paper_values = [PAPER_TFIDF_RESULTS[k][paper_metric] for k in compared["model_key"]]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=compared["short_name"], y=paper_values, name="Reference paper",
            marker_color="rgba(124,135,152,0.55)",
            text=[f"{v:.3f}" for v in paper_values], textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            x=compared["short_name"], y=compared[paper_metric], name="This project",
            marker_color=[ui.model_color(k) for k in compared["model_key"]],
            text=[f"{v:.3f}" for v in compared[paper_metric]], textposition="outside",
        )
    )
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 1.08], tickformat=".0%")
    ui.style_fig(fig, height=440, legend_bottom=True)
    st.plotly_chart(fig, use_container_width=True, config=ui.PLOTLY_CONFIG)

    delta_table = pd.DataFrame(
        {
            "Model": compared["model"].values,
            "Paper": paper_values,
            "This project": compared[paper_metric].values,
        }
    )
    delta_table["Difference"] = delta_table["This project"] - delta_table["Paper"]
    st.dataframe(
        delta_table.style.format({"Paper": "{:.3f}", "This project": "{:.4f}", "Difference": "{:+.4f}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "This is not an exact reproduction. The paper uses 3-gram TF-IDF and does not publish "
        "its vocabulary size, preprocessing code or random seeds; this project uses 1–2-gram "
        "TF-IDF capped at 50,000 terms, and its Gradient Boosting tree count was chosen on "
        "validation data rather than fixed at the paper's 100."
    )

with tab_experiments:
    st.caption(
        "Side experiments run on the training split only: 20,000 reviews to fit, 5,000 held "
        "out for validation. The test split is not used here, so these figures are not "
        "comparable with the test results on the other tabs."
    )

    gb_diag = load_gb_diagnostics()
    gb_paper = load_gb_paper_result()
    ui.section(
        "Gradient Boosting: accuracy by number of trees",
        "Why the paper's 100-tree setting scores low, and how the tree count was chosen.",
    )
    if gb_diag is None:
        st.info("Not run yet — `python experiments\\gb_diagnostics.py`.")
    else:
        stages = pd.DataFrame(gb_diag["stages"])
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=stages["n_estimators"], y=stages["fit_accuracy"], mode="lines+markers",
                name="Fit part (20,000)", line=dict(color=ui.SILVER, width=2, dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=stages["n_estimators"], y=stages["validation_accuracy"], mode="lines+markers",
                name="Validation part (5,000)", line=dict(color=ui.CRIMSON, width=3),
            )
        )
        fig.add_vline(x=gb_diag["selected_n_estimators"], line=dict(color="#F8FAFC", dash="dash", width=1))
        fig.update_xaxes(title_text="Number of trees")
        fig.update_yaxes(title_text="Accuracy", tickformat=".0%")
        ui.style_fig(fig, height=400, legend_bottom=True)
        st.plotly_chart(fig, use_container_width=True, config=ui.PLOTLY_CONFIG)

        at_100 = stages.loc[stages["n_estimators"] == 100, "validation_accuracy"]
        before_last_100 = stages.loc[
            stages["n_estimators"] == gb_diag["max_trees"] - 100, "validation_accuracy"
        ]
        last_100_gain = (
            stages.iloc[-1]["validation_accuracy"] - float(before_last_100.iloc[0])
            if len(before_last_100)
            else float("nan")
        )
        facts = [
            f"A control model fitted on shuffled labels scores {gb_diag['shuffled_label_control_accuracy'] * 100:.1f}% "
            "on validation, i.e. chance, so the pipeline is not leaking or misaligned.",
            f"At {int(stages.iloc[-1]['n_estimators'])} trees, accuracy is "
            f"{stages.iloc[-1]['fit_accuracy'] * 100:.1f}% on the fit part and "
            f"{stages.iloc[-1]['validation_accuracy'] * 100:.1f}% on validation. Validation accuracy "
            f"gained {last_100_gain * 100:.1f} points over the final 100 trees, so extra trees help, "
            "but with diminishing returns.",
            f"The {gb_diag['max_trees']}-tree model uses {gb_diag['features_used']:,} of "
            f"{gb_diag['n_features']:,} features.",
            f"Selected tree count: {gb_diag['selected_n_estimators']} "
            f"(validation accuracy {gb_diag['selected_validation_accuracy'] * 100:.2f}%"
            + (f", against {float(at_100.iloc[0]) * 100:.2f}% at 100 trees)." if len(at_100) else ")."),
        ]
        if gb_diag.get("still_improving_at_max"):
            facts.append(
                f"Validation accuracy was at its highest at {gb_diag['max_trees']} trees, the "
                "largest size tried, so a larger model might do slightly better."
            )
        if gb_paper:
            facts.append(
                "With the paper's 100 trees the model scored "
                f"{gb_paper['test_metrics']['accuracy'] * 100:.2f}% on the test split (kept for reference)."
            )
        st.markdown("\n".join(f"- {fact}" for fact in facts))
        st.caption(f"Selection rule, fixed in advance: {gb_diag['selection_rule']}.")

    features = load_feature_comparison()
    ui.section(
        "Feature representations",
        "Binary, word-count and TF-IDF features with 1-, 1–2- and 1–3-grams (validation accuracy).",
    )
    if features is None:
        st.info("Not run yet — `python experiments\\feature_comparison.py`.")
    else:
        model_names = features["model"].unique().tolist()
        cols = st.columns(len(model_names))
        zmin, zmax = float(features["validation_accuracy"].min()), float(features["validation_accuracy"].max())
        for col, model_name in zip(cols, model_names):
            pivot = (
                features[features["model"] == model_name]
                .pivot(index="representation", columns="ngrams", values="validation_accuracy")
                .reindex(index=["binary", "count", "tfidf"], columns=["1-gram", "1-2-gram", "1-3-gram"])
            )
            fig = go.Figure(
                go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns.tolist(),
                    y=["Binary", "Count", "TF-IDF"],
                    text=[[f"{v * 100:.1f}%" for v in row] for row in pivot.values],
                    texttemplate="%{text}",
                    textfont=dict(size=15),
                    colorscale=[[0, "#1F1215"], [1, "#C2182A"]],
                    zmin=zmin,
                    zmax=zmax,
                    showscale=False,
                    xgap=4,
                    ygap=4,
                )
            )
            fig.update_yaxes(autorange="reversed")
            ui.style_fig(fig, height=300, title=model_name)
            with col:
                st.plotly_chart(fig, use_container_width=True, config=ui.PLOTLY_CONFIG)
        best_row = features.loc[features["validation_accuracy"].idxmax()]
        st.markdown(
            f"- Highest validation accuracy: **{best_row['validation_accuracy'] * 100:.2f}%** "
            f"({best_row['model']}, {REPRESENTATION_NAMES[best_row['representation']]}, {best_row['ngrams']}).\n"
            "- Classifier settings were fixed, not tuned for each representation, and the "
            "vocabulary is capped at 50,000 terms, so this is one configuration rather than "
            "each representation's best case.\n"
            "- The application's pipeline (TF-IDF, 1–2-grams) was not changed by this experiment."
        )
        not_converged = int((~features["converged"]).sum())
        if not_converged:
            st.caption(f"{not_converged} of {len(features)} fits hit the solver's iteration limit before converging.")

if metadata:
    st.caption(
        f"Results generated {metadata['generated_utc'][:16].replace('T', ' ')} UTC with scikit-learn "
        f"{metadata['scikit_learn']} on {metadata['test_rows']:,} test reviews."
    )
ui.footer()
