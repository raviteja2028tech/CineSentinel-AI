# CineSentinel AI — Final Requirements

**Project:** Machine Learning-Based Classification for Sentiment Analysis of IMDb Reviews
**Application:** CineSentinel AI — IMDb Sentiment Intelligence Lab
**Course:** UE24CS352A — Machine Learning

This document separates three things that are easy to blur: what the
assignment asks for, what the reference paper does, and what this project
added on its own. The last column of each table says where the requirement
is met and how that was checked.

> **Source of section A.** The original assignment handout is not in this
> repository. Section A is written from the requirements as stated by the
> project owner. If the handout differs, the handout wins.

---

## A. Requirements from the assignment

| # | Requirement | Status | Where / how verified |
|---|---|---|---|
| A1 | Classify IMDb movie reviews as positive or negative using machine learning | Met | `src/cinesentinel/`; all six models predict both classes (`tests/test_predict.py`) |
| A2 | Dataset acquisition and exploration | Met | `scripts/download_data.py`, `scripts/build_dataset.py`, `scripts/dataset_stats.py`; Home and Methodology pages |
| A3 | Text preprocessing | Met | `src/cinesentinel/preprocessing.py`; `tests/test_preprocessing.py` |
| A4 | Feature extraction | Met | `src/cinesentinel/features.py` (TF-IDF); `tests/test_features.py` |
| A5 | Multiple classification algorithms | Met | Six classifiers in `src/cinesentinel/train.py` |
| A6 | Evaluation using standard metrics | Met | `src/cinesentinel/evaluate.py`: accuracy, precision, recall, F1, macro-F1, confusion matrix, ROC-AUC |
| A7 | Comparison of model performance | Met | Model Comparison Lab page; `reports/metrics/comparison.csv` |
| A8 | Single-review prediction | Met | Live Sentiment Analyzer page |
| A9 | CSV batch prediction | Met | Batch Prediction Studio page |
| A10 | Interactive visualization | Met | Plotly charts on every dashboard page |
| A11 | Reproducibility and documentation | Met | Fixed seed, pinned `requirements.txt`, README commands, 90 automated tests |
| A12 | Concise write-up (1–2 pages): problem, dataset, approach, implementation, conclusions | Met | `submission/CineSentinel_AI_Report.pdf` (2 pages) |
| A13 | Presentation (about 7–8 slides) | Met | `submission/CineSentinel_AI_Presentation.pptx` (8 slides) |
| A14 | Runs on a Windows laptop with no GPU or paid API | Met | Trained and run on a Windows 11 laptop CPU |

**Not an assignment requirement:** the assignment asks for *multiple*
algorithms. It does not, as far as stated, require exactly six. Six were
implemented to match the reference paper (section B).

**Still to be done by the team:** fill in the team-member names and SRNs.
They are placeholders in the report and on the title slide.

---

## B. Methodology described in the reference paper

Wu & Shin, *"Machine Learning based classification for Sentimental analysis
of IMDb reviews"* (`ML_Project_mini_REFERENCE.pdf`).

| # | What the paper does | This project |
|---|---|---|
| B1 | 50,000 IMDb reviews, half for training and half for testing, balanced classes (§3.1) | **Same.** Official 25,000 / 25,000 split, used unchanged |
| B2 | Remove punctuation, line breaks, numbers and stop words; lowercase; normalize words to their root (§3.2) | **Same steps.** Lemmatization with NLTK WordNet |
| B3 | Four vectorizations discussed: binary, word-count, n-grams, TF-IDF; results reported with 3-grams (§3.2, Table 2) | **Partly.** Main pipeline is TF-IDF with 1–2-grams. Binary, count and TF-IDF with 1-, 1–2- and 1–3-grams are compared in a side experiment for three classifiers only |
| B4 | Logistic regression: L2, tolerance 1e-4, C from {0.01, 0.05, 0.25, 0.5, 1} | **Same** |
| B5 | SVM: L2, tolerance 1e-4, same C grid | **Same**, plus probability calibration (section C) |
| B6 | Naive Bayes: Laplace smoothing 1 | **Same** |
| B7 | Random forest: 100 trees, Gini, no maximum depth, min. samples to split 2 | **Same** |
| B8 | Boosting: 100 trees, MSE criterion, learning rate 0.1, min. samples to split 2 | **Same except the tree count**, which is 270, chosen on validation data (section C) |
| B9 | Deep neural network: hidden layers (30, 30, 20, 10, 10), logistic activation, L2 0.0001, early stopping, Adam | **Same**, using scikit-learn `MLPClassifier` |
| B10 | Metrics: confusion matrix, positive precision, negative precision, accuracy (§4) | **Same**, plus recall, F1, macro-F1 and ROC-AUC |

**This is not an exact reproduction.** The paper does not publish its
vocabulary size, preprocessing code or random seeds, and its headline
results use 3-gram features. Measured accuracies differ from the paper's
TF-IDF rows by between −1.6 and +1.9 percentage points per model.

---

## C. Improvements introduced by this implementation

None of these is required by the assignment or taken from the paper.

| # | Addition | Why |
|---|---|---|
| C1 | One TF-IDF matrix shared by all six models | Makes the comparison like-for-like |
| C2 | Gradient Boosting tree count chosen on a validation split (100 → 270) | The paper's 100-tree model underfits; see `experiments/gb_diagnostics.py`. The test split was not used for the choice |
| C3 | Platt calibration of the Linear SVM | An SVM decision score is not a probability; calibration lets the app show one honestly |
| C4 | Recall, F1, macro-F1, ROC-AUC, training time and inference time | Standard metrics beyond the paper's three |
| C5 | Feature-representation side experiment (binary / count / TF-IDF × three n-gram ranges) | Tests the paper's vectorization claim without changing the main pipeline |
| C6 | Dataset integrity checks: duplicates, train–test overlap, comparison with the official archive | Experimental validity |
| C7 | Five-page Streamlit dashboard | Assignment items A8–A10, built as a product-quality interface |
| C8 | Model disagreement indicator and per-model confidence explanations | Avoids presenting a model output as a certainty |
| C9 | Word-contribution chart for Logistic Regression | Interpretability |
| C10 | CSV validation: empty files, missing values, numeric columns, colliding column names | Robust batch scoring |
| C11 | 90 automated tests, including headless runs of every page | Regression safety |
| C12 | Report and slides generated from the metrics files | Documents cannot drift from the measured results |

---

## D. Experimental-validity rules followed

1. The official train/test split is never altered, reshuffled across the
   boundary, or deduplicated.
2. The TF-IDF vectorizer is fitted on the training split only.
3. All model selection uses training data: 3-fold cross-validation for C,
   a 10% hold-out for the neural network's early stopping, and an 80/20
   fit/validation split for the Gradient Boosting tree count and the
   feature experiment.
4. The test split is used only to report final metrics.
5. No metric in the app, report or slides is typed by hand; all are read
   from `reports/metrics/`.

One disclosure on rule 4: Gradient Boosting was evaluated on the test split
twice, once with the paper's 100 trees (81.07%) and once with the
validation-selected 270 trees (84.41%). The decision to change the tree
count was prompted by the first result, but the new value was selected
using validation data only. Both results are reported.
