# CineSentinel AI — IMDb Sentiment Intelligence Lab

**Machine Learning-Based Classification for Sentiment Analysis of IMDb Reviews**
Mini-project for UE24CS352A — Machine Learning.

![CineSentinel AI home page](submission/screenshots/home.png)

## Overview

CineSentinel AI classifies IMDb movie reviews as **positive** or
**negative**. It trains six machine-learning classifiers on the Stanford
Large Movie Review Dataset, evaluates them on the dataset's official
held-out test split, and serves them through an interactive Streamlit
dashboard.

The methodology follows the reference paper *"Machine Learning based
classification for Sentimental analysis of IMDb reviews"* (Wu & Shin,
Stanford CS229), included as `ML_Project_mini_REFERENCE.pdf`.

## Problem statement

Given the text of a movie review, predict whether the reviewer's opinion is
positive or negative (document-level binary sentiment classification), and
compare how well different classifier families do this and at what cost.

## Features

- **Six classifiers** trained on identical TF-IDF features.
- **Live Sentiment Analyzer** — classify one review with a chosen model,
  see all six verdicts, and get a warning when the models disagree.
- **Model Comparison Lab** — accuracy, precision, recall, F1, confusion
  matrices, ROC curves, training time, model details, and a comparison
  with the reference paper.
- **Batch Prediction Studio** — upload a CSV, validate it, score it, and
  download the predictions.
- **Methodology** page — dataset, preprocessing, algorithms, metrics,
  architecture and limitations.
- Every figure in the app is read from metrics files produced by the
  evaluation script. Nothing is hard-coded.

## Visual theme

The dashboard uses a cinematic black / crimson / charcoal / silver theme
with these effects, all defined in `src\cinesentinel\ui.py`:

- hero banners built from the three images in `background-img\`, with a
  slow zoom, a cross-fade on the Home page, and a red light sweep;
- pointer parallax on the hero image and the web lines;
- animated spider-web lines, travelling glints and floating embers;
- 3D tilt, lift and a red highlight on cards; a sheen on buttons;
- a title entrance with a brief glitch; an animated verdict reveal;
- charts that wipe in when they appear.

Notes:

- The images are read from `app\background-img\` if it exists, otherwise
  from `background-img\` at the project root (where they are now). They
  are cropped and re-encoded in memory only; the files are never modified.
  If they are missing, the app still runs, with plain dark banners.
- Images appear only in hero banners and the sidebar. Charts and tables
  sit on solid charcoal panels.
- Motion uses only `transform` and `opacity`. Under the operating system's
  "reduce motion" setting, all animation is switched off.
- The images are third-party film and game artwork. That is fine for a
  local course demo, but check the rights before publishing them in a
  public repository. To leave them out, delete the folder: the app then
  shows plain dark banners.

## Technology stack

| Purpose | Tool |
|---|---|
| Language | Python 3.11 |
| Machine learning | scikit-learn 1.5.2 |
| Text processing | NLTK 3.9.1 |
| Data handling | pandas, NumPy, SciPy |
| Dashboard | Streamlit 1.39, Plotly 5.24 |
| Tests | pytest |

No GPU, paid API or external service is used.

## Dataset

**Stanford Large Movie Review Dataset** (Maas et al., 2011).
Official source: <https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz>

| | Reviews | Positive | Negative |
|---|---|---|---|
| Train | 25,000 | 12,500 | 12,500 |
| Test | 25,000 | 12,500 | 12,500 |

- Positive = rating ≥ 7, negative = rating ≤ 4.
- The official split is used unchanged. The test split is used only for
  final evaluation.
- `scripts\download_data.py` downloads from the HuggingFace mirror
  `stanfordnlp/imdb` because the Stanford server is slow, and falls back to
  the official archive. All 50,000 review–label pairs were compared with
  the official archive and are identical (`scripts\verify_provenance.py`,
  result in `reports\metrics\provenance_check.json`).
- The public dataset contains 96 duplicate rows within train, 199 within
  test, and 123 test reviews (0.49%) that also appear in train. These are
  reported, not removed.

## The six classifiers

| Model | Settings |
|---|---|
| Logistic Regression | L2 penalty; C from {0.01, 0.05, 0.25, 0.5, 1} by 3-fold CV → C = 1 |
| Linear SVM | L2 penalty; same C grid → C = 0.25; Platt-calibrated for probabilities |
| Multinomial Naive Bayes | Laplace smoothing = 1 |
| Random Forest | 100 trees, Gini criterion, no depth limit |
| Gradient Boosting | 270 trees, learning rate 0.1, MSE split criterion |
| Neural network (`MLPClassifier`) | hidden layers (30, 30, 20, 10, 10), logistic activation, Adam, early stopping |

Settings follow §3.3 of the reference paper with one exception: the paper
uses 100 boosting trees. That model underfit (81.07% test accuracy), so the
tree count was chosen on a validation split of the training data. See
"Experiments" below.

## Methodology

1. **Clean** each review: strip HTML, lowercase, remove punctuation and
   digits, remove stop words, lemmatize.
2. **Vectorize** with TF-IDF over unigrams and bigrams (50,000 terms,
   min_df = 5, max_df = 0.9, sublinear TF). The vectorizer is fitted on the
   training split only.
3. **Train** all six models on that one matrix. Model selection uses
   training data only.
4. **Evaluate** once on the 25,000-review test split.

## Measured results

From `python -m cinesentinel.evaluate`, on the official test split. The
positive class is "positive"; precision, recall and F1 are for that class;
Macro-F1 is the mean of both classes' F1.

| Model | Accuracy | Precision | Recall | F1 | Macro-F1 | ROC-AUC | Train time |
|---|---|---|---|---|---|---|---|
| Neural network (MLP) | **0.8948** | 0.8944 | **0.8952** | **0.8948** | **0.8948** | **0.9597** | 238.7 s |
| Linear SVM | 0.8938 | **0.8988** | 0.8874 | 0.8931 | 0.8938 | 0.9594 | 3.4 s |
| Logistic Regression | 0.8922 | 0.8904 | 0.8946 | 0.8925 | 0.8922 | 0.9581 | 8.4 s |
| Multinomial Naive Bayes | 0.8633 | 0.8787 | 0.8430 | 0.8605 | 0.8633 | 0.9386 | 0.1 s |
| Random Forest | 0.8560 | 0.8684 | 0.8391 | 0.8535 | 0.8560 | 0.9329 | 72.4 s |
| Gradient Boosting | 0.8441 | 0.8265 | 0.8710 | 0.8482 | 0.8440 | 0.9260 | 476.7 s |

How to read this:

- The top three are within 0.25 percentage points (about 60 reviews).
  Each model was trained once, so that gap does not establish a ranking.
- The linear models match the neural network at a small fraction of its
  training time.
- Training times are from one run on one laptop and show relative cost only.

Confusion matrices, per-class reports and ROC curves are in
`reports\metrics\evaluation_results.json` and in the dashboard.

## Quick start

The trained models and their metrics are in the repository, so the app runs
without downloading the dataset or training anything.

Windows PowerShell, Python 3.11:

```powershell
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
py -3.11 -m venv venv
venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app\Home.py
```

Then open <http://localhost:8501>.

- The first run downloads the NLTK word lists (about 40 MB) into
  `nltk_data\`, so it needs an internet connection once.
- If PowerShell blocks the activation script, run this once in the same
  window and try again:

  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  ```

- Run commands from the project root so Streamlit picks up the theme in
  `.streamlit\config.toml` and the images in `background-img\`.

## Dataset setup

Only needed to retrain the models or run the experiments. The dataset
(about 210 MB once built) is not in the repository.

```powershell
python scripts\download_data.py     # downloads the dataset (about 41 MB)
python scripts\build_dataset.py     # writes data\processed\train.csv and test.csv
python scripts\dataset_stats.py     # writes reports\metrics\dataset_stats.json
```

Optional — confirm the data matches the official archive (download
`aclImdb_v1.tar.gz` from the official source first):

```powershell
python scripts\verify_provenance.py --tarball C:\path\to\aclImdb_v1.tar.gz
```

## Training and evaluation

```powershell
$env:PYTHONPATH = "src"
python -m cinesentinel.train        # trains any model with no saved file
python -m cinesentinel.evaluate     # writes reports\metrics\
```

- `python -m cinesentinel.train --force` refits the vectorizer and
  retrains all six models (about 14 minutes on the development laptop).
- `python -m cinesentinel.train --models gradient_boosting` retrains
  specific models.

The model files in `models\` are committed (about 56 MB in total), so
these commands are only needed to reproduce the training. They require the
dataset (previous section). The saved models are scikit-learn pickles and
must be loaded with the pinned scikit-learn version in `requirements.txt`.

## Running the application

```powershell
venv\Scripts\Activate.ps1
streamlit run app\Home.py
```

## Tests

```powershell
venv\Scripts\Activate.ps1
pytest
```

93 tests. Those that need trained models or metrics files skip
automatically when the files are absent.

Browser-level check (starts the app, drives it in headless Microsoft Edge,
checks the visual effects and reduced-motion behaviour, saves screenshots):

```powershell
python scripts\capture_screenshots.py
```

Add `--all` to also save whole-page and per-tab captures for review.

## Experiments

Both need the dataset and run on the training split only (20,000 reviews
to fit, 5,000 held out for validation). The test split is not loaded.

```powershell
python experiments\gb_diagnostics.py       # about 8 minutes
python experiments\feature_comparison.py   # about 2 minutes
```

**Gradient Boosting tree count.** A control model trained on shuffled
labels scored 51.0% (chance), so the low score was not a pipeline bug.
Validation accuracy rose from 80.34% at 100 trees to 83.38% at 300; 270 was
selected by a rule fixed in advance. On the test split the model went from
81.07% (100 trees) to 84.41% (270 trees). Both results are kept.

**Feature representations.** Binary, word-count and TF-IDF features with
1-, 1–2- and 1–3-grams, for Logistic Regression, Linear SVM and Naive
Bayes. Validation accuracy ranged from 85.2% to 89.4%. TF-IDF was best for
every classifier and n-gram range, which differs from the paper's finding
that binary features are best. Classifier settings were not re-tuned per
representation, so this is one configuration, not a definitive comparison.

## Rebuilding the report and slides

```powershell
python -m pip install -r requirements-dev.txt
python scripts\capture_screenshots.py
python scripts\build_submission.py
```

This rewrites `submission\CineSentinel_AI_Report.pdf` and
`submission\CineSentinel_AI_Presentation.pptx` from the metrics files. It
overwrites manual edits, so fill in team details last (or edit the
placeholders at the top of `scripts\build_submission.py`).

## Repository structure

```
.
├── app/                               # Streamlit dashboard
│   ├── Home.py                        # Overview
│   └── pages/
│       ├── 1_Live_Analyzer.py
│       ├── 2_Model_Comparison_Lab.py
│       ├── 3_Batch_Studio.py
│       └── 4_Methodology.py
├── src/cinesentinel/                  # The ML package
│   ├── config.py                      # Paths, hyperparameters, model metadata
│   ├── preprocessing.py               # Text cleaning
│   ├── features.py                    # TF-IDF vectorizer
│   ├── train.py                       # Trains the six classifiers
│   ├── evaluate.py                    # Test-split metrics
│   ├── predict.py                     # Inference, consensus, CSV scoring
│   ├── app_utils.py                   # Metrics loading for the dashboard
│   └── ui.py                          # Theme, animations, cards, chart styling
├── scripts/
│   ├── download_data.py               # Download the dataset
│   ├── build_dataset.py               # Build train.csv / test.csv
│   ├── dataset_stats.py               # Dataset statistics and integrity checks
│   ├── verify_provenance.py           # Compare with the official archive
│   ├── capture_screenshots.py         # Browser-level check + screenshots
│   └── build_submission.py            # Build the report PDF and the slides
├── experiments/
│   ├── common.py
│   ├── gb_diagnostics.py              # Gradient Boosting tree-count study
│   └── feature_comparison.py          # Binary / count / TF-IDF comparison
├── tests/                             # 93 pytest tests
├── models/                            # Trained models (committed)
│   ├── tfidf_vectorizer.joblib
│   ├── logistic_regression.joblib
│   ├── linear_svm.joblib
│   ├── multinomial_nb.joblib
│   ├── random_forest.joblib
│   ├── gradient_boosting.joblib
│   ├── mlp_neural_net.joblib
│   └── gradient_boosting_paper100.joblib   # 100-tree reference, not used by the app
├── reports/metrics/                   # Measured results (JSON / CSV)
├── submission/
│   ├── CineSentinel_AI_Report.pdf     # Final report
│   ├── CineSentinel_AI_Report.md      # Its editable source
│   ├── CineSentinel_AI_Presentation.pptx
│   └── screenshots/                   # One capture per page
├── docs/
│   ├── FINAL_REQUIREMENTS.md          # Assignment vs paper vs additions
│   ├── FINAL_AUDIT.md                 # What was verified, and how
│   └── PRESENTATION_OUTLINE.md        # Slide outline and speaker notes
├── background-img/                    # Hero / sidebar images
├── .streamlit/config.toml             # Dashboard theme
├── ML_Project_mini_REFERENCE.pdf      # Reference paper, unmodified
├── requirements.txt
├── requirements-dev.txt               # Extra packages for build_submission.py
├── pytest.ini
├── .gitignore
└── .gitattributes
```

Not in the repository (see `.gitignore`): `venv/`, `data/`, `nltk_data/`,
caches, logs and secrets.

## Limitations

- **Not an exact reproduction of the paper.** The paper uses 3-gram
  features and does not publish its vocabulary size, preprocessing code or
  seeds.
- **One training run per model.** No confidence intervals; small gaps
  between models are not statistically established.
- **Bag-of-words features ignore word order**, so sarcasm and long-range
  negation are handled poorly.
- **Confidence values are model estimates.** Only the SVM is calibrated,
  and values are not comparable between models.
- **Gradient Boosting was evaluated on the test split twice** (100 and 270
  trees). The tree count was selected on validation data, but the decision
  to revisit it was prompted by the first test result.
- **English IMDb reviews only.** The models may not transfer to other
  text.
- The feature-representation experiment covers three classifiers on
  validation data, not all six on the test split.

## References

1. C.-L. Wu and S.-L. Shin, "Machine Learning based classification for
   Sentimental analysis of IMDb reviews," Stanford University, CS229
   project report.
2. A. L. Maas, R. E. Daly, P. T. Pham, D. Huang, A. Y. Ng and C. Potts,
   "Learning Word Vectors for Sentiment Analysis," *Proc. 49th Annual
   Meeting of the ACL: Human Language Technologies*, 2011.
3. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python,"
   *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
