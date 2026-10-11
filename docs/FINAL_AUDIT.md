# CineSentinel AI — Final Technical Audit

Audit date: 2026-10-11 (final audit, replacing the two earlier ones).

Each item says how it was checked. "Verified" means a command was run or a
file was opened in this audit; it does not mean a claim was copied from an
earlier document.

## Repository preparation (latest; supersedes the sections below where they differ)

The project was tidied for a GitHub submission. No model, metric or
behaviour changed.

| Action | Detail |
|---|---|
| Caches removed | Every `__pycache__/` outside `venv/`, and `.pytest_cache/`. Several held bytecode for pages that were renamed earlier |
| Duplicate removed | `reports/PROJECT_REPORT.md` — byte-identical to `submission/CineSentinel_AI_Report.md` (checked with `cmp`) |
| Intermediate removed | `submission/CineSentinel_AI_Report.html` — only an input to printing the PDF; now written to a temp folder |
| Review captures removed | 15 whole-page and per-tab screenshots. The five per-page captures are kept; the rest are produced only by `capture_screenshots.py --all` and are git-ignored |
| Documentation gathered | `FINAL_AUDIT.md`, `FINAL_REQUIREMENTS.md` and `PRESENTATION_OUTLINE.md` moved to `docs/` |
| Models | Kept on disk and **now committed** (56 MB in total, largest file 35 MB, under GitHub's 50 MB warning and 100 MB limit) so the app runs straight after cloning |
| Personal information | Absolute paths containing a Windows user name removed from the README and this file. The presentation's metadata carried the slide library's default author name; replaced with a neutral value. No API keys, tokens, `.env` or key files exist in the project (pattern scan of every tracked file) |
| Bug fixed | On a fresh clone there is no `nltk_data/`, and the Methodology page called the text cleaner before anything had downloaded it. The cleaner now fetches the word lists on first use |
| `.gitignore` | Rewritten: `venv/`, caches, `data/`, `nltk_data/`, secrets, logs, temp and editor files, review screenshots |
| `.gitattributes` | Added: line-ending normalisation; `.joblib`, `.png`, `.pdf`, `.pptx` marked binary |

Verification after the tidy-up is recorded in the hand-over notes: test
run, browser run, and a simulated fresh clone.

**Git.** The project folder has no repository of its own. It sits inside a
repository rooted at the user's home directory whose remote points to an
unrelated project; from there the project is one untracked folder. Nothing
was staged, committed, pushed, initialised or configured. Upload
instructions are in the hand-over notes.

File locations mentioned in the older sections below use the previous
layout: read `FINAL_AUDIT.md`, `FINAL_REQUIREMENTS.md` and
`reports/PRESENTATION_OUTLINE.md` as `docs/...`, and ignore references to
`reports/PROJECT_REPORT.md`.

## 0. Addendum — cinematic theme (added after the audit below)

The dashboard was restyled (black / crimson / silver) with animations and
the three supplied images. Functionality, models and metrics are unchanged.
Where this addendum and the sections below disagree on the look of the app
or the test count, this addendum is current.

| Check | Result |
|---|---|
| Tests | `pytest`: 93 passed (3 new theme tests) |
| Browser run (`scripts\capture_screenshots.py`) | All five pages, 0 exceptions; live prediction and CSV upload still work |
| Background images | 3 found in `background-img\` (project root, not `app\background-img\`); SHA-256 identical before and after; not moved, renamed or edited |
| Images inlined | As in-memory JPEG copies: about 54, 48 and 8 KB instead of 267, 325 and 46 KB |
| Parallax and card tilt | Respond to a synthetic pointer event in the browser |
| Embers, web threads, hero cross-fade, light sweep | Confirmed running by animation name |
| Idle cost | Main thread about 1% busy with all animations running (headless Edge, software rendering) |
| Reduced motion | With `prefers-reduced-motion: reduce` emulated: 0 running animations, embers hidden |
| Readability | Hero text sits on a near-solid black shade; images are confined to hero banners and the sidebar; charts sit on solid panels. Screenshots of all five pages inspected |

A bug found and fixed while doing this: Streamlit removes CSS custom
properties from inline `style` attributes, which silently disabled the
embers and web threads on the first attempt. They are now set from
stylesheet rules, and the browser check asserts that they animate.

Not verified: the animations were judged from still captures and from the
browser's animation list, not watched as video, so timing and feel need a
look on a real screen. Frame rate on a real GPU was not measured. The first
browser run after the restyle lost its DevTools connection once part-way
through; it did not recur in five later runs and the cause was not found.

## 1. Acceptance checklist

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Requirements match the implementation | Pass | `FINAL_REQUIREMENTS.md`, every row traced to code or a file |
| 2 | All six classifiers genuinely implemented | Pass | Each loaded from disk and used for prediction (§3) |
| 3 | All models have actual evaluation metrics | Pass | `reports/metrics/evaluation_results.json`, six entries (§4) |
| 4 | Dataset split preserved | Pass | 25,000 / 25,000, identical to the official archive (§2) |
| 5 | No test-set tuning or fabricated results | Pass, with one disclosure | §5 |
| 6 | The UI works | Pass | Five pages, real browser, 0 exceptions (§6) |
| 7 | Live prediction works | Pass | Typed a review and ran it in the browser (§6) |
| 8 | Batch prediction and CSV download work | Pass, one part untested | Upload and scoring done in the browser; the download click itself was not exercised (§6, §9) |
| 9 | Tests pass | Pass | 90 passed, 0 skipped, 0 failed (§7) |
| 10 | PDF report opens and is readable | Pass | Rendered and read page by page; 2 pages, tables fit (§8) |
| 11 | PowerPoint opens and slides are readable | Pass | Opened in Microsoft PowerPoint, exported, all 8 slides inspected (§8) |
| 12 | README matches the code | Pass | Rewritten against the final code; commands are the ones used here |
| 13 | FINAL_AUDIT.md accurate | Pass | This document |
| 14 | Reference PDF untouched | Pass | 271,408 bytes, modified 2026-10-11 11:13, unchanged since the first session; SHA-256 `0a1ee50d…4f3260` |
| 15 | Dataset and virtual environment excluded from git | Pass | `git check-ignore` confirms `venv/`, `data/`, `models/*.joblib`, `nltk_data/`, `__pycache__/` |
| 16 | No unrelated files staged or committed | Pass | Index empty; history is still the single pre-existing commit |
| 17 | No GitHub push | Pass | No `git add`, `commit`, `push`, `init` or config command was run |

## 2. Dataset

Verified by `scripts/dataset_stats.py` and `scripts/verify_provenance.py`.

| Check | Result |
|---|---|
| Rows | 25,000 train, 25,000 test |
| Labels | 12,500 positive and 12,500 negative in each split |
| Missing reviews | 0 |
| Median / mean review length (train) | 174 / 234 words |
| Duplicate rows within a split | 96 (train), 199 (test) |
| Test reviews also present in train | 123 (0.49%), 0 label conflicts |
| Cleaned files aligned with raw files | Yes, row for row, both splits |

**Provenance.** The project downloads from the HuggingFace mirror. In this
audit the official archive `aclImdb_v1.tar.gz` (84,125,825 bytes) was
downloaded from `ai.stanford.edu` and every (review text, label) pair in
both splits was compared with the project's data: **all 50,000 are
identical**, with nothing extra on either side. Result:
`reports/metrics/provenance_check.json`.

**Leakage checks.**
- The TF-IDF vectorizer is fitted on the training split only.
- Text cleaning is a fixed, stateless function; it learns nothing from data.
- Duplicates and overlap are left in place so the official split is used as
  published. The 123 overlapping reviews may make test accuracy optimistic
  by a small fraction of a percentage point.

**Fixed in this session.** `build_dataset.py` could not read the extracted
official archive, so the documented fallback download path was unusable.
It now can, and the loader is tested.

## 3. Models

Each artifact was loaded with `joblib` and used to predict two reviews.

| Model | Type on disk | Size | Predicts both classes | Probabilities |
|---|---|---|---|---|
| Logistic Regression | `LogisticRegression` | 0.40 MB | Yes | Yes |
| Linear SVM | `CalibratedClassifierCV` over `LinearSVC` | 1.20 MB | Yes | Yes, Platt-calibrated |
| Multinomial Naive Bayes | `MultinomialNB` | 1.60 MB | Yes | Yes |
| Random Forest | `RandomForestClassifier` | 17.77 MB | Yes | Yes |
| Gradient Boosting | `GradientBoostingClassifier` | 0.17 MB | Yes | Yes |
| Neural network | `MLPClassifier` | 34.64 MB | Yes | Yes |

Logistic Regression, Linear SVM, Naive Bayes and the vectorizer are
byte-identical to their state at the start of the previous session. Random
Forest and the neural network were trained in the previous session.
Gradient Boosting was retrained in this one (§5).

## 4. Evaluation results

`python -m cinesentinel.evaluate`, 25,000 test reviews. Positive class:
"positive". Precision, recall and F1 are for the positive class; Macro-F1
is the unweighted mean of the two per-class F1 scores.

| Model | Accuracy | Precision | Recall | F1 | Macro-F1 | ROC-AUC | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|---|
| Neural network (MLP) | 0.8948 | 0.8944 | 0.8952 | 0.8948 | 0.8948 | 0.9597 | 11,190 | 1,321 | 1,310 | 11,179 |
| Linear SVM | 0.8938 | 0.8988 | 0.8874 | 0.8931 | 0.8938 | 0.9594 | 11,093 | 1,249 | 1,407 | 11,251 |
| Logistic Regression | 0.8922 | 0.8904 | 0.8946 | 0.8925 | 0.8922 | 0.9581 | 11,182 | 1,376 | 1,318 | 11,124 |
| Multinomial Naive Bayes | 0.8633 | 0.8787 | 0.8430 | 0.8605 | 0.8633 | 0.9386 | 10,537 | 1,454 | 1,963 | 11,046 |
| Random Forest | 0.8560 | 0.8684 | 0.8391 | 0.8535 | 0.8560 | 0.9329 | 10,489 | 1,589 | 2,011 | 10,911 |
| Gradient Boosting | 0.8441 | 0.8265 | 0.8710 | 0.8482 | 0.8440 | 0.9260 | 10,888 | 2,286 | 1,612 | 10,214 |

| Model | Training time | Inference, ms per review |
|---|---|---|
| Neural network (MLP) | 238.7 s | 0.0108 |
| Linear SVM | 3.4 s | 0.0014 |
| Logistic Regression | 8.4 s | 0.0004 |
| Multinomial Naive Bayes | 0.1 s | 0.0010 |
| Random Forest | 72.4 s | 0.0444 |
| Gradient Boosting | 476.7 s | 0.0060 |

Inference time is one batch prediction over the test matrix and excludes
text cleaning and vectorization. All timings are from a single run.

Integrity checks (automated in `tests/test_metrics.py`):
- Every confusion matrix sums to 25,000, with 12,500 actual positives and
  12,500 actual negatives.
- Accuracy, precision, recall, specificity and negative precision
  recomputed from the raw counts equal the stored values.
- `comparison.csv` agrees with `evaluation_results.json`.
- The confusion-matrix orientation, F1 and macro-F1 formulas are tested
  against a hand-computed synthetic case.

The best-model label in the app is computed from these results at run
time. Today it is the neural network, by 0.10 points over the SVM. That
margin is about 25 reviews and each model was trained once, so it should
not be presented as a meaningful win.

## 5. Gradient Boosting investigation, and test-set discipline

**Finding: the 81% figure was genuine, not a bug.**

| Check | Result |
|---|---|
| Feature / label alignment | Cleaned and raw files match row for row |
| Train / test usage | Model fitted on train only; evaluated on test only |
| Preprocessing consistency | Same `clean_text` and the same saved vectorizer for train, test and the app |
| Evaluation code | Same `evaluate_model` as the other five models; orientation unit-tested |
| Shuffled-label control | 51.0% validation accuracy, i.e. chance — no leakage |
| Cause | Underfitting: a 300-tree model uses only 839 of 50,000 features |

**What was changed.** The tree count was selected on an 80/20 split of the
training data (`experiments/gb_diagnostics.py`), by a rule written before
the run: the smallest count within 0.2 points of the best validation
accuracy. Validation accuracy: 80.34% at 100 trees, 83.38% at 300; the
rule selected 270 (83.24%). The model was retrained on the full training
split and evaluated once: **84.41%** on test, up from 81.07%.

Gradient Boosting is still the weakest model. Validation accuracy was
still at its maximum at 300 trees, the largest size tried, so a larger
model would probably do somewhat better; this was not pursued.

**Disclosure.** The test split was used to evaluate Gradient Boosting
twice: once with 100 trees and once with 270. The value 270 was chosen
without the test split, but the decision to revisit the setting followed
from seeing the first test result. Both results are reported everywhere
the model is discussed. The 100-tree model and its metrics are kept in
`models/gradient_boosting_paper100.joblib` and
`reports/metrics/gb_paper_config_result.json`.

No other model was changed after its test result was seen.

**Feature-representation experiment** (`experiments/feature_comparison.py`):
27 fits (3 representations × 3 n-gram ranges × 3 classifiers), validation
data only. Range 85.2%–89.4%; TF-IDF best in all nine classifier/n-gram
combinations. Not done: the three slower classifiers, and per-representation
tuning. The main pipeline was not changed.

## 6. Dashboard verification

**The earlier "exit code 127".** That was not a browser failure. The
previous session started Streamlit as a background task, ran the browser
checks successfully, and then force-stopped the server process; the
background task reported the forced stop as exit code 127. The check is
now a script, `scripts/capture_screenshots.py`, which starts and stops the
server itself and exits 0 only if no page shows an exception.

**Result of `python scripts\capture_screenshots.py`** (headless Microsoft
Edge driven over the DevTools protocol against a live `streamlit run`):

| Step | Result |
|---|---|
| HTTP GET on all five routes | 200 each |
| Home | 2 charts, 23 cards, 0 exceptions |
| Model Comparison Lab, all six tabs opened | 16 charts, 0 exceptions |
| Live Analyzer: typed a review, clicked Analyze | Verdict banner, 3 charts, agreement banner |
| Batch Studio: uploaded `sample_batch.csv`, clicked Run | 5 rows scored, 2 charts, download button present |
| Methodology | Rendered, 0 exceptions |
| Exit code | 0 |

Screenshots are in `submission/screenshots/`. The full-page captures of
Home, Live Analyzer, Comparison Lab (ROC and Experiments tabs), Batch
Studio and Methodology were opened and inspected; layout problems found
(overlapping labels, a clipped axis label, misleading delta arrows) were
fixed.

**Headless page tests** (`tests/test_app.py`, 15 tests): every page renders
without exception; the selected model drives the verdict; exactly one of
the agree / disagree banners appears; empty and stop-word-only input give
a warning.

**Confidence wording.** The SVM's value is a Platt-calibrated probability,
never the raw decision score, and the app says so. Every model's
confidence is labelled "estimated probability" with a per-model
explanation.

## 7. Tests

```
$ pytest
90 passed in 22.12s
```

Before this session's changes: 54 passed.

| File | Tests | Covers |
|---|---|---|
| `test_preprocessing.py` | 7 | Text cleaning |
| `test_features.py` | 4 | TF-IDF vectorizer |
| `test_predict.py` | 20 | Positive / negative predictions for all six models, consensus, explanation, batch scoring, duplicate column names, CSV round trip |
| `test_pipeline.py` | 13 | Six-model configuration, paper hyperparameters, every trainer fits, saves and reloads, metric maths |
| `test_metrics.py` | 7 | Integrity of saved evaluation results |
| `test_robustness.py` | 24 | Missing artifacts, CSV validation, dataset tooling, module imports, file compilation |
| `test_app.py` | 15 | Every dashboard page, headless |

Unit tests use small synthetic data. Nothing they measure is reported as
an evaluation result.

`pip check`: no broken requirements.

## 8. Deliverables

| File | Check |
|---|---|
| `submission/CineSentinel_AI_Report.pdf` | 2 pages. Rendered and read: headings, both tables and references fit; nothing clipped |
| `submission/CineSentinel_AI_Report.md` | Editable source of the report |
| `submission/CineSentinel_AI_Presentation.pptx` | 8 slides. Opened in Microsoft PowerPoint (read-only) and exported to PDF; every slide inspected. Overlapping chart labels and unindented bullets found on the first build were fixed and re-checked |
| `submission/screenshots/` | 20 PNG files from the browser run |
| `reports/PRESENTATION_OUTLINE.md` | Matches the deck |
| `reports/PROJECT_REPORT.md` | Same text as the submitted report |

The results chart in the deck is a native, editable PowerPoint chart. All
numbers in both documents are read from `reports/metrics/` when
`scripts/build_submission.py` runs.

### Files created or changed in this session

- New: `FINAL_REQUIREMENTS.md`, `requirements-dev.txt`, `submission/`,
  `experiments/` (3 files), `app/pages/4_Methodology.py`,
  `scripts/dataset_stats.py`, `scripts/verify_provenance.py`,
  `scripts/capture_screenshots.py`, `scripts/build_submission.py`,
  `tests/test_robustness.py`, and six files in `reports/metrics/`
  (`dataset_stats.json`, `provenance_check.json`,
  `evaluation_metadata.json`, `gb_diagnostics.json`,
  `gb_paper_config_result.json`, `feature_comparison.csv`).
- Renamed: the three existing pages, to `1_Live_Analyzer.py`,
  `2_Model_Comparison_Lab.py`, `3_Batch_Studio.py`.
- Changed: `app/Home.py` and the three pages, `config.py`, `train.py`,
  `evaluate.py`, `predict.py`, `app_utils.py`, `ui.py`,
  `scripts/build_dataset.py`, three test files, `.gitignore`, `README.md`,
  this file, both files in `reports/`.
- Retrained: `models/gradient_boosting.joblib`.
- Installed into the venv, for building the documents only: `python-pptx`,
  `Markdown`.

## 9. Known limitations

1. Not an exact reproduction of the paper (1–2-grams rather than 3-grams;
   the paper's vocabulary, preprocessing code and seeds are unpublished).
2. One training run per model; no confidence intervals. The top three
   models are statistically indistinguishable on this evidence.
3. Gradient Boosting was evaluated on the test split twice (§5).
4. Bag-of-words features ignore word order.
5. Only the SVM's probabilities are calibrated.
6. The feature-representation experiment covers three of the six
   classifiers, on validation data.
7. The neural network's result depends on its random seed; one seed was run.

## 10. Manual checks still required

1. **Fill in team names and SRNs** in the report and on slide 1. They are
   placeholders.
2. **Click "Download predictions as CSV"** once in a real browser and open
   the file. The button was confirmed present and the CSV content is
   covered by `test_score_dataframe_csv_round_trip`, but the browser
   download itself was not exercised.
3. **Look at the app at your presentation resolution.** It was inspected at
   1440 px width only; narrow windows and projectors were not checked.
4. **Open the PDF and PPTX on the machine you will present from**, to
   confirm fonts (Segoe UI) render the same there.
5. **Confirm section A of `FINAL_REQUIREMENTS.md` against the actual
   assignment handout**, which is not in this repository.
6. **Decide on the git repository** (§11) before uploading anything.

## 11. Git status

- The enclosing repository is rooted at the user's home directory. Its remote `origin` points to an unrelated repository.
- The project folder has no `.git` of its own. To the home repository it
  is a single untracked directory.
- Nothing was staged, committed, pushed or configured.
- If the project is given its own repository, 77 files totalling about
  9 MB would be tracked. The largest is under 1 MB. The dataset (about
  210 MB), models (56 MB), `nltk_data/` and `venv/` are ignored.

Commands for creating a separate repository are in the hand-over notes;
they have not been run.

## 12. Submission checklist

- [ ] Team names and SRNs filled in (report and slide 1)
- [x] `submission/CineSentinel_AI_Report.pdf` — 2 pages
- [x] `submission/CineSentinel_AI_Presentation.pptx` — 8 slides
- [x] Six models trained and evaluated; metrics files present
- [x] `pytest` — 90 passed
- [x] App starts and all five pages load
- [x] README, FINAL_REQUIREMENTS and FINAL_AUDIT match the code
- [x] Reference PDF unmodified
- [ ] Download button clicked once by hand
- [ ] Deck rehearsed with the live demo
- [ ] Code uploaded to its own repository, if required

## 13. Commands

```powershell
venv\Scripts\Activate.ps1

streamlit run app\Home.py              # run the dashboard
pytest                                 # run the tests
python scripts\capture_screenshots.py  # browser-level check

# Only to regenerate models, metrics or documents:
$env:PYTHONPATH = "src"
python -m cinesentinel.train
python -m cinesentinel.evaluate
python scripts\build_submission.py
```
