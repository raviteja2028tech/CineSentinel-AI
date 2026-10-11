"""Central path and hyperparameter configuration for CineSentinel AI."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TRAIN_CSV = PROCESSED_DIR / "train.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"

MODELS_DIR = PROJECT_ROOT / "models"

# One TF-IDF vectorizer (50,000-term vocabulary) is shared by all six models,
# so every classifier is trained and evaluated on exactly the same feature
# matrix and the comparison between them is like-for-like.
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"

MODEL_PATHS = {
    "logistic_regression": MODELS_DIR / "logistic_regression.joblib",
    "linear_svm": MODELS_DIR / "linear_svm.joblib",
    "multinomial_nb": MODELS_DIR / "multinomial_nb.joblib",
    "random_forest": MODELS_DIR / "random_forest.joblib",
    "gradient_boosting": MODELS_DIR / "gradient_boosting.joblib",
    "mlp_neural_net": MODELS_DIR / "mlp_neural_net.joblib",
}

REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
NLTK_DATA_DIR = PROJECT_ROOT / "nltk_data"

# TF-IDF settings (paper: TF-IDF vectorization; we extend to (1,2)-grams per
# the paper's finding that n-grams improve over unigrams-only TF-IDF).
TFIDF_PARAMS = {
    "ngram_range": (1, 2),
    "max_features": 50000,
    "min_df": 5,
    "max_df": 0.9,
    "sublinear_tf": True,
}

# Logistic Regression / Linear SVM inverse-regularization grid, per paper 3.3.
C_GRID = [0.01, 0.05, 0.25, 0.5, 1.0]
CV_FOLDS = 3
RANDOM_STATE = 42

# Multinomial Naive Bayes: Laplace smoothing = 1, per paper 3.3(3).
NB_ALPHA = 1.0

# Random Forest — paper 3.3(4): 100 trees, Gini criterion, no max depth,
# min_samples_split=2. n_jobs is capped (not -1) to bound peak memory on a
# shared student laptop rather than parallelizing across every core at once.
RF_PARAMS = {
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": None,
    "min_samples_split": 2,
    "n_jobs": 2,
    "random_state": RANDOM_STATE,
}

# Boosting — paper 3.3(5): 100 boosting trees, MSE split criterion,
# learning rate 0.1, min_samples_split=2. Implemented with scikit-learn's
# GradientBoostingClassifier, which accepts the sparse TF-IDF matrix directly.
#
# The tree count is the one setting that departs from the paper. With 100
# trees the model underfits 50,000 sparse features, so n_estimators was chosen
# on a validation split of the training data by experiments/gb_diagnostics.py
# (results in reports/metrics/gb_diagnostics.json). The test split played no
# part in that choice. The 100-tree result is kept in
# reports/metrics/gb_paper_config_result.json.
GB_PAPER_N_ESTIMATORS = 100
GB_PARAMS = {
    "n_estimators": 270,
    "criterion": "squared_error",
    "learning_rate": 0.1,
    "min_samples_split": 2,
    "random_state": RANDOM_STATE,
}

# Deep Neural Network — paper 3.3(6): hidden layers (30,30,20,10,10),
# logistic activation, L2=0.0001, early stopping, Adam solver.
MLP_PARAMS = {
    "hidden_layer_sizes": (30, 30, 20, 10, 10),
    "activation": "logistic",
    "alpha": 0.0001,
    "early_stopping": True,
    "solver": "adam",
    "max_iter": 200,
    "random_state": RANDOM_STATE,
}

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "linear_svm": "Linear SVM",
    "multinomial_nb": "Multinomial Naive Bayes",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "mlp_neural_net": "Deep Neural Network (MLP)",
}

MODEL_SHORT_NAMES = {
    "logistic_regression": "LogReg",
    "linear_svm": "SVM",
    "multinomial_nb": "Naive Bayes",
    "random_forest": "Rand. Forest",
    "gradient_boosting": "Grad. Boosting",
    "mlp_neural_net": "DNN (MLP)",
}

MODEL_DESCRIPTIONS = {
    "logistic_regression": (
        "A linear classifier that models the log-odds of a review being "
        "positive as a weighted sum of TF-IDF features, fit by maximizing "
        "likelihood. Fast, interpretable, and a strong baseline for "
        "sparse high-dimensional text data."
    ),
    "linear_svm": (
        "Finds the hyperplane in TF-IDF feature space that separates "
        "positive and negative reviews with maximum margin. Trained as a "
        "LinearSVC and wrapped with Platt-scaling calibration so it can "
        "report calibrated probability estimates rather than a raw decision "
        "score."
    ),
    "multinomial_nb": (
        "A generative probabilistic classifier that assumes word "
        "occurrences are conditionally independent given the sentiment "
        "label (the 'naive' assumption), and predicts via Bayes' rule. "
        "Extremely fast to train and a classic text-classification "
        "baseline."
    ),
    "random_forest": (
        "An ensemble of decision trees, each trained on a bootstrap "
        "sample with a randomized feature subset at each split, with "
        "predictions combined by majority vote. Captures non-linear "
        "feature interactions that linear models cannot."
    ),
    "gradient_boosting": (
        "An ensemble of shallow trees trained sequentially, where each new "
        "tree corrects the errors of the ensemble so far. Slower to train "
        "than a random forest, but each tree is small and the saved model is "
        "compact."
    ),
    "mlp_neural_net": (
        "A feed-forward deep neural network (multilayer perceptron) with "
        "five hidden layers, trained via backpropagation with the Adam "
        "optimizer. Can model complex non-linear feature interactions at "
        "the cost of longer training time and reduced interpretability."
    ),
}

# How each model arrives at the confidence value shown in the app. Shown to
# users so a confidence figure is never mistaken for something it is not.
MODEL_CONFIDENCE_NOTES = {
    "logistic_regression": (
        "The model's own class probability: the sigmoid of its linear score."
    ),
    "linear_svm": (
        "An SVM outputs a signed distance from its decision boundary, which is "
        "not a probability. The value shown is a calibrated probability obtained "
        "by Platt scaling: a sigmoid fitted to those distances with 3-fold "
        "cross-validation on the training split."
    ),
    "multinomial_nb": (
        "The posterior probability from Bayes' rule. The independence assumption "
        "makes Naive Bayes over-confident, so values cluster near 0% and 100%."
    ),
    "random_forest": (
        "The average, over all trees, of the positive-class fraction in the "
        "leaf each tree assigns the review to. It rarely reaches the extremes."
    ),
    "gradient_boosting": (
        "The sigmoid of the summed tree outputs. Small trees move the score in "
        "small steps, so values stay closer to 50% than for the linear models."
    ),
    "mlp_neural_net": (
        "The network's sigmoid output. Neural networks are often over-confident "
        "and this one was not calibrated."
    ),
}

MODEL_LIMITATIONS = {
    "logistic_regression": (
        "Linear in the features: it cannot model interactions such as negation "
        "beyond what bigrams capture."
    ),
    "linear_svm": (
        "Linear decision boundary; probabilities come from a separate "
        "calibration step, not from the SVM itself."
    ),
    "multinomial_nb": (
        "Assumes words occur independently given the label, which is false for "
        "natural language; its probabilities are poorly calibrated."
    ),
    "random_forest": (
        "Each split tests one term and most terms are absent from most reviews, "
        "so trees are inefficient on sparse text. The saved model is large."
    ),
    "gradient_boosting": (
        "Shallow trees use only a small fraction of the 50,000 terms, so the "
        "model underfits. It is also the slowest model to train."
    ),
    "mlp_neural_net": (
        "Works on bag-of-words features, so it ignores word order. Slow to train, "
        "hard to interpret, and its result depends on the random seed."
    ),
}

MODEL_FAMILIES = {
    "logistic_regression": "Linear",
    "linear_svm": "Linear",
    "multinomial_nb": "Probabilistic",
    "random_forest": "Tree ensemble",
    "gradient_boosting": "Tree ensemble",
    "mlp_neural_net": "Neural network",
}

APP_NAME = "CineSentinel AI"
APP_TAGLINE = "IMDb Sentiment Intelligence Lab"

# TF-IDF rows of Table 2 in the reference paper (Wu & Shin), transcribed for
# side-by-side comparison in the app and report. These are the PAPER'S
# published numbers, not results produced by this project.
PAPER_TFIDF_RESULTS = {
    "logistic_regression": {"positive_precision": 0.881, "negative_precision": 0.872, "accuracy": 0.877},
    "linear_svm": {"positive_precision": 0.904, "negative_precision": 0.896, "accuracy": 0.900},
    "multinomial_nb": {"positive_precision": 0.819, "negative_precision": 0.868, "accuracy": 0.879},
    "random_forest": {"positive_precision": 0.864, "negative_precision": 0.786, "accuracy": 0.844},
    "gradient_boosting": {"positive_precision": 0.865, "negative_precision": 0.789, "accuracy": 0.825},
    "mlp_neural_net": {"positive_precision": 0.881, "negative_precision": 0.921, "accuracy": 0.901},
}

# Consistent per-model color across every chart in the app. Crimson and
# silver lead, to match the black / crimson / silver theme; the other four
# are chosen to stay clearly distinguishable from them and from each other.
MODEL_COLORS = {
    "logistic_regression": "#C9CED6",  # silver
    "linear_svm": "#5B8DEF",  # steel blue
    "multinomial_nb": "#2DD4BF",  # teal
    "random_forest": "#F59E0B",  # amber
    "gradient_boosting": "#A78BFA",  # violet
    "mlp_neural_net": "#E11D2E",  # crimson
}

SENTIMENT_COLORS = {
    "positive": "#22C55E",
    "negative": "#F43F5E",
    "neutral": "#94A3B8",
}
