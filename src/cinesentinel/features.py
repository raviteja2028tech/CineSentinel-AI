"""TF-IDF feature extraction wrapper."""

from sklearn.feature_extraction.text import TfidfVectorizer

from cinesentinel.config import TFIDF_PARAMS


def build_vectorizer(**overrides) -> TfidfVectorizer:
    """Construct the TF-IDF vectorizer with the project's configured params.

    Accepts keyword overrides (e.g. min_df=1) so callers such as unit tests
    can relax corpus-size-dependent thresholds without duplicating defaults.
    """
    params = {**TFIDF_PARAMS, **overrides}
    return TfidfVectorizer(**params)
