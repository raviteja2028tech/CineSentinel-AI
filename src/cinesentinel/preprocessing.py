"""
Text cleaning and normalization pipeline.

Mirrors the reference paper's §3.2(1) data-cleaning step:
  - strip HTML line breaks / punctuation / digits
  - lowercase
  - remove stopwords
  - normalize words to their root form (lemmatization)
"""

import re
from functools import lru_cache

import nltk

from cinesentinel.config import NLTK_DATA_DIR

nltk.data.path.insert(0, str(NLTK_DATA_DIR))

from nltk.corpus import stopwords  # noqa: E402
from nltk.stem import WordNetLemmatizer  # noqa: E402
from nltk.tokenize import word_tokenize  # noqa: E402

HTML_BREAK_PATTERN = re.compile(r"<br\s*/?>", re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
NON_ALPHA_PATTERN = re.compile(r"[^a-z\s]")
MULTI_SPACE_PATTERN = re.compile(r"\s+")


def ensure_nltk_resources() -> None:
    """Download required NLTK corpora into the project-local nltk_data dir."""
    resources = {
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
        "tokenizers/punkt_tab": "punkt_tab",
    }
    NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for lookup_path, package in resources.items():
        try:
            nltk.data.find(lookup_path)
        except LookupError:
            nltk.download(package, download_dir=str(NLTK_DATA_DIR), quiet=True)


@lru_cache(maxsize=1)
def _ensure_resources_once() -> None:
    ensure_nltk_resources()


@lru_cache(maxsize=1)
def _get_stopwords() -> frozenset:
    return frozenset(stopwords.words("english"))


@lru_cache(maxsize=1)
def _get_lemmatizer() -> WordNetLemmatizer:
    return WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Clean and normalize a single review's raw text.

    Steps: strip HTML breaks/tags, lowercase, remove non-alpha characters
    (punctuation + digits), tokenize, drop stopwords, lemmatize to root form.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = HTML_BREAK_PATTERN.sub(" ", text)
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = text.lower()
    text = NON_ALPHA_PATTERN.sub(" ", text)
    text = MULTI_SPACE_PATTERN.sub(" ", text).strip()

    if not text:
        return ""

    _ensure_resources_once()
    tokens = word_tokenize(text)
    stop_words = _get_stopwords()
    lemmatizer = _get_lemmatizer()

    cleaned_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words and len(token) > 1
    ]
    return " ".join(cleaned_tokens)


def clean_texts(texts) -> list:
    """Vectorized convenience wrapper over an iterable of raw review strings."""
    return [clean_text(t) for t in texts]
