from cinesentinel.features import build_vectorizer

CORPUS = [
    "great movie loved acting",
    "terrible film hated plot",
    "great acting wonderful story",
    "awful boring waste time",
]


def test_vectorizer_builds_with_configured_params():
    vectorizer = build_vectorizer()
    assert vectorizer.ngram_range == (1, 2)
    assert vectorizer.min_df == 5
    assert vectorizer.max_df == 0.9


def test_vectorizer_fit_transform_shape():
    # min_df/max_df relaxed here: those thresholds are tuned for the real
    # 25,000-document training corpus, not a 4-sentence fixture.
    vectorizer = build_vectorizer(min_df=1, max_df=1.0)
    X = vectorizer.fit_transform(CORPUS)
    assert X.shape[0] == len(CORPUS)
    assert X.shape[1] == len(vectorizer.vocabulary_)
    assert X.shape[1] > 0


def test_vectorizer_includes_bigrams():
    vectorizer = build_vectorizer(min_df=1, max_df=1.0)
    vectorizer.fit(CORPUS)
    bigram_terms = [term for term in vectorizer.vocabulary_ if " " in term]
    assert len(bigram_terms) > 0


def test_vectorizer_transform_unseen_text_does_not_error():
    vectorizer = build_vectorizer(min_df=1, max_df=1.0)
    vectorizer.fit(CORPUS)
    X = vectorizer.transform(["completely new words never seen before"])
    assert X.shape[0] == 1
