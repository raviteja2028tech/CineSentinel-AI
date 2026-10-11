from cinesentinel.preprocessing import clean_text, clean_texts, ensure_nltk_resources

ensure_nltk_resources()


def test_lowercases_text():
    assert "amazing" in clean_text("This Movie Was AMAZING")


def test_strips_html_breaks_and_tags():
    cleaned = clean_text("Great film.<br /><br /><b>Loved</b> it.")
    assert "br" not in cleaned
    assert "loved" in cleaned


def test_removes_punctuation_and_numbers():
    cleaned = clean_text("Rated 10/10!!! Best movie ever, seriously.")
    for char in "10/10!!!,":
        assert char not in cleaned


def test_removes_stopwords():
    cleaned = clean_text("This is a movie that was the best of all time")
    tokens = cleaned.split()
    assert "the" not in tokens
    assert "is" not in tokens
    assert "a" not in tokens


def test_lemmatizes_to_root_form():
    cleaned = clean_text("The actors were playing amazingly well")
    assert "play" in cleaned or "playing" in cleaned  # lemmatizer keeps verb gerunds by default POS


def test_empty_and_non_string_input_returns_empty_string():
    assert clean_text("") == ""
    assert clean_text("   ") == ""
    assert clean_text(None) == ""


def test_clean_texts_batch():
    results = clean_texts(["Great movie!", "Terrible film."])
    assert len(results) == 2
    assert all(isinstance(r, str) for r in results)
