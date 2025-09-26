from __future__ import annotations

import pytest

from evisearch.analyzer import Analyzer


@pytest.fixture(scope="module")
def az() -> Analyzer:
    return Analyzer()


def test_empty_and_whitespace(az: Analyzer) -> None:
    assert az.tokenize("") == []
    assert az.tokenize("   \t\n  ") == []


def test_basic_lower_and_punct(az: Analyzer) -> None:

    text = "Evidence-level retrieval matrix; this is a demo."
    assert az.tokenize(text) == ["evidence", "level", "retrieval", "matrix", "demo"]


def test_apostrophes_and_hyphens() -> None:
    # Use min_token_len=2 to drop the single-letter token "t" from "don't".
    az = Analyzer(min_token_len=2)
    text = "Don't stop-believing: resilient systems."
    toks = az.tokenize(text)

    assert "stop" not in toks
    # Meaningful words remain
    for w in ("believing", "resilient", "systems"):
        assert w in toks


def test_unicode_accents_and_symbols(az: Analyzer) -> None:
    text = "Café déjà-vu is about déjà vu!"
    toks = az.tokenize(text)
    assert {"cafe", "deja"}.issubset(set(toks))


def test_keep_stopwords_for_phrase() -> None:
    az = Analyzer()
    text = "The quick brown fox jumps over the lazy dog"
    toks_no_stop = az.tokenize(text)  # default removes stopwords
    toks_keep = az.tokenize(text, keep_stopwords=True)
    assert "the" not in toks_no_stop
    assert "the" in toks_keep  # kept when keep_stopwords=True


def test_positions_without_stopwords() -> None:
    az = Analyzer()
    text = "A simple, simple check."
    pairs = az.tokenize_with_positions(text)  # default removes "a"
    assert [t for t, _ in pairs] == ["simple", "simple", "check"]
    assert [p for _, p in pairs] == [0, 1, 2]


def test_positions_with_stopwords() -> None:
    az = Analyzer()
    text = "The quick brown fox"
    pairs = az.tokenize_with_positions(text, keep_stopwords=True)
    assert pairs == [("the", 0), ("quick", 1), ("brown", 2), ("fox", 3)]


def test_min_token_length_filters_single_letters() -> None:
    az = Analyzer(min_token_len=2)
    text = "I am a CS student in AI"
    toks = az.tokenize(text, keep_stopwords=True)
    assert "i" not in toks and "a" not in toks
    for expected in ("am", "in", "cs", "ai", "student"):
        assert expected in toks


def test_numbers_are_stopwords_in_your_list(az: Analyzer) -> None:

    text = "Top 10 of 39 items"
    toks = az.tokenize(text)
    for w in ("10", "39", "of", "top"):
        assert w not in toks
    assert "items" in toks


def test_domain_like_tokens_removed(az: Analyzer) -> None:

    text = "Visit co. example com domain"
    toks = az.tokenize(text)
    assert "co" not in toks and "com" not in toks
    assert "domain" in toks
