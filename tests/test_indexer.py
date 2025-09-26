from __future__ import annotations

import pytest

from evisearch.analyzer import Analyzer
from evisearch.indexer import IndexWriter


def test_add_and_positions_single_doc() -> None:
    az = Analyzer()
    w = IndexWriter(az, index_stopwords=True)
    w.add("d1", "Alpha beta alpha, gamma.")
    idx = w.commit()

    # postings are lowercased and positional
    assert idx.get_posting("alpha")["d1"] == [0, 2]
    assert idx.get_posting("beta")["d1"] == [1]
    assert idx.get_posting("gamma")["d1"] == [3]

    # doc length is computed with stopwords removed (none here)
    assert idx.doc_lengths["d1"] == 4
    assert "d1" in idx.doc_ids


def test_doc_length_filters_stopwords() -> None:
    az = Analyzer()
    w = IndexWriter(az, index_stopwords=True)  # indexing choice doesn't affect doc length calc
    w.add("d2", "The alpha and the beta")
    idx = w.commit()
    # "the" and "and" are stopwords; only alpha/beta count
    assert idx.doc_lengths["d2"] == 2


def test_multiple_docs_and_vocab() -> None:
    az = Analyzer()
    w = IndexWriter(az, index_stopwords=True)
    w.add("a", "alpha beta")
    w.add("b", "beta gamma")
    idx = w.commit()

    # vocabulary contains alpha, beta, gamma
    assert idx.vocabulary_size() >= 3
    # term "beta" appears in both docs
    assert set(idx.get_posting("beta").keys()) == {"a", "b"}


def test_index_stopwords_false_positions_stream() -> None:
    az = Analyzer()
    w = IndexWriter(az, index_stopwords=False)
    # stopwords removed before positions are assigned
    w.add("x", "alpha the beta and beta")
    idx = w.commit()
    # tokens stream after removing stopwords: ["alpha","beta","beta"]
    assert idx.get_posting("alpha")["x"] == [0]
    assert idx.get_posting("beta")["x"] == [1, 2]
    # and no stopword terms should exist in the vocabulary
    assert "the" not in idx.postings and "and" not in idx.postings


def test_postings_are_sorted_and_deduped() -> None:
    az = Analyzer()
    w = IndexWriter(az, index_stopwords=True)
    w.add("dup", "alpha alpha alpha")
    idx = w.commit()
    assert idx.get_posting("alpha")["dup"] == [0, 1, 2]


def test_invalid_doc_id_raises() -> None:
    az = Analyzer()
    w = IndexWriter(az)
    with pytest.raises(ValueError):
        w.add("", "alpha")
