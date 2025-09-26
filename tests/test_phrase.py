from __future__ import annotations

from evisearch.analyzer import Analyzer
from evisearch.indexer import IndexWriter
from evisearch.searcher import PhraseMatcher


def build_idx(index_stopwords: bool = True):
    az = Analyzer()
    w = IndexWriter(az, index_stopwords=index_stopwords)
    w.add("X", "alpha beta alpha beta")
    w.add("Y", "alpha the beta")  # has a stopword between, not an exact phrase
    w.add("Z", "Hello, WORLD! hello world")  # case/punct variations normalize
    return w.commit(), az


def test_basic_phrase_matches():
    idx, az = build_idx(index_stopwords=True)
    pm = PhraseMatcher(idx, az)

    hits = pm.match("alpha beta", keep_stopwords=True)
    assert sorted(hits.keys()) == ["X"]
    assert hits["X"] == [0, 2]  # two occurrences


def test_phrase_does_not_cross_stopwords():
    idx, az = build_idx(index_stopwords=True)
    pm = PhraseMatcher(idx, az)

    hits = pm.match("alpha beta", keep_stopwords=True)
    # Y has "alpha the beta" -> not adjacent -> no hit
    assert "Y" not in hits


def test_phrase_with_case_and_punct():
    idx, az = build_idx(index_stopwords=True)
    pm = PhraseMatcher(idx, az)

    hits = pm.match("hello world", keep_stopwords=True)
    assert hits["Z"] == [0, 2]
