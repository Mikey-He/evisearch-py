from __future__ import annotations

from evisearch.analyzer import Analyzer
from evisearch.indexer import IndexWriter
from evisearch.searcher import Searcher


def build_small_index(index_stopwords: bool = False):
    az = Analyzer()
    w = IndexWriter(az, index_stopwords=index_stopwords)
    w.add("A", "alpha beta gamma")
    w.add("B", "beta gamma")
    w.add("C", "gamma delta")
    w.add("D", "epsilon")
    return w.commit(), az


def test_and_or_not_basic() -> None:
    idx, az = build_small_index()
    s = Searcher(idx, az)

    assert s.search_boolean("alpha AND beta") == ["A"]
    assert s.search_boolean("beta OR delta") == ["A", "B", "C"]
    assert s.search_boolean("beta AND NOT alpha") == ["B"]
    assert s.search_boolean("NOT delta") == ["A", "B", "D"]
    assert s.search_boolean("unknown AND alpha") == []


def test_precedence_and_parentheses() -> None:
    idx, az = build_small_index()
    s = Searcher(idx, az)

    # precedence: NOT > AND > OR
    assert s.search_boolean("alpha OR beta AND gamma") == ["A", "B"]
    assert s.search_boolean("(alpha OR beta) AND gamma") == ["A", "B"]
    assert s.search_boolean("alpha AND (beta OR delta)") == ["A"]


def test_index_stopwords_true_still_works() -> None:
    idx, az = build_small_index(index_stopwords=True)
    s = Searcher(idx, az)
    assert s.search_boolean("gamma AND delta") == ["C"]
