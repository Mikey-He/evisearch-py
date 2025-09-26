from __future__ import annotations

from evisearch.analyzer import Analyzer
from evisearch.indexer import IndexWriter
from evisearch.searcher import Searcher


def build_idx():
    az = Analyzer()
    w = IndexWriter(az, index_stopwords=False)
    # D1 has both alpha and beta; beta appears twice
    w.add("D1", "alpha beta beta gamma")
    # D2 has alpha and beta once each
    w.add("D2", "alpha beta")
    # D3 only has beta and gamma (should rank lower for query 'alpha beta')
    w.add("D3", "beta gamma gamma")
    return w.commit(), az


def test_ranked_order_and_topk() -> None:
    idx, az = build_idx()
    s = Searcher(idx, az)

    results = s.search_ranked("alpha beta", k=2)
    docs = [doc for doc, _ in results]
    # Either order is acceptable depending on length normalization; both must be in Top-2
    assert docs in (["D1", "D2"], ["D2", "D1"])
    # Non-increasing scores
    assert results[0][1] >= results[1][1]


def test_ranked_handles_unknown_terms() -> None:
    idx, az = build_idx()
    s = Searcher(idx, az)

    assert s.search_ranked("unknownterm", k=5) == []
    out = s.search_ranked("unknownterm beta", k=5)
    assert len(out) > 0
