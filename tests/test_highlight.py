from __future__ import annotations

from evisearch.analyzer import Analyzer
from evisearch.highlight import Highlighter
from evisearch.indexer import IndexWriter
from evisearch.searcher import PhraseMatcher


def test_highlight_returns_original_line_and_number() -> None:
    az = Analyzer()
    w = IndexWriter(az, index_stopwords=True)
    text = "AP credit policy allows certain exams.\nContact the registrar for details."
    w.add("doc", text)
    idx = w.commit()

    pm = PhraseMatcher(idx, az)
    hits = pm.match("ap credit policy", keep_stopwords=True)
    assert "doc" in hits
    start = hits["doc"][0]

    hi = Highlighter(idx)
    snippet, line_no = hi.build_snippet("doc", start_pos=start)
    assert line_no == 1
    assert snippet == "AP credit policy allows certain exams."
