from __future__ import annotations

from evisearch.analyzer import Analyzer
from evisearch.indexer import IndexWriter
from evisearch.storage import load_index, save_index


def test_storage_roundtrip(tmp_path) -> None:
    az = Analyzer()
    w = IndexWriter(az, index_stopwords=True)
    w.add("d1", "hello world\nsecond line")
    w.add("d2", "hello again")
    idx = w.commit()

    out = tmp_path / "idx.json"
    save_index(idx, out)
    idx2 = load_index(out)

    # 基本一致性
    assert set(idx2.doc_ids) == set(idx.doc_ids)
    assert idx2.vocabulary_size() == idx.vocabulary_size()

    # 取一个常见词核对 postings
    p1 = idx.get_posting("hello")
    p2 = idx2.get_posting("hello")
    assert set(p1.keys()) == set(p2.keys())
    for d in p1:
        assert p2[d] == p1[d]

    # 行号映射应存在且非空
    assert "d1" in idx2.pos_to_line and len(idx2.pos_to_line["d1"]) > 0
