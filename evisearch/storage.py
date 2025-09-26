from __future__ import annotations

import json
from pathlib import Path

from .indexer import InvertedIndex


def save_index(index: InvertedIndex, path: str | Path) -> None:
    """Serialize the index to JSON on disk (fields aligned with our InvertedIndex)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "postings": index.postings,          # term -> {doc_id: [pos, ...]}
        "doc_lengths": index.doc_lengths,
        "doc_ids": index.doc_ids,
        "doc_lines": index.doc_lines,        # doc_id -> [line, ...]
        "line_of_pos": index.line_of_pos,    # doc_id -> [lineIndex, ...]
        "page_map": getattr(index, "page_map", {}),
    }
    p.write_text(json.dumps(data, ensure_ascii=False))


def load_index(path: str | Path) -> InvertedIndex:
    """Load an index saved by `save_index`."""
    p = Path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))

    # Provide defaults for older snapshots.
    postings = obj["postings"]
    doc_lengths = obj["doc_lengths"]
    doc_ids = obj["doc_ids"]
    doc_lines = obj.get("doc_lines", {})
    line_of_pos = obj.get("line_of_pos", {})
    page_map = obj.get("page_map", {})

    return InvertedIndex(
        postings=postings,
        doc_lengths=doc_lengths,
        doc_ids=doc_ids,
        doc_lines=doc_lines,
        line_of_pos=line_of_pos,
        page_map=page_map,
    )
