from __future__ import annotations

import json
from pathlib import Path

from evisearch.indexer import InvertedIndex


def save_index(index: InvertedIndex, path: str | Path) -> None:
    """Serialize the index to JSON on disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "postings": index.postings,  # term -> {doc_id: [pos, ...]}
        "doc_lengths": index.doc_lengths,
        "doc_ids": index.doc_ids,
        "doc_lines": index.doc_lines,
        # inner dict keys (positions) must be strings for JSON to be stable
        "pos_to_line": {
            doc: {str(pos): int(line) for pos, line in mapping.items()}
            for doc, mapping in index.pos_to_line.items()
        },
        "line_token_offsets": index.line_token_offsets,
    }
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def load_index(path: str | Path) -> InvertedIndex:
    """Load an index saved by save_index()."""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))

    # Convert back to the dataclass structure; make sure numeric types are ints.
    pos_to_line = {
        str(doc): {int(pos): int(line) for pos, line in mapping.items()}
        for doc, mapping in data.get("pos_to_line", {}).items()
    }

    idx = InvertedIndex(
        postings={
            str(term): {str(doc): [int(x) for x in positions] for doc, positions in by_doc.items()}
            for term, by_doc in data["postings"].items()
        },
        doc_lengths={str(doc): int(n) for doc, n in data["doc_lengths"].items()},
        doc_ids=[str(d) for d in data["doc_ids"]],
        doc_lines={
            str(doc): [str(line) for line in lines] for doc, lines in data["doc_lines"].items()
        },
        pos_to_line=pos_to_line,
        line_token_offsets={
            str(doc): [int(x) for x in offs]
            for doc, offs in data.get("line_token_offsets", {}).items()
        },
    )
    return idx
