from __future__ import annotations

import threading
from collections.abc import Iterable
from typing import Annotated

from fastapi import Body, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from evisearch.analyzer import Analyzer
from evisearch.highlight import Highlighter
from evisearch.indexer import IndexWriter, InvertedIndex
from evisearch.searcher import PhraseMatcher, Searcher

# ---------------------------
# App & global state
# ---------------------------
app = FastAPI(title="EviSearch-Py", version="1.0.0")

_ANALYZER = Analyzer()
_INDEX_LOCK = threading.Lock()
_INDEX: InvertedIndex | None = None


# ---------------------------
# Pydantic models (I/O)
# ---------------------------
class DocIn(BaseModel):
    id: str = Field(..., min_length=1)
    text: str


class IndexIn(BaseModel):
    docs: list[DocIn]
    index_stopwords: bool = True


class IndexOut(BaseModel):
    ok: bool
    indexed: int
    docs: int
    vocab: int


class SearchItem(BaseModel):
    doc_id: str
    score: float | None = None
    snippet: str = ""
    line: int = 0


class SearchOut(BaseModel):
    results: list[SearchItem]


# ---------------------------
# Helpers
# ---------------------------
def _require_index() -> InvertedIndex:
    if _INDEX is None:
        raise HTTPException(status_code=400, detail="No index in memory. POST /index first.")
    return _INDEX


def _first_hit_pos(index: InvertedIndex, terms: Iterable[str], doc_id: str) -> int | None:
    """First global token position in doc where ANY term occurs."""
    best: int | None = None
    for t in terms:
        posting = index.get_posting(t)
        if not posting:
            continue
        positions = posting.get(doc_id)
        if not positions:
            continue
        pos0 = positions[0]
        if best is None or pos0 < best:
            best = pos0
    return best


# ---------------------------
# Routes
# ---------------------------
@app.get("/")
def root() -> dict[str, str | int]:
    idx = _INDEX
    return {
        "app": "EviSearch-Py",
        "docs": 0 if idx is None else len(idx.doc_ids),
        "vocab": 0 if idx is None else idx.vocabulary_size(),
    }


@app.post("/index", response_model=IndexOut)
def post_index(payload: Annotated[IndexIn, Body(...)]) -> IndexOut:
    global _INDEX
    if not payload.docs:
        raise HTTPException(status_code=400, detail="No docs provided.")

    writer = IndexWriter(_ANALYZER, index_stopwords=payload.index_stopwords)
    for d in payload.docs:
        writer.add(d.id, d.text)
    idx = writer.commit()

    with _INDEX_LOCK:
        _INDEX = idx

    return IndexOut(
        ok=True,
        indexed=len(payload.docs),
        docs=len(idx.doc_ids),
        vocab=idx.vocabulary_size(),
    )


@app.get("/search", response_model=SearchOut)
def get_search(
    q: Annotated[str, Query(min_length=1, description="Query string")],
    k: Annotated[int, Query(ge=1, le=100, description="Top-K for ranked mode")] = 10,
    mode: Annotated[str, Query(pattern="^(ranked|boolean|phrase)$")] = "ranked",
) -> SearchOut:
    idx = _require_index()
    s = Searcher(idx, _ANALYZER)
    hi = Highlighter(idx)

    if mode == "boolean":
        doc_ids = s.search_boolean(q)
        terms = _ANALYZER.tokenize(q, keep_stopwords=False)
        bool_items: list[SearchItem] = []
        for d in doc_ids:
            pos = _first_hit_pos(idx, terms, d)
            snippet, line = hi.build_snippet(d, pos) if pos is not None else ("", 0)
            bool_items.append(SearchItem(doc_id=d, snippet=snippet, line=line))
        return SearchOut(results=bool_items)

    if mode == "phrase":
        pm = PhraseMatcher(idx, _ANALYZER)
        hits = pm.match(q, keep_stopwords=True)
        phrase_items: list[SearchItem] = []
        for d in sorted(hits.keys()):
            start = hits[d][0]  # first occurrence
            snippet, line = hi.build_snippet(d, start)
            phrase_items.append(SearchItem(doc_id=d, snippet=snippet, line=line))
        return SearchOut(results=phrase_items)

    # ranked (BM25)
    results = s.search_ranked(q, k=k)
    terms = _ANALYZER.tokenize(q, keep_stopwords=False)
    ranked_items: list[SearchItem] = []
    for d, score in results:
        pos = _first_hit_pos(idx, terms, d)
        snippet, line = hi.build_snippet(d, pos) if pos is not None else ("", 0)
        ranked_items.append(SearchItem(doc_id=d, score=score, snippet=snippet, line=line))
    return SearchOut(results=ranked_items)
