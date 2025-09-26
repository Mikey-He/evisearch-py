# isort: skip_file
# ruff: noqa: E501
from __future__ import annotations

import io
import os
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated

from fastapi import Body, Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field

from .analyzer import Analyzer
from .highlight import Context as Ctx, Highlighter
from .indexer import IndexWriter, InvertedIndex
from .searcher import PhraseMatcher, Searcher
from .storage import load_index, save_index


# App & global state

app = FastAPI(title="EviSearch-Py", version="1.0.0")

_ANALYZER = Analyzer()
_INDEX_LOCK = threading.Lock()
_INDEX: InvertedIndex | None = None

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
SNAPSHOT = DATA_DIR / "index.json"


# Optional Basic Auth (enable by setting BASIC_USER/PASS)
security = HTTPBasic()
_USER = os.getenv("BASIC_USER")
_PASS = os.getenv("BASIC_PASS")


def _auth_enabled() -> bool:
    return bool(_USER and _PASS)


def require_auth(credentials: Annotated[HTTPBasicCredentials, Depends(security)]) -> None:
    if not _auth_enabled():
        return
    if credentials.username != _USER or credentials.password != _PASS:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )


# Models


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


class Hit(BaseModel):
    line: int
    page: int | None = None
    snippet: str


class DocGroup(BaseModel):
    doc_id: str
    score: float | None = None
    hits: list[Hit]


class SearchXOut(BaseModel):
    groups: list[DocGroup]


# Helpers
def _require_index() -> InvertedIndex:
    if _INDEX is None:
        raise HTTPException(
            status_code=400, detail="No index in memory. POST /index or /index-files first."
        )
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


def _extract_text_from_upload(up: UploadFile) -> tuple[str, str]:
    """
    Read an UploadFile and return (doc_id, text).
    - .txt: decode as utf-8
    - .pdf: use pdfminer.six to extract text (in-memory)
    """
    name = (up.filename or "doc").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    base = name.rsplit(".", 1)[0] or "doc"
    raw = up.file.read()

    ctype = (up.content_type or "").lower()
    is_pdf = ctype == "application/pdf" or name.lower().endswith(".pdf")

    if is_pdf:
        try:
            from pdfminer.high_level import extract_text  # lazy import
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"pdfminer.six not installed or failed to import: {e}",
            ) from None
        try:
            text = extract_text(io.BytesIO(raw)) or ""
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {e}") from None
        return (base, text)

    # treat as plain text
    try:
        text = raw.decode("utf-8", errors="ignore")
    except Exception:
        text = ""
    return (base, text)


# Startup: try to load existing snapshot (best-effort)
@app.on_event("startup")
def _load_snapshot() -> None:
    global _INDEX
    try:
        if SNAPSHOT.exists():
            _INDEX = load_index(SNAPSHOT)
    except Exception as e:  # non-critical
        print(f"[warn] failed to load index snapshot: {e}")


# Routes
@app.get("/")
def root() -> dict[str, str | int]:
    idx = _INDEX
    return {
        "app": "EviSearch-Py",
        "docs": 0 if idx is None else len(idx.doc_ids),
        "vocab": 0 if idx is None else idx.vocabulary_size(),
        "auth": "on" if _auth_enabled() else "off",
    }


@app.post("/index", response_model=IndexOut)
def post_index(
    _: Annotated[None, Depends(require_auth)],
    payload: Annotated[IndexIn, Body(...)],
) -> IndexOut:
    global _INDEX
    if not payload.docs:
        raise HTTPException(status_code=400, detail="No docs provided.")

    writer = IndexWriter(_ANALYZER, index_stopwords=payload.index_stopwords)
    for d in payload.docs:
        writer.add(d.id, d.text)
    idx = writer.commit()

    with _INDEX_LOCK:
        _INDEX = idx
    # best-effort snapshot (optional)
    try:
        save_index(idx, SNAPSHOT)
    except Exception as e:
        print(f"[warn] failed to save index snapshot: {e}")

    return IndexOut(
        ok=True, indexed=len(payload.docs), docs=len(idx.doc_ids), vocab=idx.vocabulary_size()
    )


@app.post("/index-files", response_model=IndexOut)
def post_index_files(
    _: Annotated[None, Depends(require_auth)],
    files: Annotated[list[UploadFile], File(...)],
) -> IndexOut:
    """
    Upload .pdf or .txt files and build in-memory index.
    This replaces any previous index.
    """
    global _INDEX
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    docs: list[DocIn] = []
    for f in files:
        doc_id, text = _extract_text_from_upload(f)
        if not text.strip():
            continue
        docs.append(DocIn(id=doc_id, text=text))

    if not docs:
        raise HTTPException(status_code=400, detail="All uploaded files were empty/unreadable.")

    writer = IndexWriter(_ANALYZER, index_stopwords=True)
    for d in docs:
        writer.add(d.id, d.text)
    idx = writer.commit()

    with _INDEX_LOCK:
        _INDEX = idx
    # best-effort snapshot (optional)
    try:
        save_index(idx, SNAPSHOT)
    except Exception as e:
        print(f"[warn] failed to save index snapshot: {e}")

    return IndexOut(ok=True, indexed=len(docs), docs=len(idx.doc_ids), vocab=idx.vocabulary_size())


@app.get("/search", response_model=SearchOut)
def get_search(
    _: Annotated[None, Depends(require_auth)],
    q: Annotated[str, Query(min_length=1, description="Query string")],
    k: Annotated[int, Query(ge=1, le=100, description="Top-K for ranked mode")] = 10,
    mode: Annotated[str, Query(pattern="^(ranked|boolean|phrase)$")] = "ranked",
) -> SearchOut:
    """
    Legacy search endpoint (kept for compatibility). For richer context/grouping, use /searchx.
    """
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
            start = hits[d][0]
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


@app.get("/searchx", response_model=SearchXOut)
def get_searchx(
    _: Annotated[None, Depends(require_auth)],
    q: Annotated[str, Query(min_length=1, description="Query string")],
    mode: Annotated[str, Query(pattern="^(ranked|phrase|boolean)$")] = "ranked",
    k: Annotated[int, Query(ge=1, le=100)] = 10,  # ranked mode
    context: Annotated[str, Query(pattern="^(paragraph|window)$")] = "paragraph",
    window: Annotated[int, Query(ge=0, le=10)] = 2,  # when context=window
    per_doc: Annotated[int, Query(ge=1, le=10)] = 3,  # hits per document
    group_by_doc: bool = True,  # reserved for future
) -> SearchXOut:
    """
    Rich search endpoint: returns multiple context blocks per document, grouped by doc.
    """
    # Mark as used to satisfy linters while keeping API parameter name stable
    _ = group_by_doc

    idx = _require_index()
    s = Searcher(idx, _ANALYZER)
    hi = Highlighter(idx)

    def _ctx(doc_id: str, start_pos: int) -> Hit:
        c: Ctx = hi.build_context(doc_id, start_pos, mode=context, window=window)
        return Hit(line=c.line, page=c.page, snippet=c.snippet)

    groups: list[DocGroup] = []

    if mode == "phrase":
        pm = PhraseMatcher(idx, _ANALYZER)
        hits = pm.match(q, keep_stopwords=True)  # dict[doc_id] -> [start_pos...]
        for d in sorted(hits.keys()):
            chosen = hits[d][:per_doc]
            groups.append(DocGroup(doc_id=d, score=None, hits=[_ctx(d, p) for p in chosen]))
        return SearchXOut(groups=groups)

    if mode == "boolean":
        doc_ids = s.search_boolean(q)
        terms = _ANALYZER.tokenize(q, keep_stopwords=False)
        for d in doc_ids:
            pos_candidates: list[int] = []
            for t in terms:
                posting = idx.get_posting(t)
                if posting and d in posting and posting[d]:
                    pos_candidates.append(posting[d][0])
            pos_chosen = sorted(set(pos_candidates))[:per_doc]
            if not pos_chosen:
                continue
            groups.append(DocGroup(doc_id=d, score=None, hits=[_ctx(d, p) for p in pos_chosen]))
        return SearchXOut(groups=groups)

    # ranked (BM25)
    ranked = s.search_ranked(q, k=k)  # list[(doc_id, score)]
    terms = _ANALYZER.tokenize(q, keep_stopwords=False)
    for d, score in ranked:
        pos_candidates: list[int] = []
        for t in terms:
            posting = idx.get_posting(t)
            if posting and d in posting and posting[d]:
                pos_candidates.append(posting[d][0])
        pos_chosen = sorted(set(pos_candidates))[:per_doc]
        if not pos_chosen:
            continue
        groups.append(DocGroup(doc_id=d, score=score, hits=[_ctx(d, p) for p in pos_chosen]))
    return SearchXOut(groups=groups)


# Minimal Web UI (improved)
_HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EviSearch-Py</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:980px;margin:28px auto;padding:0 16px}
h1{font-size:22px;margin:0 0 8px}
small,.muted{color:#666}
.card{border:1px solid #e5e7eb;border-radius:14px;padding:14px 16px;margin:12px 0;box-shadow:0 1px 2px rgba(0,0,0,.03)}
.row{display:flex;gap:12px;flex-wrap:wrap;align-items:center}
input[type="text"]{flex:1;min-width:280px;padding:10px;border:1px solid #ddd;border-radius:10px}
select,button{padding:10px;border-radius:10px;border:1px solid #ddd;background:#fff;cursor:pointer}
button.primary{background:#111;color:#fff;border-color:#111}
.badge{display:inline-block;border:1px solid #ddd;border-radius:999px;padding:2px 8px;font-size:12px;margin-left:8px;color:#555}
.doc{font-weight:600}
.snip{white-space:pre-wrap;background:#fafafa;border:1px dashed #eee;border-radius:10px;padding:10px;margin-top:8px}
.hr{height:1px;background:linear-gradient(90deg,#eee,#f9f9f9);margin:8px 0}
.ok{color:#0a7}.err{color:#c33}
</style>
</head>
<body>
<h1>EviSearch-Py</h1>
<div class="muted">Upload PDFs/TXT, then search (ranked / phrase / boolean). Context can be paragraph or ±N lines.</div>

<div class="card">
  <div class="row">
    <input id="files" type="file" multiple accept=".pdf,.txt">
    <button class="primary" id="btnIndex">Index files</button>
    <span id="idxMsg" class="muted"></span>
  </div>
  <small>Index is stored in memory; uploading again will replace the current index.</small>
</div>

<div class="card">
  <div class="row">
    <input id="q" type="text" placeholder='e.g. pue OR "power usage effectiveness"'>
    <select id="mode">
      <option value="ranked" selected>ranked (BM25)</option>
      <option value="phrase">phrase (exact)</option>
      <option value="boolean">boolean</option>
    </select>
    <select id="context">
      <option value="paragraph" selected>paragraph</option>
      <option value="window">±N lines</option>
    </select>
    <select id="window">
      <option>0</option><option>1</option><option selected>2</option><option>3</option><option>5</option>
    </select>
    <select id="perdoc">
      <option>1</option><option selected>3</option><option>5</option>
    </select>
    <select id="k">
      <option>5</option><option selected>10</option><option>20</option>
    </select>
    <button class="primary" id="btnSearch">Search</button>
    <span id="searchMsg" class="muted"></span>
  </div>
  <div id="results"></div>
</div>

<script>
const $ = (id)=>document.getElementById(id);

$("btnIndex").onclick = async () => {
  const files = $("files").files;
  if (!files.length) { $("idxMsg").textContent = "Choose files"; return; }
  $("idxMsg").textContent = "Indexing...";
  const fd = new FormData();
  for (const f of files) fd.append("files", f);
  try {
    const r = await fetch("/index-files", { method:"POST", body: fd });
    const j = await r.json();
    $("idxMsg").innerHTML = r.ok
      ? `<span class="ok">OK</span> indexed=${j.indexed}, vocab=${j.vocab}`
      : `<span class="err">Error:</span> ${j.detail||JSON.stringify(j)}`;
  } catch (e) {
    $("idxMsg").innerHTML = `<span class="err">Network error</span>`;
  }
};

$("btnSearch").onclick = async () => {
  const q = $("q").value.trim();
  if (!q) { $("searchMsg").textContent = "Enter a query"; return; }
  $("searchMsg").textContent = "Searching...";
  const url = new URL("/searchx", location.origin);
  url.searchParams.set("q", q);
  url.searchParams.set("mode", $("mode").value);
  url.searchParams.set("context", $("context").value);
  url.searchParams.set("window", $("window").value);
  url.searchParams.set("per_doc", $("perdoc").value);
  url.searchParams.set("group_by_doc", "true");
  url.searchParams.set("k", $("k").value);
  try {
    const r = await fetch(url);
    const j = await r.json();
    $("searchMsg").textContent = "";
    if (!r.ok) { $("results").innerHTML = `<div class="err">Error: ${j.detail||JSON.stringify(j)}</div>`; return; }
    if (!j.groups.length) { $("results").innerHTML = `<div class="muted">No results</div>`; return; }
    $("results").innerHTML = j.groups.map(g => `
      <div class="card">
        <div class="doc">${g.doc_id}
          ${g.score!=null?`<span class="badge">score ${g.score.toFixed(3)}</span>`:""}
          ${g.hits?.length?`<span class="badge">${g.hits.length} hit${g.hits.length>1?"s":""}</span>`:""}
        </div>
        <div class="hr"></div>
        ${g.hits.map(h => `
          <div class="muted">line ${h.line}${h.page?`, page ${h.page}`:""}</div>
          <div class="snip">${(h.snippet||"").replace(/</g,"&lt;")}</div>
        `).join("")}
      </div>`).join("");
  } catch (e) {
    $("results").innerHTML = `<div class="err">Network error</div>`;
  }
};
</script>
</body>
</html>
"""


@app.get("/ui", response_class=HTMLResponse)
def ui(_: Annotated[None, Depends(require_auth)]) -> HTMLResponse:
    return HTMLResponse(_HTML)
