# ruff: noqa
# mypy: ignore-errors
from __future__ import annotations

import io
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import fitz  # PyMuPDF
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

# -----------------------------------------------------------------------------
# Utilities & Globals
# -----------------------------------------------------------------------------

APP_TITLE = "EviSearch-Py"
DESCRIPTION = (
    "Evidence-level mini search engine: PDF/Text indexing, phrase/boolean/BM25 search, "
    "multi-hit per doc, page snapshots with zoom, file-level add/remove."
)

# Uploads dir (tmp, persisted across app lifetime)
UPLOAD_DIR = Path(tempfile.gettempdir()) / "evisearch_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Index structures—very small, in-memory, simplified
@dataclass
class Posting:
    # positions of term occurrences in doc
    positions: list[int]

@dataclass
class InvertedIndex:
    # map: term -> doc_id -> positions
    postings: dict[str, dict[str, list[int]]]
    # doc store: doc_id -> lines
    doc_lines: dict[str, list[str]]
    # doc_ids to keep order
    doc_ids: list[str]
    # optional page map: doc_id -> list of (start_pos_inclusive for each page)
    page_map: dict[str, list[int]]

    @property
    def vocabulary_size(self) -> int:
        return len(self.postings)

    def get_posting(self, term: str) -> dict[str, list[int]] | None:
        return self.postings.get(term)

# Minimal global index state
_INDEX: Optional[InvertedIndex] = None
_DOC_PATHS: dict[str, str] = {}

# -----------------------------------------------------------------------------
# Very small text analyzer (tokenization, lowercasing)
# -----------------------------------------------------------------------------

WORD_RE = re.compile(r"[A-Za-z0-9_\-]+")

def analyze_text(text: str) -> list[str]:
    # Lowercase and tokenize into "terms"
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]

# -----------------------------------------------------------------------------
# Index writer/reader
# -----------------------------------------------------------------------------

class IndexWriter:
    def __init__(self, index_stopwords: bool = True):
        self.postings: dict[str, dict[str, list[int]]] = {}
        self.doc_lines: dict[str, list[str]] = {}
        self.doc_ids: list[str] = []
        self.page_map: dict[str, list[int]] = {}

        self.index_stopwords = index_stopwords

    def add(self, doc_id: str, text: str, *, page_map: list[int] | None = None) -> None:
        if doc_id in self.doc_lines:
            # Replace if already exists (simple)
            self.remove(doc_id)

        # Store lines (for snippet)
        lines = text.splitlines()
        self.doc_lines[doc_id] = lines
        if doc_id not in self.doc_ids:
            self.doc_ids.append(doc_id)

        # Build postings (positions across entire document)
        pos = 0
        posting_map: dict[str, list[int]] = {}
        for ln in lines:
            terms = analyze_text(ln)
            for _t in terms:
                posting_map.setdefault(_t, []).append(pos)
                pos += 1
            # add 1 position gap per line to separate a bit
            pos += 1

        for t, plist in posting_map.items():
            self.postings.setdefault(t, {})[doc_id] = plist

        # Save optional page map
        if page_map:
            self.page_map[doc_id] = page_map

    def remove(self, doc_id: str) -> None:
        if doc_id in self.doc_lines:
            del self.doc_lines[doc_id]
        if doc_id in self.doc_ids:
            self.doc_ids.remove(doc_id)
        # Remove from postings
        remove_keys = []
        for term, dmap in self.postings.items():
            if doc_id in dmap:
                del dmap[doc_id]
            if not dmap:
                remove_keys.append(term)
        for k in remove_keys:
            del self.postings[k]
        if doc_id in self.page_map:
            del self.page_map[doc_id]

    def commit(self) -> InvertedIndex:
        return InvertedIndex(
            postings=self.postings,
            doc_lines=self.doc_lines,
            doc_ids=self.doc_ids,
            page_map=self.page_map,
        )

# -----------------------------------------------------------------------------
# PDF text extraction with page map
# -----------------------------------------------------------------------------

def extract_pdf_text_and_page_map(path: Path) -> tuple[str, list[int]]:
    """
    Return text concatenation of pages, and a page_map list where
    page_map[i] = starting position index (in terms) for page i+1.
    """
    doc = fitz.open(path)
    all_text_lines: list[str] = []
    page_map: list[int] = []
    pos = 0

    for pg in doc:
        page_map.append(pos)
        # simple text extraction line by line
        text = pg.get_text("text")
        lines = text.splitlines()
        all_text_lines.extend(lines)
        # rough position increments
        for ln in lines:
            terms = analyze_text(ln)
            pos += len(terms) + 1  # +1 as our line gap

    doc.close()
    return "\n".join(all_text_lines), page_map

# -----------------------------------------------------------------------------
# Highlighter / snippet
# -----------------------------------------------------------------------------

def highlight_terms(s: str, terms: list[str]) -> str:
    # very naive highlighting: wrap <mark>... for each term occurrence
    def repl(m: re.Match[str]) -> str:
        token = m.group(0)
        if token.lower() in terms:
            return f"<mark>{token}</mark>"
        return token

    return re.sub(r"[A-Za-z0-9_\-]+", repl, s)

def paragraph_snippet(doc_id: str, pos: int, terms: list[str]) -> dict[str, Any]:
    assert _INDEX is not None
    lines = _INDEX.doc_lines.get(doc_id, [])
    # find a line surrounding the approximate pos; we can't map exactly since pos was across doc
    # naive approach: choose some line by estimating
    # for better accuracy, we could store line->pos ranges; here keep simple
    # We'll just attempt to pick a line near pos // (avg terms per line + gap)
    # fallback: join a window of lines
    window = 3
    # a simpler approach: pick mid line if range unknown
    mid = min(len(lines) - 1, max(0, pos // 12))
    start = max(0, mid - window)
    end = min(len(lines), mid + window + 1)
    snippet_text = "\n".join(lines[start:end])
    html = "<pre>" + highlight_terms(snippet_text, terms) + "</pre>"
    return {
        "kind": "text",
        "doc_id": doc_id,
        "pos": pos,
        "snippet_html": html,
    }

# For table-like view, we can attempt to construct a pseudo "table snippet"
# This is a placeholder since we don't have structured tables in raw text.
# We present the snippet as text and, if we can map to a page, also provide a snapshot_url.
def table_snippet(doc_id: str, page: int, terms: list[str]) -> dict[str, Any] | None:
    assert _INDEX is not None
    if page < 1:
        page = 1
    pm = _INDEX.page_map.get(doc_id, [])
    if not pm:
        return None

    # Build a "snippet" for the page region
    lines = _INDEX.doc_lines.get(doc_id, [])
    # We don't track exact line boundaries per page; show a short "page summary"
    # (could be improved by storing page->line range at index time)
    left = max(0, (page - 1) * 40)
    right = min(len(lines), page * 40)
    page_text = "\n".join(lines[left:right])
    html = "<pre>" + highlight_terms(page_text, terms) + "</pre>"

    # Provide a snapshot url for the entire page (no bbox cropping here)
    snapshot_url = (
        f"/page-snapshot?doc_id={doc_id}&page={page}"
        f"&x0=0&y0=0&x1=10000&y1=10000"
    )
    return {
        "kind": "table",
        "doc_id": doc_id,
        "page": page,
        "bbox": [0, 0, 10000, 10000],
        "table_html": html,
        "snapshot_url": snapshot_url,
    }

# Map position to approximate page using page_map
def page_of_pos(doc_id: str, pos: int) -> int | None:
    assert _INDEX is not None
    pm = _INDEX.page_map.get(doc_id, [])
    if not pm:
        return None
    # pm is a sorted list of page starts; find largest start <= pos
    lo, hi = 0, len(pm) - 1
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if pm[mid] <= pos:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best + 1  # pages are 1-based

# -----------------------------------------------------------------------------
# Search (phrase / boolean / ranked) — minimal implementations
# -----------------------------------------------------------------------------

def search_phrase(query: str, per_doc: int) -> list[dict[str, Any]]:
    """
    Very naive phrase matching: split into terms and look for consecutive positions
    in docs. Return up to `per_doc` hits per doc.
    """
    assert _INDEX is not None
    terms = analyze_text(query)
    if not terms:
        return []

    # gather candidate doc ids: intersection of postings
    candidate_docs: Optional[set[str]] = None
    for t in terms:
        dmap = _INDEX.get_posting(t) or {}
        docs = set(dmap.keys())
        candidate_docs = docs if candidate_docs is None else (candidate_docs & docs)
        if not candidate_docs:
            return []

    results: list[dict[str, Any]] = []
    for doc_id in sorted(candidate_docs or []):
        positions_lists = [_INDEX.get_posting(t)[doc_id] for t in terms]  # type: ignore[index]
        # find consecutive positions
        # naive: for each pos in first list, see if pos+1 in second list, pos+2 in third list, ...
        first = positions_lists[0]
        hits: list[int] = []
        pos_set_next = None
        # Build set chain to speed up basic membership tests
        sets = [set(lst) for lst in positions_lists[1:]]
        for p in first:
            ok = True
            for j, s in enumerate(sets, start=1):
                if (p + j) not in s:
                    ok = False
                    break
            if ok:
                hits.append(p)
            if len(hits) >= per_doc:
                break

        out_hits: list[dict[str, Any]] = []
        for hp in hits[:per_doc]:
            page = page_of_pos(doc_id, hp) or 1
            table = table_snippet(doc_id, page, terms)
            if table:
                out_hits.append(table)
            else:
                out_hits.append(paragraph_snippet(doc_id, hp, terms))

        results.append(
            {"doc_id": doc_id, "hits": out_hits}
        )
    return results

def search_boolean(terms: list[str], per_doc: int) -> list[dict[str, Any]]:
    """
    Return docs that contain all terms. For each doc, return up to per_doc hit snippets
    based on positional occurrences (not consecutive).
    """
    assert _INDEX is not None
    if not terms:
        return []

    # Intersection for AND
    candidate_docs: Optional[set[str]] = None
    for t in terms:
        dmap = _INDEX.get_posting(t) or {}
        docs = set(dmap.keys())
        candidate_docs = docs if candidate_docs is None else (candidate_docs & docs)
        if not candidate_docs:
            return []

    results: list[dict[str, Any]] = []
    for doc_id in sorted(candidate_docs or []):
        positions: list[int] = []
        for t in terms:
            dmap = _INDEX.get_posting(t) or {}
            positions.extend(dmap.get(doc_id, []))
        positions = sorted(set(positions))[:max(1, per_doc)]
        out_hits: list[dict[str, Any]] = []
        for hp in positions:
            page = page_of_pos(doc_id, hp) or 1
            table = table_snippet(doc_id, page, terms)
            if table:
                out_hits.append(table)
            else:
                out_hits.append(paragraph_snippet(doc_id, hp, terms))
        results.append({"doc_id": doc_id, "hits": out_hits})
    return results

def search_ranked(terms: list[str], per_doc: int) -> list[dict[str, Any]]:
    """
    Super-minimal BM25-ish scoring replaced with a simple count-based score for demo.
    Return up to per_doc hit snippets per doc.
    """
    assert _INDEX is not None
    if not terms:
        return []
    scores: list[tuple[str, float]] = []
    for doc_id in _INDEX.doc_ids:
        # score = total occurrences across all terms
        sc = 0.0
        for t in terms:
            dmap = _INDEX.get_posting(t) or {}
            sc += float(len(dmap.get(doc_id, [])))
        if sc > 0.0:
            scores.append((doc_id, sc))
    scores.sort(key=lambda x: x[1], reverse=True)

    results: list[dict[str, Any]] = []
    for doc_id, sc in scores:
        positions: list[int] = []
        for t in terms:
            dmap = _INDEX.get_posting(t) or {}
            positions.extend(dmap.get(doc_id, []))
        positions = sorted(set(positions))[:max(1, per_doc)]
        out_hits: list[dict[str, Any]] = []
        for hp in positions:
            page = page_of_pos(doc_id, hp) or 1
            table = table_snippet(doc_id, page, terms)
            if table:
                out_hits.append(table)
            else:
                out_hits.append(paragraph_snippet(doc_id, hp, terms))
        results.append({"doc_id": doc_id, "score": sc, "hits": out_hits})
    return results

# -----------------------------------------------------------------------------
# FastAPI models
# -----------------------------------------------------------------------------

class IndexOut(BaseModel):
    ok: bool
    indexed: int
    vocab: int

class SearchHit(BaseModel):
    # A hit can be text or table
    kind: str
    doc_id: str
    snippet_html: Optional[str] = None
    table_html: Optional[str] = None
    page: Optional[int] = None
    bbox: Optional[list[float]] = None
    snapshot_url: Optional[str] = None
    pos: Optional[int] = None

class SearchDocResult(BaseModel):
    doc_id: str
    score: Optional[float] = None
    hits: list[SearchHit]

class SearchOut(BaseModel):
    ok: bool
    mode: str
    q: str
    results: list[SearchDocResult]

# -----------------------------------------------------------------------------
# Auth (no-op placeholder)
# -----------------------------------------------------------------------------

def check_auth() -> None:
    return None

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------

app = FastAPI(title=APP_TITLE, description=DESCRIPTION)

# -----------------------------------------------------------------------------
# UI: Single-page minimal HTML (inline CSS/JS)
# -----------------------------------------------------------------------------

UI_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>EviSearch-Py UI</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; padding: 24px; color: #222; }
    h1 { margin: 0 0 16px; font-size: 20px; }
    .bar { display: grid; grid-template-columns: 1fr auto; gap: 8px; align-items: center; margin-bottom: 12px; }
    input[type=text] { font-size: 14px; padding: 8px 10px; width: 100%; border: 1px solid #ddd; border-radius: 8px; }
    button { padding: 6px 12px; border: 1px solid #ddd; background: #fafafa; border-radius: 8px; cursor: pointer; }
    button:hover { background: #f0f0f0; }
    .muted { color: #666; }
    .small { font-size: 12px; }
    .row { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 8px 10px; border: 1px solid #eee; border-radius: 8px; margin-bottom: 6px; background: #fff; }
    .name { font-weight: 600; }
    .status { font-size: 12px; color: #555; }
    .progress-wrap { position: relative; height: 6px; background: #f2f2f2; border-radius: 999px; flex: 1; }
    .progress { position: absolute; inset: 0 0 0 0; width: 0%; background: #4caf50; border-radius: 999px; }
    .badge { font-size: 12px; padding: 2px 6px; border: 1px solid #ddd; border-radius: 999px; }
    .close { font-weight: 700; cursor: pointer; padding: 0 8px; }

    .dropzone { padding: 16px; border: 2px dashed #ddd; text-align: center; border-radius: 12px; margin: 10px 0; }
    .dropzone.drag { background: #fafafa; border-color: #ccc; }

    .controls { display: flex; gap: 8px; align-items: center; }

    .cards { display: grid; grid-template-columns: 1fr; gap: 12px; margin-top: 16px; }
    .card { border: 1px solid #eee; border-radius: 12px; padding: 12px; background: #fff; }
    .card h3 { margin: 0 0 6px; font-size: 16px; }
    .hits { display: grid; gap: 8px; margin-top: 10px; }
    .hit { display: grid; grid-template-columns: 1fr auto; gap: 12px; border: 1px dashed #eee; padding: 8px; border-radius: 8px; }
    .left pre { margin: 0; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; white-space: pre-wrap; }
    .right img.snapshot { max-width: 420px; cursor: zoom-in; border-radius: 6px; border: 1px solid #eee; }

    .kv { display: flex; gap: 12px; align-items: center; }

    #lightbox { position: fixed; inset: 0; display: none; place-items: center; background: rgba(0,0,0,.65); z-index: 9999; }
    #lightbox img { max-width: 90vw; max-height: 90vh; background: #000; }
  </style>
</head>
<body>
  <h1>EviSearch-Py</h1>
  <div class="bar">
    <input id="q" type="text" placeholder="Search phrase / boolean / ranked (e.g., pue & wue or &quot;power usage effectiveness&quot;)" />
    <div class="controls">
      <select id="mode">
        <option value="ranked">ranked</option>
        <option value="boolean">boolean</option>
        <option value="phrase">phrase</option>
      </select>
      <label class="small muted">per doc</label>
      <input id="perdoc" type="number" min="1" max="10" value="3" style="width:60px;"/>
      <button id="go">Search</button>
    </div>
  </div>

  <div class="dropzone" id="dropzone">
    <div class="muted">Drag & drop files here, or <label style="color:#06c;cursor:pointer;"><u>browse</u><input id="file" type="file" multiple style="display:none"/></label></div>
  </div>

  <div id="files"></div>

  <div style="margin-top: 10px;">
    <button id="rebuild">Rebuild index from uploaded files</button>
    <span class="small muted" id="stat"></span>
  </div>

  <div class="cards" id="results"></div>

  <div id="lightbox"><img/></div>

  <script>
    const fileListEl = document.getElementById('files');
    const results = document.getElementById('results');
    const statEl = document.getElementById('stat');
    const perdocEl = document.getElementById('perdoc');
    const qEl = document.getElementById('q');
    const modeEl = document.getElementById('mode');
    const dropzone = document.getElementById('dropzone');
    const inputFile = document.getElementById('file');

    const uploads = new Map(); // filename -> xhr

    function fileRow(name) {
      const row = document.createElement('div');
      row.className = 'row';
      row.innerHTML = `
        <div class="name" title="${name}">${name}</div>
        <div class="progress-wrap"><div class="progress"></div></div>
        <div class="badge">0%</div>
        <div class="status muted small">pending</div>
        <div class="close" title="Remove">×</div>
      `;
      return row;
    }

    async function refreshState() {
      // very small stats about index
      try {
        const r = await fetch('/state');
        if (!r.ok) throw 0;
        const s = await r.json();
        statEl.textContent = `Indexed docs: ${s.indexed} · Vocab: ${s.vocab}`;
      } catch {}
    }

    async function rebuildFromUploads() {
      // send all current files under UPLOAD_DIR again (server will read from dir)
      const r = await fetch('/index-files', { method: 'POST' });
      if (r.ok) {
        await refreshState();
        alert('Rebuilt index from uploaded files.');
      } else {
        alert('Failed to rebuild.');
      }
    }

    document.getElementById('rebuild').onclick = rebuildFromUploads;

    function uploadOne(file) {
      const row = fileRow(file.name);
      fileListEl.prepend(row);

      const xhr = new XMLHttpRequest();
      uploads.set(file.name, xhr);

      xhr.upload.onprogress = (e) => {
        if (!e.lengthComputable) return;
        const pct = Math.round((e.loaded / e.total) * 100);
        row.querySelector('.progress').style.width = pct + '%';
        row.querySelector('.badge').textContent = pct + '%';
      };

      xhr.onreadystatechange = () => {
        if (xhr.readyState === 4) {
          uploads.delete(file.name);
          if (xhr.status >= 200 && xhr.status < 300) {
            row.querySelector('.status').textContent = 'done';
            refreshState();
          } else if (xhr.status !== 0) {
            row.querySelector('.status').textContent = 'error';
          }
        }
      };

      row.querySelector('.close').onclick = async () => {
        const running = uploads.get(file.name);
        if (running) {
          running.abort();
          row.remove();
          return;
        }
        // already uploaded: delete by doc_id (name without extension)
        const docId = file.name.replace(/\\.[^.]+$/, '');
        const r = await fetch('/doc/' + encodeURIComponent(docId), { method: 'DELETE' });
        if (r.ok) row.remove();
        refreshState();
      };

      const fd = new FormData();
      fd.append('file', file);
      xhr.open('POST', '/add-file');
      xhr.send(fd);
    }

    dropzone.ondragover = (e) => {
      e.preventDefault(); dropzone.classList.add('drag');
    };
    dropzone.ondragleave = () => dropzone.classList.remove('drag');
    dropzone.ondrop = (e) => {
      e.preventDefault(); dropzone.classList.remove('drag');
      const fs = Array.from(e.dataTransfer.files || []);
      for (const f of fs) uploadOne(f);
    };

    inputFile.onchange = () => {
      const fs = Array.from(inputFile.files || []);
      for (const f of fs) uploadOne(f);
      inputFile.value = '';
    };

    document.getElementById('go').onclick = async () => {
      const q = qEl.value.trim();
      if (!q) return;
      const mode = modeEl.value;
      const perdoc = Math.min(10, Math.max(1, parseInt(perdocEl.value || '3', 10)));
      const url = `/search?q=${encodeURIComponent(q)}&mode=${encodeURIComponent(mode)}&per_doc=${perdoc}`;
      const r = await fetch(url);
      if (!r.ok) {
        results.innerHTML = '<div class="muted">Search failed.</div>';
        return;
      }
      const data = await r.json();
      renderResults(data);
    };

    function renderResults(data) {
      const cards = [];
      for (const row of data.results || []) {
        const header = `<h3>${row.doc_id} <span class="muted small">${
          row.score != null ? 'score ' + row.score.toFixed(1) : ''
        } · ${row.hits?.length || 0} hit(s)</span></h3>`;

        const hitHtml = (row.hits || []).map(h => {
          if (h.kind === 'table') {
            const snap = h.snapshot_url
              ? `<div class="right"><img class="snapshot" src="${h.snapshot_url}"/></div>` : '';
            return `<div class="hit"><div class="left">${h.table_html || ''}</div>${snap}</div>`;
          }
          return `<div class="hit"><div class="left">${h.snippet_html || ''}</div></div>`;
        }).join('');

        cards.push(`<div class="card">${header}<div class="hits">${hitHtml}</div></div>`);
      }
      results.innerHTML = cards.join('') || '<div class="muted">No results.</div>';
    }

    // Lightbox
    const lb = document.getElementById('lightbox');
    results.onclick = (e) => {
      const img = e.target.closest('img.snapshot');
      if (!img) return;
      const big = img.src.includes('scale=') ? img.src : (img.src + '&scale=3');
      lb.querySelector('img').src = big;
      lb.style.display = 'grid';
    };
    lb.onclick = () => {
      lb.style.display = 'none';
      lb.querySelector('img').src = '';
    };

    refreshState();
  </script>
</body>
</html>
"""

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.get("/ui", response_class=HTMLResponse)
def ui() -> str:
    return UI_HTML

@app.get("/")
def root() -> dict[str, Any]:
    return {"ok": True, "app": APP_TITLE}

@app.get("/state")
def state() -> dict[str, Any]:
    if _INDEX is None:
        return {"indexed": 0, "vocab": 0}
    return {"indexed": len(_INDEX.doc_ids), "vocab": _INDEX.vocabulary_size}

# Index whole upload dir (replace index)
@app.post("/index-files", response_model=IndexOut)
def index_files(_auth: None = Depends(check_auth)) -> IndexOut:
    global _INDEX, _DOC_PATHS
    writer = IndexWriter(index_stopwords=True)
    _DOC_PATHS = {}

    # Iterate files in UPLOAD_DIR
    for p in sorted(UPLOAD_DIR.glob("*")):
        if not p.is_file():
            continue
        name = p.name
        doc_id = os.path.splitext(name)[0]
        text = ""
        page_map: list[int] = []

        # PDF vs text
        if p.suffix.lower() == ".pdf":
            try:
                text, page_map = extract_pdf_text_and_page_map(p)
            except Exception:
                text = ""
        else:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                text = ""

        writer.add(doc_id, text, page_map=page_map if page_map else None)
        _DOC_PATHS[doc_id] = str(p)

    _INDEX = writer.commit()
    return IndexOut(ok=True, indexed=len(_INDEX.doc_ids), vocab=_INDEX.vocabulary_size if _INDEX else 0)

# Add single file (append)
@app.post("/add-file", response_model=IndexOut)
async def add_file(
    file: UploadFile = File(...),
    _auth: None = Depends(check_auth),
) -> IndexOut:
    global _INDEX, _DOC_PATHS
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")

    name = file.filename or "doc"
    # store physically
    p = UPLOAD_DIR / name
    p.write_bytes(data)

    # prepare text/page map
    if p.suffix.lower() == ".pdf":
        try:
            text, page_map = extract_pdf_text_and_page_map(p)
        except Exception:
            text, page_map = "", []
    else:
        try:
            text = data.decode("utf-8", errors="ignore")
        except Exception:
            text = ""
        page_map = []

    doc_id = os.path.splitext(name)[0]

    # rebuild index by carrying over existing docs then adding new one
    writer = IndexWriter(index_stopwords=True)
    if _INDEX is not None:
        for did in _INDEX.doc_ids:
            exist_lines = _INDEX.doc_lines.get(did, [])
            exist_text = "\n".join(exist_lines)
            exist_pm = _INDEX.page_map.get(did, [])
            writer.add(did, exist_text, page_map=exist_pm if exist_pm else None)

    writer.add(doc_id, text, page_map=page_map if page_map else None)
    _INDEX = writer.commit()
    _DOC_PATHS[doc_id] = str(p)
    return IndexOut(ok=True, indexed=len(_INDEX.doc_ids), vocab=_INDEX.vocabulary_size)

# Delete an indexed doc (and optional physical file)
@app.delete("/doc/{doc_id}", response_model=IndexOut)
def delete_doc(doc_id: str, _auth: None = Depends(check_auth)) -> IndexOut:
    global _INDEX, _DOC_PATHS
    if _INDEX is None or doc_id not in _INDEX.doc_ids:
        raise HTTPException(404, "Doc not found")

    # try remove file
    p = Path(_DOC_PATHS.get(doc_id, ""))
    try:
        if p.exists():
            p.unlink()
    except Exception:
        pass

    # rebuild index without that doc
    writer = IndexWriter(index_stopwords=True)
    for did in _INDEX.doc_ids:
        if did == doc_id:
            continue
        exist_lines = _INDEX.doc_lines.get(did, [])
        exist_text = "\n".join(exist_lines)
        exist_pm = _INDEX.page_map.get(did, [])
        writer.add(did, exist_text, page_map=exist_pm if exist_pm else None)
    _INDEX = writer.commit()
    _DOC_PATHS.pop(doc_id, None)
    return IndexOut(ok=True, indexed=len(_INDEX.doc_ids), vocab=_INDEX.vocabulary_size)

# Search
@app.get("/search", response_model=SearchOut)
def search(
    q: str = Query(..., min_length=1),
    mode: str = Query("ranked", pattern="^(ranked|boolean|phrase)$"),
    per_doc: int = Query(3, ge=1, le=10),
    _auth: None = Depends(check_auth),
) -> SearchOut:
    assert _INDEX is not None, "Index is empty. Upload files and (re)build index."
    terms = analyze_text(q)
    if not terms:
        return SearchOut(ok=True, mode=mode, q=q, results=[])

    if mode == "phrase" and ('"' in q or len(terms) > 1):
        # phrase: treat q as phrase (quote optional)
        res = search_phrase(q.replace('"', ' '), per_doc)
    elif mode == "boolean":
        # boolean: a & b, a | b not implemented; we do AND of all terms
        res = search_boolean(terms, per_doc)
    else:
        # ranked
        res = search_ranked(terms, per_doc)

    # Cast to Pydantic models
    out: list[SearchDocResult] = []
    for r in res:
        hits = []
        for h in r.get("hits", []):
            hits.append(SearchHit(**h))
        out.append(
            SearchDocResult(
                doc_id=r["doc_id"],
                score=r.get("score"),
                hits=hits,
            )
        )

    return SearchOut(ok=True, mode=mode, q=q, results=out)

# Page snapshot (PNG), with zoom scale
@app.get("/page-snapshot")
def page_snapshot(
    doc_id: str,
    page: int = Query(..., ge=1),
    x0: float = Query(0.0),
    y0: float = Query(0.0),
    x1: float = Query(10000.0),
    y1: float = Query(10000.0),
    scale: float = Query(2.0, ge=1.0, le=4.0),
    _auth: None = Depends(check_auth),
) -> StreamingResponse:
    global _DOC_PATHS
    p = _path_for_doc(doc_id)
    if not p:
        raise HTTPException(404, "Doc not found or no file on disk.")
    if p.suffix.lower() != ".pdf":
        raise HTTPException(400, "Snapshots available only for PDFs.")

    doc = fitz.open(p)
    if page < 1 or page > len(doc):
        doc.close()
        raise HTTPException(400, f"Page out of range: [1..{len(doc)}]")

    pg = doc[page - 1]
    rect = fitz.Rect(x0, y0, x1, y1)
    # If rect is default "huge", just render full page
    if rect.width <= 1 or rect.height <= 1 or rect.width > 9000 or rect.height > 9000:
        rect = pg.rect

    mat = fitz.Matrix(scale, scale)
    pix = pg.get_pixmap(matrix=mat, clip=rect)
    data = pix.tobytes("png")
    doc.close()
    return StreamingResponse(io.BytesIO(data), media_type="image/png")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _safe_doc_id(name: str) -> str:
    n = name.strip().replace("/", "_").replace("\\", "_")
    return n

def _path_for_doc(doc_id: str) -> Optional[Path]:
    p = _DOC_PATHS.get(doc_id)
    if not p:
        # fallback: try to find file in uploads dir
        pdf = UPLOAD_DIR / f"{doc_id}.pdf"
        if pdf.exists():
            return pdf
        # any extension
        for cand in UPLOAD_DIR.glob(f"{doc_id}.*"):
            if cand.is_file():
                return cand
        return None
    pp = Path(p)
    return pp if pp.exists() else None
