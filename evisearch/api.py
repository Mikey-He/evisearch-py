# ruff: noqa: E501
from __future__ import annotations

import html
import io
import os
from pathlib import Path
import re
import threading
from typing import Annotated, Any

from fastapi import Body, Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import fitz  # PyMuPDF  # type: ignore[import-untyped]
import pdfplumber
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

from .analyzer import Analyzer
from .indexer import IndexWriter, InvertedIndex
from .searcher import PhraseMatcher, Searcher

# App & global state

app = FastAPI(title="EviSearch-Py", version="1.3.0")

DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads"
DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_ANALYZER = Analyzer()
_INDEX_LOCK = threading.Lock()
_INDEX: InvertedIndex | None = None

# doc_id -> original file path (only for uploaded PDFs/TXTs)
_DOC_PATHS: dict[str, str] = {}

# windows for snippets
LINES_WINDOW = 1  # text: show hit line ±N lines
COL_WINDOW = 1  # table: show hit col ±N columns

# Maximum hits to return
MAX_HITS_DEFAULT = 3  # Default number of hits to show
MAX_HITS_ALL = 50     # Maximum when showing all

# Optional Basic Auth (via env BASIC_USER/BASIC_PASS)

basic_user = os.getenv("BASIC_USER") or ""
basic_pass = os.getenv("BASIC_PASS") or ""
security = HTTPBasic()


def _needs_auth() -> bool:
    return bool(basic_user and basic_pass)


def _check_auth(
    credentials: HTTPBasicCredentials = Depends(security)  # noqa: B008
) -> None:
    if not _needs_auth():
        return
    ok = credentials.username == basic_user and credentials.password == basic_pass
    if not ok:
        raise HTTPException(status_code=401, detail="Unauthorized")


# Models


class DocIn(BaseModel):
    id: str
    text: str
    page_map: list[tuple[int, int]] = Field(default_factory=list)


class IndexIn(BaseModel):
    docs: list[DocIn]


class IndexOut(BaseModel):
    ok: bool
    indexed: int
    vocab: int


class SearchOut(BaseModel):
    results: list[dict[str, Any]]


# Helpers
def _safe_doc_id(name: str) -> str:
    base = os.path.basename(name or "doc")
    base = re.sub(r"[^\w\-.]+", "_", base)
    return base


def _extract_pdf_text_and_page_map(
    path: Path,
) -> tuple[str, list[tuple[int, int]]]:
    """
    Extract page texts and a page_map: (token_pos_offset, 1-based page_no).
    """
    texts: list[str] = []
    page_map: list[tuple[int, int]] = []
    pos = 0

    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text() or ""
            texts.append(txt)
            toks = list(_ANALYZER.iter_tokens(txt, keep_stopwords=True))
            page_map.append((pos, i))
            pos += len(toks)

    joined = "\n\n".join(texts)
    return joined, page_map


def _page_of_pos(doc_id: str, pos: int) -> int | None:
    if _INDEX is None:
        return None
    pm = _INDEX.page_map.get(doc_id)
    if not pm:
        return None
    page = None
    for start_pos, pg in pm:
        if start_pos <= pos:
            page = pg
        else:
            break
    return page


def _representative_pos(doc_id: str, terms: list[str]) -> int | None:
    """Pick earliest posting position of any term in the doc."""
    if _INDEX is None:
        return None
    best = None
    for t in terms:
        posting = _INDEX.get_posting(t)
        if not posting:
            continue
        plist = posting.get(doc_id, [])
        if plist:
            cand = plist[0]
            if best is None or cand < best:
                best = cand
    return best


def _all_term_positions(
    doc_id: str, 
    terms: list[str]
) -> list[int]:
    """Get all positions where any term appears in the doc."""
    if _INDEX is None:
        return []
    positions = []
    for t in terms:
        posting = _INDEX.get_posting(t)
        if not posting:
            continue
        plist = posting.get(doc_id, [])
        positions.extend(plist)
    return sorted(set(positions))


def _highlight_terms(text: str, terms: list[str]) -> str:
    """
    Escape HTML, then wrap matches with <mark> (word boundary, case-insensitive).
    """
    if not terms:
        return html.escape(text)
    pats = sorted(set(t for t in terms if t), key=len, reverse=True)
    rx = re.compile(
        r"\b(" + "|".join(re.escape(t) for t in pats) + r")\b",
        re.IGNORECASE,
    )
    return rx.sub(
        lambda m: f"<mark>{html.escape(m.group(0))}</mark>", 
        html.escape(text)
    )


def _paragraph_snippet(
    doc_id: str, 
    start_pos: int | None, 
    terms: list[str]
) -> dict[str, Any]:
    """
    Build a line-window snippet around the hit line:
    [hit-LINES_WINDOW, hit+LINES_WINDOW].
    """
    if _INDEX is None or start_pos is None:
        return {"kind": "text", "snippet_html": "", "line": 0, "page": None}

    lines = _INDEX.doc_lines.get(doc_id, [])
    line_of_pos = _INDEX.line_of_pos.get(doc_id, [])
    if not lines or not line_of_pos or start_pos >= len(line_of_pos):
        return {"kind": "text", "snippet_html": "", "line": 0, "page": None}

    hit = line_of_pos[start_pos]
    s = max(0, hit - LINES_WINDOW)
    e = min(len(lines) - 1, hit + LINES_WINDOW)
    block = "\n".join(lines[s : e + 1])

    page = _page_of_pos(doc_id, start_pos)
    html_block = (
        "<pre class='snippet'>" + 
        _highlight_terms(block, terms) + 
        "</pre>"
    )
    return {
        "kind": "text", 
        "snippet_html": html_block, 
        "line": hit + 1, 
        "page": page
    }


def _is_numericish(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return False
    return bool(re.fullmatch(r"[+-]?\d[\d,]*(\.\d+)?%?", s))


def _table_snippet(
    doc_id: str,
    page_no: int,
    terms: list[str],
    cwin: int = COL_WINDOW,
    rwin: int = 1,
) -> dict[str, Any] | None:
    """
    Return a dict for a table window if found:
      {
        "kind": "table",
        "table_html": "<table>...</table>",
        "bbox": [x0, y0, x1, y1],
        "page": 1-based page number
      }
    """
    pdf_path = _DOC_PATHS.get(doc_id)
    if not pdf_path or not os.path.exists(pdf_path):
        return None

    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_no < 0 or page_no >= len(pdf.pages):
                return None
            page = pdf.pages[page_no]

            tables = page.find_tables(
                table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "intersection_tolerance": 5,
                }
            )
            if not tables:
                tables = page.find_tables()

            terms_l = [t.lower() for t in terms if t]
            best: tuple[int, Any] | None = None
            for tb in tables:
                grid = tb.extract()
                flat = " ".join((c or "") for row in grid for c in row)
                hits = sum(
                    1 for t in terms_l 
                    if re.search(rf"\b{re.escape(t)}\b", flat, re.I)
                )
                if hits > 0 and (best is None or hits > best[0]):
                    best = (hits, tb)

            if best is None:
                return None

            tb = best[1]
            grid = tb.extract()

            # locate a hit cell
            hit_rc: list[tuple[int, int]] = []
            for ri, row in enumerate(grid):
                for ci, cell in enumerate(row):
                    val = (cell or "").lower()
                    if any(
                        re.search(rf"\b{re.escape(t)}\b", val, re.I) 
                        for t in terms_l
                    ):
                        hit_rc.append((ri, ci))
            if not hit_rc:
                return None

            r0, c0 = hit_rc[0]
            r1, c1 = r0, c0
            r0 = max(0, r0 - rwin)
            r1 = min(len(grid) - 1, r1 + rwin)

            max_cols = max(len(row) for row in grid) if grid else 0
            c0 = max(0, c0 - cwin)
            c1 = min(
                (len(grid[0]) - 1) if grid and grid[0] else max_cols - 1,
                c1 + cwin,
            )

            header_row_idx = 0
            first_row_empty = (
                grid and 
                not any((grid[0][ci] or "").strip() for ci in range(len(grid[0])))
            )
            if first_row_empty:
                for rr in range(r0 - 1, -1, -1):
                    row_not_empty = any(
                        (grid[rr][ci] or "").strip() 
                        for ci in range(len(grid[rr]))
                    )
                    if row_not_empty:
                        header_row_idx = rr
                        break

            sub_rows: list[list[str]] = []
            for rr in range(max(header_row_idx, r0), r1 + 1):
                row = grid[rr]
                if len(row) <= c1:
                    row = row + [""] * (c1 - len(row) + 1)
                cells = [
                    _highlight_terms((row[cc] or "").strip(), terms) 
                    for cc in range(c0, c1 + 1)
                ]
                sub_rows.append(cells)

            def tr(cells: list[str], th: bool = False) -> str:
                tag = "th" if th else "td"
                cell_html = "".join(
                    f"<{tag}>{c}</{tag}>" for c in cells
                )
                return "<tr>" + cell_html + "</tr>"

            html_rows = []
            if sub_rows:
                html_rows.append(tr(sub_rows[0], th=True))
                for r in sub_rows[1:]:
                    html_rows.append(tr(r, th=False))

            x0, y0, x1, y1 = tb.bbox  # type: ignore[attr-defined]
            table_html = (
                "<table class='snippet-table'>" + 
                "".join(html_rows) + 
                "</table>"
            )
            return {
                "kind": "table",
                "table_html": table_html,
                "bbox": [float(x0), float(y0), float(x1), float(y1)],
                "page": page_no + 1,
            }
    except Exception:
        return None


def _auto_mode(q: str) -> str:
    q2 = q.strip()
    if not q2:
        return "ranked"
    if '"' in q2 or "'" in q2:
        return "phrase"
    if re.search(r"\b(and|or|not)\b|\(|\)", q2, re.IGNORECASE):
        return "boolean"
    return "ranked"


# Routes


@app.get("/", response_class=JSONResponse)
def root_status() -> dict[str, Any]:
    return {
        "app": "EviSearch-Py",
        "docs": len(_INDEX.doc_ids) if _INDEX else 0,
        "vocab": _INDEX.vocabulary_size() if _INDEX else 0,
        "auth": "on" if _needs_auth() else "off",
    }


@app.get("/ui", response_class=HTMLResponse)
def ui_page() -> HTMLResponse:
    # Enhanced UI with file management and image zoom
    return HTMLResponse(  # noqa: E501
        """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>EviSearch-Py</title>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <style>
    :root {
      --bg:#0b0f18; --fg:#e7eaf2; --muted:#9aa3b2; --card:#131a27;
      --accent:#2ea043; --mark:#fff59d; --line:#2b3a55;
      --danger:#dc3545;
    }
    html,body{
      background:var(--bg); color:var(--fg);
      font:16px/1.45 system-ui,Segoe UI,Roboto,Arial;
    }
    .wrap{max-width:960px;margin:32px auto;padding:0 16px;}
    h1{font-size:28px;margin:0 0 12px;}
    .sub{color:var(--muted);margin-bottom:16px}
    .zone{
      border:2px dashed #30405c; border-radius:14px; padding:18px;
      display:flex; gap:12px; align-items:center; background:var(--card)
    }
    .zone.drag{outline:2px solid #4b6cb7}
    .btn{
      background:#1f2937; color:var(--fg);
      border:1px solid var(--line); border-radius:10px;
      padding:8px 12px; cursor:pointer
    }
    .btn.primary{background:#2ea043;border-color:#2ea043;color:#08110a}
    .btn.danger{
      background:transparent; color:var(--danger);
      border:none; padding:4px; cursor:pointer;
      font-size:20px; line-height:1;
    }
    .btn.danger:hover{color:#ff4458;}
    .muted{color:var(--muted)}
    .list{margin-top:10px}
    .file{
      display:flex; align-items:center; gap:10px; margin:8px 0; padding:8px;
      border:1px solid var(--line); border-radius:10px; background:var(--card);
      position: relative;
    }
    .file-name{flex:1; min-width:0; overflow:hidden; text-overflow:ellipsis;}
    progress{width:160px;height:10px}
    .controls{display:flex;gap:10px;margin:18px 0}
    input[type=text]{
      flex:1; min-width:0; background:#0e1420;
      border:1px solid var(--line); border-radius:10px; padding:10px; color:var(--fg)
    }
    .card{
      margin:16px 0; padding:14px;
      border:1px solid var(--line); border-radius:12px; background:var(--card)
    }
    .doc{font-weight:700; margin-bottom:10px;}
    .badge{
      font-size:12px; color:#0b2913; background:#1c4e2f;
      border:1px solid #2ea043; padding:1px 6px; border-radius:999px
    }
    pre.snippet{
      white-space:pre-wrap; background:#0e1420;
      border:1px solid #1d2a44; padding:10px; border-radius:10px;
      margin:8px 0;
    }
    table.snippet-table{
      border-collapse:collapse;background:#0e1420; margin:8px 0;
    }
    table.snippet-table th, table.snippet-table td{
      border:1px solid var(--line); padding:6px 8px
    }
    mark{background:var(--mark);color:#222}
    .hit{margin:12px 0; padding:12px; background:#0e1420; border-radius:8px;}
    .hit .meta{color:var(--muted); font-size:14px; margin-bottom:8px;}
    .hit .body{display:flex; gap:16px; align-items:flex-start}
    .hit .left{flex:1 1 60%;}
    .hit .right{flex:0 0 auto}
    .hit .right img{
      max-width:420px;
      border:1px solid var(--line);
      border-radius:8px;
      cursor:zoom-in;
    }
    .modal{
      display:none; position:fixed; z-index:1000;
      left:0; top:0; width:100%; height:100%;
      background:rgba(0,0,0,0.9); overflow:auto;
    }
    .modal.active{display:flex; align-items:center; justify-content:center;}
    .modal-content{
      max-width:90%; max-height:90%; 
      cursor:zoom-out; position:relative;
    }
    .modal-close{
      position:absolute; top:20px; right:35px;
      color:#f1f1f1; font-size:40px; font-weight:bold;
      cursor:pointer;
    }
    .modal-close:hover{color:#fff;}
    .show-all{
      color:var(--accent); cursor:pointer; 
      margin-left:10px; font-size:14px;
      text-decoration:underline;
    }
    .show-all:hover{color:#3fb254;}
  </style>
</head>
<body>
<div class="wrap">
  <h1>EviSearch-Py</h1>
  <div class="sub">
    Drop PDFs/TXT here. Files index automatically. Then search (auto mode).
  </div>

  <div id="zone" class="zone">
    <input id="pick" type="file" multiple style="display:none"/>
    <button class="btn" id="choose">Choose files</button>
    <div class="muted">or drag files into this area…</div>
    <div id="state" class="muted" style="margin-left:auto">docs: 0, vocab: 0</div>
  </div>
  <div id="files" class="list"></div>

  <div class="controls">
    <input id="q" type="text"
      placeholder="Search (e.g., pue or &quot;power usage effectiveness&quot;)" />
    <button class="btn primary" id="go">Search</button>
  </div>

  <div id="results"></div>
</div>

<div id="imageModal" class="modal">
  <span class="modal-close">&times;</span>
  <img class="modal-content" id="modalImg">
</div>

<script>
const zone = document.getElementById('zone');
const pick = document.getElementById('pick');
const choose = document.getElementById('choose');
const filesDiv = document.getElementById('files');
const stateEl = document.getElementById('state');
const q = document.getElementById('q');
const go = document.getElementById('go');
const results = document.getElementById('results');
const modal = document.getElementById('imageModal');
const modalImg = document.getElementById('modalImg');
const modalClose = document.querySelector('.modal-close');

let indexedFiles = new Map();

choose.onclick = () => pick.click();

['dragenter','dragover'].forEach(ev => zone.addEventListener(ev, e => {
  e.preventDefault(); e.stopPropagation(); zone.classList.add('drag');
}));
['dragleave','drop'].forEach(ev => zone.addEventListener(ev, e => {
  e.preventDefault(); e.stopPropagation(); zone.classList.remove('drag');
}));
zone.addEventListener('drop', e => {
  const files = e.dataTransfer.files;
  if (files && files.length) handleFiles(files);
});
pick.addEventListener('change', () => {
  if (pick.files && pick.files.length) handleFiles(pick.files);
});

async function refreshState() {
  const r = await fetch('/', {cache:'no-store'});
  const j = await r.json();
  stateEl.textContent = `docs: ${j.docs}, vocab: ${j.vocab}`;
}

function createFileRow(fileName, fileId) {
  const row = document.createElement('div');
  row.className = 'file';
  row.dataset.fileId = fileId;
  row.innerHTML = `
    <div class="file-name">${fileName}</div>
    <progress value="0" max="100"></progress>
    <span class="muted">waiting</span>
    <button class="btn danger" style="display:none;" title="Remove file">×</button>
  `;
  
  const deleteBtn = row.querySelector('.btn.danger');
  deleteBtn.onclick = () => removeFile(fileId, row);
  
  return row;
}

async function removeFile(fileId, row) {
  if (!confirm('Remove this file from index?')) return;
  
  row.style.opacity = '0.5';
  indexedFiles.delete(fileId);
  
  if (indexedFiles.size > 0) {
    const filesToReindex = Array.from(indexedFiles.values());
    await reindexFiles(filesToReindex);
  } else {
    await clearIndex();
  }
  
  row.remove();
  await refreshState();
}

async function clearIndex() {
  const response = await fetch('/index', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({docs: []})
  });
  return response.ok;
}

async function reindexFiles(files) {
  const fd = new FormData();
  for (const file of files) {
    fd.append('files', file);
  }
  const response = await fetch('/index-files', {
    method: 'POST',
    body: fd
  });
  return response.ok;
}

function handleFiles(fileList) {
  const files = Array.from(fileList);
  files.forEach(file => {
    const fileId = `${file.name}_${Date.now()}_${Math.random()}`;
    const row = createFileRow(file.name, fileId);
    filesDiv.prepend(row);
    startIndexSingle(file, row, fileId);
  });
}

async function startIndexSingle(file, row, fileId) {
  try {
    indexedFiles.set(fileId, file);
    const allFiles = Array.from(indexedFiles.values());
    await xhrUpload(allFiles, row);
    
    const deleteBtn = row.querySelector('.btn.danger');
    deleteBtn.style.display = 'block';
    
    await refreshState();
  } catch (e) {
    row.querySelector('span').textContent = 'error';
    indexedFiles.delete(fileId);
    console.error(e);
  }
}

function xhrUpload(files, row) {
  return new Promise((resolve, reject) => {
    const fd = new FormData();
    for (const f of files) fd.append('files', f);
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/index-files');
    xhr.upload.onprogress = e => {
      if (e.lengthComputable) {
        const p = Math.round((e.loaded / e.total) * 100);
        row.querySelector('progress').value = p;
        row.querySelector('span').textContent = `upload ${p}%`;
      }
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        row.querySelector('progress').value = 100;
        row.querySelector('span').textContent = 'indexed';
        resolve();
      } else reject(new Error('upload failed'));
    };
    xhr.onerror = () => reject(new Error('network error'));
    xhr.send(fd);
  });
}

let allSearchResults = new Map();

async function doSearch(showAll = false) {
  if (!showAll) {
    results.innerHTML = '<div class="muted">Searching…</div>';
    const r = await fetch('/search?q=' + encodeURIComponent(q.value));
    const j = await r.json();
    allSearchResults = new Map(j.results.map(r => [r.doc_id, r]));
  }
  
  const out = [];
  for (const [docId, rdoc] of allSearchResults) {
    const sc = (rdoc.score !== undefined)
      ? `<span class="badge">score ${rdoc.score.toFixed(3)}</span>` : '';
    
    const totalHits = rdoc.total_hits || (rdoc.hits ? rdoc.hits.length : 0);
    const displayedHits = rdoc.hits ? rdoc.hits.length : 0;
    const hasMore = rdoc.has_more === true;
    
    const ht = totalHits
      ? `<span class="badge">${displayedHits} of ${totalHits} hit${totalHits>1?'s':''}</span>`
      : '';
    
    const showAllLink = hasMore && !showAll
      ? `<span class="show-all" onclick="showAllHits('${docId}')">Show all ${totalHits} hits</span>`
      : '';
    
    const hitHtml = (rdoc.hits || []).map((h, idx) => {
      const hitNum = `<div class="meta">Hit ${idx + 1}${h.page ? ` - Page ${h.page}` : ''}${h.line ? `, Line ${h.line}` : ''}</div>`;
      if (h.kind === 'table') {
        const snap = h.snapshot_url
          ? `<div class="right"><img src="${h.snapshot_url}" onclick="openModal(this.src)" alt="Table snapshot"/></div>` 
          : '';
        return `<div class="hit">${hitNum}<div class="body"><div class="left">${h.table_html}</div>${snap}</div></div>`;
      }
      return `<div class="hit">${hitNum}<div class="body"><div class="left">${h.snippet_html||''}</div></div></div>`;
    }).join('');
    
    out.push(
      `<div class="card" id="card-${docId}">
        <div class="doc">${rdoc.doc_id} ${sc} ${ht} ${showAllLink}</div>
        ${hitHtml}
      </div>`
    );
  }
  results.innerHTML = out.join('') || '<div class="muted">No results.</div>';
}

async function showAllHits(docId) {
  const r = await fetch('/search?q=' + encodeURIComponent(q.value) + '&all_hits=true&doc_id=' + docId);
  const j = await r.json();
  
  if (j.results && j.results.length > 0) {
    const rdoc = j.results.find(r => r.doc_id === docId);
    if (rdoc) {
      allSearchResults.set(docId, rdoc);
      updateSingleCard(docId, rdoc);
    }
  }
}

function updateSingleCard(docId, rdoc) {
  const card = document.getElementById(`card-${docId}`);
  if (!card) return;
  
  const sc = (rdoc.score !== undefined)
    ? `<span class="badge">score ${rdoc.score.toFixed(3)}</span>` : '';
  
  const totalHits = rdoc.hits ? rdoc.hits.length : 0;
  const ht = totalHits
    ? `<span class="badge">${totalHits} hit${totalHits>1?'s':''}</span>`
    : '';
  
  const hitHtml = (rdoc.hits || []).map((h, idx) => {
    const hitNum = `<div class="meta">Hit ${idx + 1}${h.page ? ` - Page ${h.page}` : ''}${h.line ? `, Line ${h.line}` : ''}</div>`;
    if (h.kind === 'table') {
      const snap = h.snapshot_url
        ? `<div class="right"><img src="${h.snapshot_url}" onclick="openModal(this.src)" alt="Table snapshot"/></div>` 
        : '';
      return `<div class="hit">${hitNum}<div class="body"><div class="left">${h.table_html}</div>${snap}</div></div>`;
    }
    return `<div class="hit">${hitNum}<div class="body"><div class="left">${h.snippet_html||''}</div></div></div>`;
  }).join('');
  
  card.innerHTML = `
    <div class="doc">${rdoc.doc_id} ${sc} ${ht}</div>
    ${hitHtml}
  `;
}

function openModal(src) {
  modal.classList.add('active');
  modalImg.src = src;
}

modalClose.onclick = () => {
  modal.classList.remove('active');
}

modal.onclick = (e) => {
  if (e.target === modal || e.target === modalImg) {
    modal.classList.remove('active');
  }
}

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && modal.classList.contains('active')) {
    modal.classList.remove('active');
  }
});

go.onclick = doSearch;
q.addEventListener('keydown', (e) => { if (e.key === 'Enter') doSearch(); });

refreshState();
</script>
</body>
</html>
        """
    )


# Indexing


@app.post("/index", response_model=IndexOut)
def index_docs(
    payload: Annotated[IndexIn, Body(...)],
    _auth: None = Depends(_check_auth),  # noqa: B008
) -> IndexOut:
    """Index from raw JSON (replaces current index)."""
    global _INDEX, _DOC_PATHS
    with _INDEX_LOCK:
        w = IndexWriter(_ANALYZER, index_stopwords=True)
        for d in payload.docs:
            w.add(d.id, d.text, page_map=d.page_map or None)
        _INDEX = w.commit()
        _DOC_PATHS = {}
        return IndexOut(
            ok=True,
            indexed=len(_INDEX.doc_ids),
            vocab=_INDEX.vocabulary_size(),
        )


@app.post("/index-files", response_model=IndexOut)
async def index_files(
    files: list[UploadFile] = File(...),  # noqa: B008
    _auth: None = Depends(_check_auth),  # noqa: B008
) -> IndexOut:
    """
    Upload and index multiple files.
    - PDFs: extract text & page_map; remember file paths for table/snapshot.
    - TXTs: read UTF-8 text.
    Index is replaced on each call.
    """
    if not files:
        raise HTTPException(400, "No files")

    saved: list[tuple[str, Path, str, list[tuple[int, int]]]] = []
    for f in files:
        data = await f.read()
        if not data:
            continue
        name = _safe_doc_id(f.filename or "doc")
        path_out = UPLOAD_DIR / name
        path_out.write_bytes(data)

        if name.lower().endswith(".pdf"):
            text, page_map = _extract_pdf_text_and_page_map(path_out)
        else:
            try:
                text = data.decode("utf-8", errors="ignore")
            except Exception:
                text = ""
            page_map = []
        saved.append((name, path_out, text, page_map))

    if not saved:
        raise HTTPException(400, "Empty files")

    global _INDEX, _DOC_PATHS
    with _INDEX_LOCK:
        w = IndexWriter(_ANALYZER, index_stopwords=True)
        for name, _p, text, page_map in saved:
            doc_id = os.path.splitext(name)[0]  # nicer id: drop extension
            w.add(doc_id, text, page_map=page_map)
        _INDEX = w.commit()
        
        # Store paths for PDFs
        _DOC_PATHS = {
            os.path.splitext(name)[0]: str(p) 
            for name, p, _t, _pm in saved
        }

    return IndexOut(
        ok=True,
        indexed=len(_INDEX.doc_ids),
        vocab=_INDEX.vocabulary_size(),
    )


# Searching


@app.get("/search", response_model=SearchOut)
def search(
    q: Annotated[str, Query(min_length=1)],
    all_hits: bool = Query(
        False, 
        description="Return all hits instead of just first"
    ),
    doc_id: str | None = Query(
        None, 
        description="Filter results to specific document"
    ),
    _auth: None = Depends(_check_auth),  # noqa: B008
) -> SearchOut:
    if _INDEX is None:
        raise HTTPException(400, "Index empty. Upload on /ui first.")

    mode = _auto_mode(q)
    s = Searcher(_INDEX, _ANALYZER)
    results: list[dict[str, Any]] = []

    if mode == "phrase":
        pm = PhraseMatcher(_INDEX, _ANALYZER)
        hits = pm.match(q, keep_stopwords=True)
        terms = _ANALYZER.tokenize(q, keep_stopwords=True)
        
        for doc_id_iter in sorted(hits.keys()):
            # Filter by doc_id if specified
            if doc_id and doc_id_iter != doc_id:
                continue
                
            starts = hits[doc_id_iter]
            
            # Determine how many hits to show
            if all_hits:
                hits_to_show = starts[:MAX_HITS_ALL]
            else:
                hits_to_show = starts[:MAX_HITS_DEFAULT]
            
            out_hits: list[dict[str, Any]] = []
            for st in hits_to_show:
                page = _page_of_pos(doc_id_iter, st) or 1
                table = _table_snippet(doc_id_iter, page - 1, terms)
                if table:
                    table["snapshot_url"] = (
                        f"/page-snapshot?doc_id={doc_id_iter}"
                        f"&page={table['page']}"
                        f"&x0={table['bbox'][0]}&y0={table['bbox'][1]}"
                        f"&x1={table['bbox'][2]}&y1={table['bbox'][3]}"
                    )
                    out_hits.append(table)
                else:
                    snippet_data = _paragraph_snippet(
                        doc_id_iter, st, terms
                    )
                    out_hits.append(snippet_data)
            
            result_data = {
                "doc_id": doc_id_iter, 
                "hits": out_hits,
                "total_hits": len(starts)
            }
            
            # Mark if there are more hits available
            if not all_hits and len(starts) > MAX_HITS_DEFAULT:
                result_data["has_more"] = True
                
            results.append(result_data)

    elif mode == "boolean":
        docs = s.search_boolean(q)
        terms = _ANALYZER.tokenize(q, keep_stopwords=False)
        
        for doc_id_iter in docs:
            # Filter by doc_id if specified
            if doc_id and doc_id_iter != doc_id:
                continue
                
            # Get all positions for boolean mode
            all_positions = _all_term_positions(doc_id_iter, terms)
            
            # Limit positions shown
            if all_hits:
                positions_to_show = all_positions[:MAX_HITS_ALL]
            else:
                positions_to_show = all_positions[:MAX_HITS_DEFAULT]
            
            out_hits: list[dict[str, Any]] = []
            for pos in positions_to_show:
                page = _page_of_pos(doc_id_iter, pos) or 1
                table = _table_snippet(doc_id_iter, page - 1, terms)
                hit: dict[str, Any] = (
                    table or _paragraph_snippet(doc_id_iter, pos, terms)
                )
                if table:
                    hit["snapshot_url"] = (
                        f"/page-snapshot?doc_id={doc_id_iter}"
                        f"&page={table['page']}"
                        f"&x0={table['bbox'][0]}&y0={table['bbox'][1]}"
                        f"&x1={table['bbox'][2]}&y1={table['bbox'][3]}"
                    )
                out_hits.append(hit)
            
            result_data = {
                "doc_id": doc_id_iter,
                "hits": out_hits,
                "total_hits": len(all_positions)
            }
            
            if not all_hits and len(all_positions) > MAX_HITS_DEFAULT:
                result_data["has_more"] = True
                
            results.append(result_data)

    else:  # ranked
        top = s.search_ranked(q, k=20)
        terms = _ANALYZER.tokenize(q, keep_stopwords=False)
        
        for doc_id_iter, score in top:
            # Filter by doc_id if specified
            if doc_id and doc_id_iter != doc_id:
                continue
                
            # Get all positions for ranked mode
            all_positions = _all_term_positions(doc_id_iter, terms)
            
            # Limit positions shown
            if all_hits:
                positions_to_show = all_positions[:MAX_HITS_ALL]
            else:
                positions_to_show = all_positions[:MAX_HITS_DEFAULT]
            
            out_hits: list[dict[str, Any]] = []
            for pos in positions_to_show:
                page = _page_of_pos(doc_id_iter, pos) or 1
                table = _table_snippet(doc_id_iter, page - 1, terms)
                hit_data: dict[str, Any] = (
                    table or _paragraph_snippet(doc_id_iter, pos, terms)
                )
                if table:
                    hit_data["snapshot_url"] = (
                        f"/page-snapshot?doc_id={doc_id_iter}"
                        f"&page={table['page']}"
                        f"&x0={table['bbox'][0]}&y0={table['bbox'][1]}"
                        f"&x1={table['bbox'][2]}&y1={table['bbox'][3]}"
                    )
                out_hits.append(hit_data)
            
            result_data = {
                "doc_id": doc_id_iter, 
                "score": float(score), 
                "hits": out_hits,
                "total_hits": len(all_positions)
            }
            
            if not all_hits and len(all_positions) > MAX_HITS_DEFAULT:
                result_data["has_more"] = True
                
            results.append(result_data)

    return SearchOut(results=results)


# Page snapshot with highlight box (for tables)


@app.get("/page-snapshot")
def page_snapshot(
    doc_id: str,
    page: int = Query(..., ge=1),
    x0: float = Query(...),
    y0: float = Query(...),
    x1: float = Query(...),
    y1: float = Query(...),
) -> StreamingResponse:
    """
    Render the PDF page and draw a rectangle for the given bbox.
    Coordinates are PDF points (same as pdfplumber's).
    """
    path = _DOC_PATHS.get(doc_id)
    if not path or not os.path.exists(path):
        raise HTTPException(404, "Original file not found.")

    try:
        doc = fitz.open(path)
        pno = page - 1
        if pno < 0 or pno >= doc.page_count:
            raise HTTPException(400, "Invalid page.")
        pg = doc[pno]

        # 2x zoom for readability
        pix = pg.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes(
            "RGB", 
            (pix.width, pix.height), 
            pix.samples
        )

        # map PDF coords -> pixels
        rx, ry = pg.rect.width, pg.rect.height
        sx = pix.width / rx
        sy = pix.height / ry
        box = (
            int(x0 * sx), 
            int(y0 * sy), 
            int(x1 * sx), 
            int(y1 * sy)
        )

        draw = ImageDraw.Draw(img)
        draw.rectangle(box, outline=(255, 204, 0), width=4)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover
        raise HTTPException(500, f"render failed: {e}") from e