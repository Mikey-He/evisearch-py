# ruff: noqa: E501
from __future__ import annotations

import html
import io
import os
from pathlib import Path
import re
import threading
from typing import Annotated, Any
import urllib.parse as ul

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
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

app = FastAPI(title="EviSearch-Py", version="1.4.1-FIXED") # Version update

DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads"
DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_ANALYZER = Analyzer()
_INDEX_LOCK = threading.Lock()
_INDEX: InvertedIndex | None = None

# doc_id -> original file path
_DOC_PATHS: dict[str, str] = {}
# doc_id -> (text, page_map) for rebuilding index
_DOC_DATA: dict[str, tuple[str, list[tuple[int, int]]]] = {}

# Default context windows
DEFAULT_LINES_WINDOW = 1
DEFAULT_COL_WINDOW = 1

# Maximum hits settings
MAX_HITS_DEFAULT = 3
MAX_HITS_ALL = 50

# Optional Basic Auth
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


class SearchIn(BaseModel):
    q: str
    mode: str | None = None
    doc_id: str | None = None
    max_hits_per_doc: int = MAX_HITS_DEFAULT
    context_lines: int = DEFAULT_LINES_WINDOW


class HitData(BaseModel):
    kind: str
    snippet_html: str | None = None
    table_html: str | None = None
    line: int | None = None
    page: int | None = None
    snapshot_url: str | None = None
    bbox: list[float] | None = None


class ResultDoc(BaseModel):
    doc_id: str
    score: float | None = None
    hits: list[HitData]
    total_hits: int
    has_more: bool = False


class SearchOut(BaseModel):
    results: list[ResultDoc]


class DeleteOut(BaseModel):
    ok: bool
    message: str


class FileListOut(BaseModel):
    files: list[str]


# Helpers

def _terms_qs(terms: list[str]) -> str:
    if not terms:
        return ""
    qs = ",".join(ul.quote_plus(t) for t in terms if t)
    return f"&terms={qs}"

def _safe_doc_id(name: str) -> str:
    """Create safe document ID from filename"""
    base = os.path.basename(name or "doc")
    # Keep original filename with extension as doc_id
    return base

def _extract_pdf_text_and_page_map(
    path: Path,
) -> tuple[str, list[tuple[int, int]]]:
    """Extract page texts and page_map using PyMuPDF (fitz)"""
    texts: list[str] = []
    page_map: list[tuple[int, int]] = []
    pos = 0

    with fitz.open(str(path)) as doc: # fitz
        for i, page in enumerate(doc, start=1):
            # Extract text with fallback to empty string
            txt = page.get_text() or "" 
            texts.append(txt)
            toks = list(_ANALYZER.iter_tokens(txt, keep_stopwords=True))
            page_map.append((pos, i))
            pos += len(toks)

    joined = "\n\n".join(texts)
    return joined, page_map

def _page_of_pos(doc_id: str, pos: int) -> int | None:
    """Get page number for token position"""
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


def _all_term_positions(
    doc_id: str, 
    terms: list[str]
) -> list[int]:
    """Get all positions where any term appears"""
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
    """Escape HTML and wrap matches with <mark>"""
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
    terms: list[str],
    context_lines: int = DEFAULT_LINES_WINDOW
) -> HitData:
    """Build line-window snippet"""
    if _INDEX is None or start_pos is None:
        return HitData(kind="text", snippet_html="", line=0, page=None)

    lines = _INDEX.doc_lines.get(doc_id, [])
    line_of_pos = _INDEX.line_of_pos.get(doc_id, [])
    if not lines or not line_of_pos or start_pos >= len(line_of_pos):
        return HitData(kind="text", snippet_html="", line=0, page=None)

    hit = line_of_pos[start_pos]
    s = max(0, hit - context_lines)
    e = min(len(lines) - 1, hit + context_lines)
    block = "\n".join(lines[s : e + 1])

    page = _page_of_pos(doc_id, start_pos)
    html_block = (
        "<pre class='snippet'>" + 
        _highlight_terms(block, terms) + 
        "</pre>"
    )
    return HitData(
        kind="text", 
        snippet_html=html_block, 
        line=hit + 1, 
        page=page
    )


def _table_snippet(
    doc_id: str,
    page_no: int,
    terms: list[str],
    cwin: int = DEFAULT_COL_WINDOW,
    rwin: int = 1,
) -> HitData | None:
    """Extract table snippet if terms found in table"""
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

            # Find hit cells
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

            # Extract sub-table
            sub_rows: list[list[str]] = []
            for rr in range(r0, r1 + 1):
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
                cell_html = "".join(f"<{tag}>{c}</{tag}>" for c in cells)
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
            return HitData(
                kind="table",
                table_html=table_html,
                bbox=[float(x0), float(y0), float(x1), float(y1)],
                page=page_no + 1,
            )
    except Exception:
        return None


def _auto_mode(q: str) -> str:
    """Detect search mode from query"""
    q2 = q.strip()
    if not q2:
        return "ranked"
    if '"' in q2 or "'" in q2:
        return "phrase"
    if re.search(r"\b(and|or|not)\b|\(|\)", q2, re.IGNORECASE):
        return "boolean"
    return "ranked"


def _perform_search(
    query: str,
    mode: str | None = None,
    doc_id_filter: str | None = None,
    max_hits_per_doc: int = MAX_HITS_DEFAULT,
    context_lines: int = DEFAULT_LINES_WINDOW
) -> list[ResultDoc]:
    """Core search logic used by both GET and POST endpoints"""
    if _INDEX is None:
        raise HTTPException(400, "Index empty. Upload files first.")

    # Auto-detect mode if not specified
    search_mode = mode or _auto_mode(query)
    s = Searcher(_INDEX, _ANALYZER)
    results: list[ResultDoc] = []
    
    # Track shown pages per document to avoid duplicates
    shown_pages: dict[str, set[int]] = {}

    if search_mode == "phrase":
        pm = PhraseMatcher(_INDEX, _ANALYZER)
        hits = pm.match(query, keep_stopwords=True)
        terms = _ANALYZER.tokenize(query, keep_stopwords=True)
        
        for doc_id in sorted(hits.keys()):
            if doc_id_filter and doc_id != doc_id_filter:
                continue
                
            starts = hits[doc_id]
            hits_to_show = starts[:max_hits_per_doc]
            shown_pages[doc_id] = set()
            
            out_hits: list[HitData] = []
            for st in hits_to_show:
                page = _page_of_pos(doc_id, st) or 1
                
                # Try table snippet first
                table = _table_snippet(doc_id, page - 1, terms)
                if table:
                    # Only add snapshot if not already shown for this page
                    if page not in shown_pages.get(doc_id, set()):
                        table.snapshot_url = (
                            f"/page-snapshot?doc_id={doc_id}"
                            f"&page={table.page}"
                            f"&full=1"
                            f"&boxes={table.bbox[0]},{table.bbox[1]},{table.bbox[2]},{table.bbox[3]}"
                            f"{_terms_qs(terms)}&scale=3"
                        )
                        shown_pages.setdefault(doc_id, set()).add(page)
                    out_hits.append(table)
                else:
                    # Fall back to text snippet
                    snippet = _paragraph_snippet(doc_id, st, terms, context_lines)
                    # Generate page snapshot if not already shown
                    if page not in shown_pages.get(doc_id, set()):
                        snippet.snapshot_url = (
                            f"/page-snapshot?doc_id={doc_id}"
                            f"&page={page}"
                            f"&full=1"
                            f"{_terms_qs(terms)}&scale=3"
                        )
                        shown_pages.setdefault(doc_id, set()).add(page)
                    out_hits.append(snippet)
            
            results.append(ResultDoc(
                doc_id=doc_id,
                hits=out_hits,
                total_hits=len(starts),
                has_more=len(starts) > max_hits_per_doc
            ))

    elif search_mode == "boolean":
        docs = s.search_boolean(query)
        terms = _ANALYZER.tokenize(query, keep_stopwords=False)
        
        for doc_id in docs:
            if doc_id_filter and doc_id != doc_id_filter:
                continue
                
            all_positions = _all_term_positions(doc_id, terms)
            positions_to_show = all_positions[:max_hits_per_doc]
            shown_pages[doc_id] = set()
            
            out_hits: list[HitData] = []
            for pos in positions_to_show:
                page = _page_of_pos(doc_id, pos) or 1
                
                table = _table_snippet(doc_id, page - 1, terms)
                if table:
                    if page not in shown_pages.get(doc_id, set()):
                        table.snapshot_url = (
                            f"/page-snapshot?doc_id={doc_id}"
                            f"&page={table.page}"
                            f"&full=1"
                            f"&boxes={table.bbox[0]},{table.bbox[1]},{table.bbox[2]},{table.bbox[3]}"
                            f"{_terms_qs(terms)}&scale=3"
                        )
                        shown_pages.setdefault(doc_id, set()).add(page)
                    out_hits.append(table)
                else:
                    snippet = _paragraph_snippet(doc_id, pos, terms, context_lines)
                    if page not in shown_pages.get(doc_id, set()):
                        snippet.snapshot_url = (
                            f"/page-snapshot?doc_id={doc_id}"
                            f"&page={page}"
                            f"&full=1"
                            f"{_terms_qs(terms)}&scale=3"
                          )
                        shown_pages.setdefault(doc_id, set()).add(page)
                    out_hits.append(snippet)
            
            results.append(ResultDoc(
                doc_id=doc_id,
                hits=out_hits,
                total_hits=len(all_positions),
                has_more=len(all_positions) > max_hits_per_doc
            ))

    else:  # ranked
        top = s.search_ranked(query, k=20)
        terms = _ANALYZER.tokenize(query, keep_stopwords=False)
        
        for doc_id, score in top:
            if doc_id_filter and doc_id != doc_id_filter:
                continue
                
            all_positions = _all_term_positions(doc_id, terms)
            positions_to_show = all_positions[:max_hits_per_doc]
            shown_pages[doc_id] = set()
            
            out_hits: list[HitData] = []
            for pos in positions_to_show:
                page = _page_of_pos(doc_id, pos) or 1
                
                table = _table_snippet(doc_id, page - 1, terms)
                if table:
                    if page not in shown_pages.get(doc_id, set()):
                        table.snapshot_url = (
                            f"/page-snapshot?doc_id={doc_id}"
                            f"&page={table.page}"
                            f"&full=1"
                            f"&boxes={table.bbox[0]},{table.bbox[1]},{table.bbox[2]},{table.bbox[3]}"
                            f"{_terms_qs(terms)}&scale=3"
                        )
                        shown_pages.setdefault(doc_id, set()).add(page)
                    out_hits.append(table)
                else:
                    snippet = _paragraph_snippet(doc_id, pos, terms, context_lines)
                    if page not in shown_pages.get(doc_id, set()):
                        snippet.snapshot_url = (
                            f"/page-snapshot?doc_id={doc_id}"
                            f"&page={page}"
                            f"&full=1"
                            f"{_terms_qs(terms)}&scale=3"
                        )
                        shown_pages.setdefault(doc_id, set()).add(page)
                    out_hits.append(snippet)
            
            results.append(ResultDoc(
                doc_id=doc_id,
                score=float(score),
                hits=out_hits,
                total_hits=len(all_positions),
                has_more=len(all_positions) > max_hits_per_doc
            ))

    return results


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
    # Enhanced UI with larger drop zone and improved features
    return HTMLResponse("""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>EviSearch-Py</title>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    :root {
      /* --- Dark Mode Palette (Default) --- */
      --bg: #121212;
      --bg-secondary: #1E1E1E;
      --fg: #E0E0E0;
      --fg-muted: #888888;
      --border: #333333;
      --accent: #4CAF50;
      --accent-hover: #66BB6A;
      --danger: #F44336;
      --danger-hover: #E57373;
      --mark-bg: #FFF59D;
      --mark-fg: #333;
      --shadow: rgba(0, 0, 0, 0.2);
    }

    body.light-mode {
      /* --- Light Mode Palette --- */
      --bg: #F5F5F5;
      --bg-secondary: #FFFFFF;
      --fg: #212121;
      --fg-muted: #757575;
      --border: #E0E0E0;
      --accent: #2E7D32;
      --accent-hover: #388E3C;
      --danger: #D32F2F;
      --danger-hover: #E53935;
      --mark-bg: #FFD54F;
      --mark-fg: #212121;
      --shadow: rgba(0, 0, 0, 0.1);
    }

    html, body {
      background: var(--bg);
      color: var(--fg);
      font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      font-size: 16px;
      line-height: 1.6;
      transition: background-color 0.3s, color 0.3s;
    }

    .wrap { max-width: 960px; margin: 32px auto; padding: 0 16px; position: relative; }
    h1 { font-size: 32px; margin: 0 0 8px; font-weight: 700; }
    .sub { color: var(--fg-muted); margin-bottom: 24px; }


        /* --- Theme Toggle --- */
    .theme-toggle {
      position: absolute;
      top: 8px;
      right: 16px;
      background: var(--bg-secondary);
      border: 1px solid var(--border);
      color: var(--fg);
      width: 40px;
      height: 40px;
      border-radius: 50%;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 20px;
      transition: all 0.3s ease;
    }
    .theme-toggle:hover { background: var(--border); }
    .theme-toggle .icon-sun { display: none; }
    .theme-toggle .icon-moon { display: block; }
    body.light-mode .theme-toggle .icon-sun { display: block; }
    body.light-mode .theme-toggle .icon-moon { display: none; }                        
    
    .zone {
      border: 2px dashed var(--border);
      border-radius: 12px;
      padding: 48px 24px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      align-items: center;
      justify-content: center;
      background: var(--bg-secondary);
      position: relative;
      transition: all 0.3s ease;
    }
    .zone.drag { border-color: var(--accent); background: rgba(76, 175, 80, 0.1); }
    .zone-content { display: flex; gap: 12px; align-items: center; }
    
    .btn {
      background: var(--bg-secondary);
      color: var(--fg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 16px;
      cursor: pointer;
      font-size: 14px;
      font-weight: 500;
      transition: all 0.2s ease;
    }
    .btn:hover { border-color: var(--accent); color: var(--accent); }
    .btn.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
    .btn.primary:hover { background: var(--accent-hover); border-color: var(--accent-hover); color: #fff; }
    .btn.danger { border: 1px solid var(--danger); color: var(--danger); }
    .btn.danger:hover { background: var(--danger); color: #fff; }
    
    .btn.icon-btn { /* Style for single-file delete button */
        background: transparent;
        color: var(--fg-muted);
        border: none;
        padding: 4px;
        font-size: 20px;
        line-height: 1;
    }
    .btn.icon-btn:hover { color: var(--danger); }
    
    #deleteAllContainer { text-align: right; margin: 12px 0; }
    #deleteAllBtn { font-size: 12px; padding: 6px 12px; } /* Smaller font for delete all */

    .list { margin-top: 10px; }
    .file {
      display: flex;
      align-items: center;
      gap: 12px;
      margin: 8px 0;
      padding: 12px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--bg-secondary);
      box-shadow: 0 2px 4px var(--shadow);
    }
    .file-name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; }

    .controls { display: flex; gap: 10px; margin: 24px 0; }
    input[type=text] {
      flex: 1;
      min-width: 0;
      background: var(--bg-secondary);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 14px;
      color: var(--fg);
      font-size: 16px;
      transition: all 0.3s ease;
    }
    input[type=text]:focus { border-color: var(--accent); outline: none; box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.2); }

    .card {
      margin: 16px 0;
      padding: 16px;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: var(--bg-secondary);
      box-shadow: 0 4px 8px var(--shadow);
    }
    .doc { font-weight: 500; margin-bottom: 10px; display: flex; align-items: center; gap: 10px; }
    .badge {
      font-size: 12px;
      color: var(--accent);
      background: rgba(76, 175, 80, 0.15);
      border: 1px solid rgba(76, 175, 80, 0.3);
      padding: 2px 8px;
      border-radius: 999px;
    }
    mark { background: var(--mark-bg); color: var(--mark-fg); border-radius: 3px; padding: 1px 3px;}
    .hit { margin: 12px 0; padding: 12px; background: var(--bg); border-radius: 8px; border: 1px solid var(--border); }
    .hit .meta { color: var(--fg-muted); font-size: 14px; margin-bottom: 8px; }
    .hit img { max-width: 100%; border: 1px solid var(--border); border-radius: 8px; cursor: zoom-in; }

    /* Lightbox styles */
    .lightbox{
      display:none; position:fixed; z-index:1000;
      left:0; top:0; width:100%; height:100%;
      background:rgba(0,0,0,0.95); 
      cursor:grab;
    }
    .lightbox.active{display:flex; align-items:center; justify-content:center;}
    .lightbox.dragging{cursor:grabbing;}
    .lightbox-img{
      position:absolute;
      max-width:90%; max-height:90%;
      user-select:none;
      transition:none;
    }
    .lightbox-close{
      position:absolute; top:20px; right:35px;
      color:#f1f1f1; font-size:40px; font-weight:bold;
      cursor:pointer; z-index:1001;
    }
    .lightbox-close:hover{color:#fff;}
    .lightbox-hint{
      position:absolute; bottom:20px; left:50%;
      transform:translateX(-50%);
      color:#fff; background:rgba(0,0,0,0.7);
      padding:8px 16px; border-radius:8px;
      font-size:14px;
    }
    
    .show-all{
      color:var(--accent); cursor:pointer; 
      margin-left:10px; font-size:14px;
      text-decoration:underline;
    }
    .show-all:hover{color:#3fb254;}
    
    #state{
      position:absolute; top:12px; right:12px;
      font-size:14px;
    }
  </style>
</head>
<body>
<div class="wrap">
  <button class="theme-toggle" id="themeToggle" title="Toggle theme">
      <span class="icon-moon">🌙</span>
      <span class="icon-sun">☀️</span>
  </button>
  <h1>EviSearch-Py</h1>
  <div class="sub">Drop PDFs/TXT here. Files index automatically. Then search.</div>

  <div id="zone" class="zone">
    <div class="zone-content">
      <input id="pick" type="file" multiple style="display:none"/>
      <button class="btn" id="choose">Choose files</button>
      <div class="muted">or drag files into this area…</div>
    </div>
    <div id="state" class="muted">docs: 0, vocab: 0</div>
  </div>
  <div id="files" class="list"></div>

  <div id="deleteAllContainer" style="text-align: right; margin: 10px 0; display: none;">
      <button class="btn danger" id="deleteAllBtn">Delete All Files</button>
  </div>
  <div id="files" class="list"></div>
                        
  <div class="controls">
    <input id="q" type="text"
      placeholder="Search (e.g., pue or &quot;power usage effectiveness&quot;)" />
    <button class="btn primary" id="go">Search</button>
  </div>

  <div id="results"></div>
</div>

<div id="lightbox" class="lightbox">
  <span class="lightbox-close">&times;</span>
  <img class="lightbox-img" id="lightboxImg">
  <div class="lightbox-hint">Scroll to zoom • Drag to pan • ESC to close</div>
</div>

<script>
// --- START: All JavaScript code is now encapsulated in one block ---

// Theme Toggle Logic
const themeToggle = document.getElementById('themeToggle');
const body = document.body;

themeToggle.onclick = () => {
    body.classList.toggle('light-mode');
    if (body.classList.contains('light-mode')) {
        localStorage.setItem('theme', 'light');
    } else {
        localStorage.setItem('theme', 'dark');
    }
};

function applyInitialTheme() {
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    if (savedTheme === 'light') {
        body.classList.add('light-mode');
    } else if (savedTheme === 'dark') {
        // Already default, do nothing
    } else if (prefersDark) {
        // Default to system preference if no choice saved
    } else {
        body.classList.add('light-mode');
    }
}
                        
// Element Lookups
const MAX_HITS_ALL = 50;
const zone = document.getElementById('zone');
const pick = document.getElementById('pick');
const choose = document.getElementById('choose');
const filesDiv = document.getElementById('files');
const stateEl = document.getElementById('state');
const q = document.getElementById('q');
const go = document.getElementById('go');
const results = document.getElementById('results');
const deleteAllBtn = document.getElementById('deleteAllBtn');
const deleteAllContainer = document.getElementById('deleteAllContainer');
const lightbox = document.getElementById('lightbox');
const lightboxImg = document.getElementById('lightboxImg');
const lightboxClose = document.querySelector('.lightbox-close');

let indexedFiles = new Map();
let activeXHRs = new Map();

// Lightbox state
let scale = 1, translateX = 0, translateY = 0, isDragging = false, startX, startY;

// Helper to manage "Delete All" button visibility
function updateDeleteAllButtonVisibility() {
    deleteAllContainer.style.display = filesDiv.children.length > 0 ? 'block' : 'none';
}

// "Delete All" button event handler
deleteAllBtn.onclick = async () => {
    if (!confirm('Are you sure you want to delete ALL indexed files? This cannot be undone.')) return;
    try {
        const response = await fetch('/files', { method: 'DELETE' });
        if (response.ok) {
            filesDiv.innerHTML = '';
            results.innerHTML = '';
            q.value = '';
            await refreshState();
            updateDeleteAllButtonVisibility();
        } else {
            alert('Failed to delete all files.');
        }
    } catch (e) {
        console.error('Error deleting all files:', e);
        alert('An error occurred while deleting all files.');
    }
};


choose.onclick = () => pick.click();
['dragenter', 'dragover'].forEach(ev => zone.addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); zone.classList.add('drag'); }));
['dragleave', 'drop'].forEach(ev => zone.addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); zone.classList.remove('drag'); }));
zone.addEventListener('drop', e => { if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files); });
pick.addEventListener('change', () => { if (pick.files.length) handleFiles(pick.files); });

async function refreshState() {
    const r = await fetch('/', { cache: 'no-store' });
    const j = await r.json();
    stateEl.textContent = `docs: ${j.docs}, vocab: ${j.vocab}`;
}

async function getFileList() {
    try {
        const r = await fetch('/files');
        const j = await r.json();
        // This function is called on page load after the list is built in the main script flow
        // The visibility will be updated there
        return j.files || [];
    } catch (e) {
        return [];
    }
}

function createFileRow(fileName, fileId) {
    const row = document.createElement('div');
    row.className = 'file';
    row.dataset.fileId = fileId;
    row.innerHTML = `
        <div class="file-name">${fileName}</div>
        <progress value="0" max="100"></progress>
        <span class="muted">waiting</span>
        <button class="btn cancel" style="display:none;">Cancel</button>
        <button class="btn icon-btn danger" style="display:none;" title="Remove file">×</button>
    `;
    const deleteBtn = row.querySelector('.btn.danger');
    const cancelBtn = row.querySelector('.btn.cancel');
    deleteBtn.onclick = () => removeFile(fileName, row);
    cancelBtn.onclick = () => cancelUpload(fileId, row);
    return row;
}


async function removeFile(fileName, row) {
    if (!confirm('Remove this file from index?')) return;
    row.style.opacity = '0.5';
    try {
        const response = await fetch(`/files/${encodeURIComponent(fileName)}`, { method: 'DELETE' });
        if (response.ok) {
            row.remove();
            await refreshState();
            updateDeleteAllButtonVisibility();
        } else {
            row.style.opacity = '1'; alert('Failed to remove file');
        }
    } catch (e) {
        row.style.opacity = '1'; alert('Error removing file');
    }
}

function cancelUpload(fileId, row) {
    const xhr = activeXHRs.get(fileId);
    if (xhr) {
        xhr.abort();
        activeXHRs.delete(fileId);
        row.remove();
        updateDeleteAllButtonVisibility();
    }
}

function handleFiles(fileList) {
    Array.from(fileList).forEach(file => {
        const fileId = `${file.name}_${Date.now()}`;
        const row = createFileRow(file.name, fileId);
        filesDiv.prepend(row);
        startIndexSingle(file, row, fileId);
    });
}

async function startIndexSingle(file, row, fileId) {
    try {
        const cancelBtn = row.querySelector('.btn.cancel');
        cancelBtn.style.display = 'block';
        await xhrUpload(file, row, fileId);
        cancelBtn.style.display = 'none';
        row.querySelector('.btn.danger').style.display = 'block';
        await refreshState();
        updateDeleteAllButtonVisibility();
    } catch (e) {
        if (e.message !== 'Aborted') console.error(e);
        row.querySelector('span').textContent = e.message === 'Aborted' ? 'cancelled' : 'error';
        activeXHRs.delete(fileId);
    }
}

function xhrUpload(file, row, fileId) {
    return new Promise((resolve, reject) => {
        const fd = new FormData();
        fd.append('files', file);
        const xhr = new XMLHttpRequest();
        activeXHRs.set(fileId, xhr);
        xhr.open('POST', '/index-files');
        xhr.upload.onprogress = e => {
            if (e.lengthComputable) {
                const p = Math.round((e.loaded / e.total) * 100);
                row.querySelector('progress').value = p;
                row.querySelector('span').textContent = `uploading ${p}%`;
            }
        };
        xhr.onload = () => {
            activeXHRs.delete(fileId);
            if (xhr.status >= 200 && xhr.status < 300) {
                row.querySelector('progress').value = 100;
                row.querySelector('span').textContent = 'indexed';
                resolve();
            } else { reject(new Error('Upload failed')); }
        };
        xhr.onerror = () => { activeXHRs.delete(fileId); reject(new Error('Network error')); };
        xhr.onabort = () => { activeXHRs.delete(fileId); reject(new Error('Aborted')); };
        xhr.send(fd);
    });
}

// Helper to convert hits data to HTML
function createHitHtml(hits) {
    return hits.map((h, idx) => {
      // Only show images, no text/table snippets in the UI
      if (h.snapshot_url) {
        return `<div class="hit">
          <div class="meta">Hit ${idx + 1}${h.page ? ` - Page ${h.page}` : ''}</div>
          <img src="${h.snapshot_url}" onclick="openLightbox(this.src)" alt="Page snapshot"/>
        </div>`;
      }
      return '';
    }).join('');
}
                        

async function doSearch() {
  results.innerHTML = '<div class="muted">Searching…</div>';
  
  const payload = {
    q: q.value,
    max_hits_per_doc: 5,
    context_lines: 2
  };
  
  let jsonResponse;
  try {
      const r = await fetch('/search', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });
      jsonResponse = await r.json();
  } catch (e) {
      results.innerHTML = '<div class="muted">Search error. Check console for details.</div>';
      console.error("Search API call failed:", e);
      return;
  }
  
  const out = [];
  for (const rdoc of jsonResponse.results) {
    const sc = rdoc.score !== undefined
      ? `<span class="badge">score ${rdoc.score.toFixed(3)}</span>` : '';
    
    const ht = rdoc.total_hits
      ? `<span class="badge">${rdoc.hits.length} of ${rdoc.total_hits} hits</span>`
      : '';
    
    const showAllLink = rdoc.has_more
      // Pass the query value to showAllHits
      ? `<span class="show-all" onclick="showAllHits('${rdoc.doc_id}', '${q.value.replace(/'/g, "\\'")}')">Show all hits</span>`
      : '';
    
    const hitHtml = createHitHtml(rdoc.hits);
    
    out.push(
      `<div class="card" data-doc-id="${rdoc.doc_id}">
        <div class="doc">${rdoc.doc_id} ${sc} ${ht} ${showAllLink}</div>
        <div class="hit-container">
          ${hitHtml}
        </div>
      </div>`
    );
  }
  
  results.innerHTML = out.join('') || '<div class="muted">No results.</div>';
}

// showAllHits 
async function showAllHits(docId, currentQuery) {
  // Find the card element to update
  const card = document.querySelector(`.card[data-doc-id='${docId}']`);
  if (!card) return;
  
  const hitContainer = card.querySelector('.hit-container');
  const showAllLink = card.querySelector('.show-all');
  const hitBadge = card.querySelector('.doc .badge:nth-child(2)'); // Total hits badge

  if (hitContainer) hitContainer.innerHTML = '<div class="muted" style="margin-left:10px;">Loading all hits...</div>';
  if (showAllLink) showAllLink.style.display = 'none';

  const payload = {
    q: currentQuery, // Use the query passed from doSearch
    doc_id: docId, // Crucial: filter to this specific document
    max_hits_per_doc: MAX_HITS_ALL, // Request the maximum number of hits (50)
    context_lines: 2
  };
  
  let jsonResponse;
  try {
    const r = await fetch('/search', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    jsonResponse = await r.json();
  } catch (e) {
    if (hitContainer) hitContainer.innerHTML = '<div class="muted" style="margin-left:10px;">Error loading all hits.</div>';
    console.error(`Failed to load all hits for ${docId}:`, e);
    return;
  }
  
  const rdoc = jsonResponse.results[0];
  if (!rdoc) return;
  
  // 1. Update the hit container with ALL hits
  if (hitContainer) {
      hitContainer.innerHTML = createHitHtml(rdoc.hits);
  }

  // 2. Update the total hits badge
  if (hitBadge) {
      hitBadge.textContent = `${rdoc.total_hits} of ${rdoc.total_hits} hits`;
  }
}

// Lightbox functions
function openLightbox(src) {
  lightbox.classList.add('active');
  lightboxImg.src = src;
  scale = 1;
  translateX = 0;
  translateY = 0;
  updateTransform();
  
  // Hide hint after 3 seconds
  setTimeout(() => {
    const hint = document.querySelector('.lightbox-hint');
    if (hint) hint.style.opacity = '0';
  }, 3000);
}

function closeLightbox() {
  lightbox.classList.remove('active');
  const hint = document.querySelector('.lightbox-hint');
  if (hint) hint.style.opacity = '1';
}

function updateTransform() {
  lightboxImg.style.transform = `translate(${translateX}px, ${translateY}px) scale(${scale})`;
}

// Lightbox event handlers
lightboxClose.onclick = closeLightbox;

lightbox.addEventListener('wheel', e => {
  if (!lightbox.classList.contains('active')) return;
  e.preventDefault();
  
  const delta = e.deltaY > 0 ? 0.9 : 1.1;
  const newScale = scale * delta;
  
  if (newScale >= 0.5 && newScale <= 5) {
    scale = newScale;
    updateTransform();
  }
});

lightbox.addEventListener('mousedown', e => {
  if (e.target === lightboxImg) {
    isDragging = true;
    startX = e.clientX - translateX;
    startY = e.clientY - translateY;
    lightbox.classList.add('dragging');
    e.preventDefault();
  }
});

lightbox.addEventListener('mousemove', e => {
  if (isDragging) {
    translateX = e.clientX - startX;
    translateY = e.clientY - startY;
    updateTransform();
    e.preventDefault();
  }
});

lightbox.addEventListener('mouseup', e => {
  if (isDragging) {
    isDragging = false;
    lightbox.classList.remove('dragging');
  }
});

lightbox.addEventListener('mouseleave', e => {
  if (isDragging) {
    isDragging = false;
    lightbox.classList.remove('dragging');
  }
});

lightbox.addEventListener('click', e => {
  if (e.target === lightbox) {
    closeLightbox();
  }
});

document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && lightbox.classList.contains('active')) {
    closeLightbox();
  }
});

go.onclick = doSearch;
q.addEventListener('keydown', e => { if (e.key === 'Enter') doSearch(); });

// Initial Page Load Actions
applyInitialTheme();
refreshState();
// Logic to populate initial file list and then update button visibility
(async () => {
    const initialFiles = await getFileList();
    filesDiv.innerHTML = '';
    initialFiles.forEach(fileName => {
        const row = createFileRow(fileName, fileName); // Use fileName as a simple ID
        row.querySelector('progress').style.display = 'none';
        row.querySelector('span').textContent = 'indexed';
        row.querySelector('.btn.danger').style.display = 'block';
        filesDiv.append(row);
    });
    updateDeleteAllButtonVisibility();
})();

// --- END: JavaScript block ---
</script>
</body>
</html>
""")

# Indexing and Search Endpoints

def _rebuild_index_from_docs() -> None:
    global _INDEX
    writer = IndexWriter(_ANALYZER, index_stopwords=True)
    for did, (text, page_map) in _DOC_DATA.items():
        writer.add(did, text, page_map=page_map)
    _INDEX = writer.commit()

@app.post("/index-files", response_model=IndexOut, dependencies=[Depends(_check_auth)])
async def index_files(files: list[UploadFile] = File(...)) -> IndexOut: # noqa: B008
    if not files:
        raise HTTPException(400, "no files")

    added = 0
    for f in files:
        dest = UPLOAD_DIR / f.filename
        with dest.open("wb") as w:
            w.write(await f.read())

        doc_id = _safe_doc_id(f.filename)
        _DOC_PATHS[doc_id] = str(dest)

        if f.filename.lower().endswith(".pdf"):
            try:
                text, page_map = _extract_pdf_text_and_page_map(dest)
            except Exception as e:
                # Add logging for PDF extraction failure
                print(f"Error extracting PDF text for {f.filename}: {e}")
                # Clean up the partially processed file/data if needed
                if dest.exists():
                    dest.unlink()
                _DOC_PATHS.pop(doc_id, None)
                continue # Skip this file and proceed to the next one
        else:
            text = dest.read_text(encoding="utf-8", errors="ignore")
            page_map = []

        _DOC_DATA[doc_id] = (text, page_map)
        added += 1

    with _INDEX_LOCK:
        _rebuild_index_from_docs()

    return IndexOut(ok=True, indexed=added, vocab=_INDEX.vocabulary_size() if _INDEX else 0)

@app.get("/files", response_model=FileListOut, dependencies=[Depends(_check_auth)])
def list_files() -> FileListOut:
    return FileListOut(files=sorted(_DOC_DATA.keys()))

@app.delete("/files", response_model=DeleteOut, dependencies=[Depends(_check_auth)])
def delete_all_files() -> DeleteOut:
    """Deletes all uploaded files and clears the index."""
    global _INDEX
    doc_ids_to_delete = list(_DOC_DATA.keys())

    # Delete physical files
    for doc_id in doc_ids_to_delete:
        p = UPLOAD_DIR / doc_id
        if p.exists():
            try:
                p.unlink()
            except Exception:
                # Log or ignore errors if a file is locked, etc.
                pass

    # Clear in-memory data stores
    _DOC_DATA.clear()
    _DOC_PATHS.clear()

    # Reset the index
    with _INDEX_LOCK:
        _INDEX = None

    return DeleteOut(ok=True, message="All files removed and index cleared.")

@app.delete("/files/{name}", response_model=DeleteOut, dependencies=[Depends(_check_auth)])
def delete_file(name: str) -> DeleteOut:
    doc_id = _safe_doc_id(name)
    _DOC_DATA.pop(doc_id, None)
    _DOC_PATHS.pop(doc_id, None)
    p = UPLOAD_DIR / doc_id
    if p.exists():
        try:
            p.unlink()
        except Exception:
            pass

    with _INDEX_LOCK:
        if _DOC_DATA:
            _rebuild_index_from_docs()
        else:

            global _INDEX
            _INDEX = None

    return DeleteOut(ok=True, message=f"removed {doc_id}")

@app.post("/search", response_model=SearchOut, dependencies=[Depends(_check_auth)])
def search_endpoint(payload: SearchIn) -> SearchOut:
    results = _perform_search(
        payload.q,
        mode=payload.mode,
        doc_id_filter=payload.doc_id,
        max_hits_per_doc=payload.max_hits_per_doc,
        context_lines=payload.context_lines,
    )
    return SearchOut(results=results)

@app.get(
    "/page-snapshot",
    response_class=StreamingResponse,
    dependencies=[Depends(_check_auth)],
)
def page_snapshot(
    doc_id: Annotated[str, Query(..., description="document id (= filename)")],
    page: Annotated[int, Query(..., ge=1, description="1-based page number")],
    x0: float = 0,  
    y0: float = 0,
    x1: float = 612,
    y1: float = 792,
    full: bool = False,             
    terms: str | None = None,        
    boxes: str | None = None,        
    scale: float = 3.0,              
):
    """High-DPI PNG snapshot; highlight terms and draw table boxes (yellow)."""
    pdf_path = _DOC_PATHS.get(doc_id)
    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(404, "file not found")

    if not (1.0 <= scale <= 4.0):
        scale = 3.0

    # Parse boxes
    box_list: list[tuple[float, float, float, float]] = []
    if boxes:
        try:
            for seg in re.split(r"[|]", boxes):
                if not seg:
                    continue
                xs = [float(v) for v in seg.split(",")]
                if len(xs) == 4:
                    box_list.append((xs[0], xs[1], xs[2], xs[3]))
        except Exception:
            box_list = []

    # Parse terms
    term_list: list[str] = []
    if terms:
        # Decode URL-encoded terms (e.g., "power%20usage" -> "power usage")
        decoded_terms = ul.unquote_plus(terms)
        # Split by comma or whitespace
        term_list = [t for t in re.split(r"[,\s]+", decoded_terms) if t]

    doc = None
    try:
        # Open PDF
        doc = fitz.open(pdf_path)  # type: ignore[no-untyped-call]
        if page < 1 or page > doc.page_count:
            raise HTTPException(404, f"Page {page} not found in {doc_id}")
            
        p = doc.load_page(page - 1)  # 0-based

        # highlight terms (case-insensitive without unavailable flags)
        if term_list:
            def _search_rects_ci(page: fitz.Page, term: str):
                # Try common case variants and deduplicate rectangles
                variants = {term, term.lower(), term.upper(), term.title()}
                rects = []
                for v in variants:
                    try:
                        # PyMuPDF 1.26.4: search_for is case-sensitive; no TEXT_IGNORECASE flag.
                        rects.extend(page.search_for(v))  # type: ignore[attr-defined]
                    except Exception:
                        pass
                # de-dup by rounded coords to avoid multi-variant duplicates
                seen, out = set(), []
                for r in rects:
                    key = (round(r.x0, 2), round(r.y0, 2), round(r.x1, 2), round(r.y1, 2))
                    if key not in seen:
                        seen.add(key)
                        out.append(r)
                return out

            for t in term_list:
                for r in _search_rects_ci(p, t):
                    try:
                        p.add_highlight_annot(r)  # default highlight (yellow-ish)
                    except Exception:
                        pass

        #  draw boxes
        for (bx0, by0, bx1, by1) in box_list:
            rect = fitz.Rect(bx0, by0, bx1, by1)
            ann = p.add_rect_annot(rect)
            try:
                ann.set_colors(stroke=(1, 1, 0))  # yellow
                ann.set_border(width=2)
                ann.update()
            except Exception:
                pass

        # get pixmap
        if full:
            clip_rect = p.mediabox
        else:
            clip_rect = fitz.Rect(x0, y0, x1, y1)

        mat = fitz.Matrix(scale, scale)
        pix = p.get_pixmap(matrix=mat, clip=clip_rect, alpha=False)  
        buf = io.BytesIO(pix.tobytes("png")) 
        img = Image.open(buf).convert("RGB")  

        # 
        draw = ImageDraw.Draw(img)
        w, h = img.size
        # Draw a subtle border around the image
        draw.rectangle([(0, 0), (w - 1, h - 1)], outline=(43, 58, 85))

        bio = io.BytesIO()
        img.save(bio, format="PNG")
        bio.seek(0)
        return StreamingResponse(bio, media_type="image/png")
    
    except HTTPException:
        # Re-raise explicit HTTP errors (404, etc.)
        raise
        
    except Exception as e:
        # 捕获并记录其他所有异常，这是解决 500 问题的关键
        import traceback
        print(f"🚨 FATAL: Error generating snapshot for {doc_id} page {page}: {e}")
        print(traceback.format_exc())
        
        # 返回 500 错误信息
        raise HTTPException(  # noqa: B904
            status_code=500, 
            detail=f"Failed to generate page snapshot due to internal error. Check server logs for details. Error: {type(e).__name__}"
        )
        
    finally:
        if doc:
            doc.close()