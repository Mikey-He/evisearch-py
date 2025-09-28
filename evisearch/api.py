#ruff: noqa: E501
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

# stdlib
import html
import io
import math
import os
from pathlib import Path
import re
import threading
from typing import Annotated, Any

# third-party
from fastapi import Body, Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import pdfplumber
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

# optional third-party
try:
    import fitz  # type: ignore[import-untyped]  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None  # type: ignore


# -----------------------------------------------------------------------------
# Simple text analyzer / indexer / searcher (self-contained)
# -----------------------------------------------------------------------------
class Analyzer:
    _rx = re.compile(r"\w+", re.UNICODE)

    def iter_tokens(self, text: str) -> Iterable[str]:
        for m in self._rx.finditer(text.lower()):
            yield m.group(0)


@dataclass
class InvertedIndex:
    postings: dict[str, dict[str, list[int]]] = field(default_factory=dict)
    doc_ids: list[str] = field(default_factory=list)
    doc_lengths: dict[str, int] = field(default_factory=dict)
    page_map: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    doc_lines: dict[str, list[str]] = field(default_factory=dict)
    line_of_pos: dict[str, list[int]] = field(default_factory=dict)

    def vocabulary_size(self) -> int:
        return len(self.postings)

    def get_posting(self, term: str) -> dict[str, list[int]]:
        return self.postings.get(term, {})

    def df(self, term: str) -> int:
        return len(self.postings.get(term, {}))


class IndexWriter:
    def __init__(self, analyzer: Analyzer) -> None:
        self.analyzer = analyzer
        self.idx = InvertedIndex()

    def add(
        self,
        doc_id: str,
        text: str,
        page_map: list[tuple[int, int]] | None = None,
    ) -> None:
        tokens = list(self.analyzer.iter_tokens(text))
        self.idx.doc_ids.append(doc_id)
        self.idx.doc_lengths[doc_id] = len(tokens)
        self.idx.page_map[doc_id] = list(page_map or [])

        # Build line mapping
        lines = text.splitlines()
        self.idx.doc_lines[doc_id] = lines
        line_of_pos: list[int] = []
        pos = 0
        for ln, line in enumerate(lines):
            for _tok in self.analyzer.iter_tokens(line):
                line_of_pos.append(ln)
                pos += 1
        # If the count mismatches, fall back to proportional fill
        if len(line_of_pos) != len(tokens):
            total = max(1, len(tokens))
            last_line = max(0, len(lines) - 1)
            line_of_pos = [min(last_line, int(i * len(lines) / total)) for i in range(len(tokens))]
        self.idx.line_of_pos[doc_id] = line_of_pos

        # Postings with positions
        ppos = 0
        for t in tokens:
            self.idx.postings.setdefault(t, {}).setdefault(doc_id, []).append(ppos)
            ppos += 1

    def commit(self) -> InvertedIndex:
        return self.idx


class Searcher:
    def __init__(self, index: InvertedIndex, analyzer: Analyzer) -> None:
        self.index = index
        self.analyzer = analyzer

    def _idf(self, term: str) -> float:
        df = self.index.df(term)
        n = max(1, len(self.index.doc_ids))
        return math.log((n + 1) / (df + 1)) + 1.0

    def ranked(self, q: str) -> list[tuple[str, float]]:
        terms = list(self.analyzer.iter_tokens(q))
        scores: dict[str, float] = {}
        for t in terms:
            posting = self.index.get_posting(t)
            idf = self._idf(t)
            for did, plist in posting.items():
                tf = len(plist)
                scores[did] = scores.get(did, 0.0) + (tf * idf)
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    def boolean(self, q: str) -> list[str]:
        # VERY small boolean: terms + AND/OR/NOT with parentheses
        terms = set(self.analyzer.iter_tokens(q))
        if not terms:
            return []

        def term_docs(t: str) -> set[str]:
            return set(self.index.get_posting(t).keys())

        def prec(op: str) -> int:
            return {"NOT": 3, "AND": 2, "OR": 1}.get(op, 0)

        tokens: list[str] = []
        for m in re.finditer(r"\(|\)|AND|OR|NOT|\w+", q, re.I):
            tokens.append(m.group(0).upper())

        output: list[str] = []
        ops: list[str] = []
        for tok in tokens:
            if tok == "(":
                ops.append(tok)
            elif tok == ")":
                while ops and ops[-1] != "(":
                    output.append(ops.pop())
                if ops and ops[-1] == "(":
                    ops.pop()
            elif tok in {"AND", "OR", "NOT"}:
                while ops and ops[-1] != "(" and prec(ops[-1]) >= prec(tok):
                    output.append(ops.pop())
                ops.append(tok)
            else:
                output.append(tok)
        while ops:
            output.append(ops.pop())

        stack: list[set[str]] = []
        for tok in output:
            if tok in {"AND", "OR", "NOT"}:
                if tok == "NOT":
                    a = stack.pop() if stack else set()
                    all_docs = set(self.index.doc_ids)
                    stack.append(all_docs - a)
                else:
                    b = stack.pop() if stack else set()
                    a = stack.pop() if stack else set()
                    stack.append((a & b) if tok == "AND" else (a | b))
            else:
                stack.append(term_docs(tok.lower()))
        return sorted(stack[-1] if stack else set())

    def phrase_starts(self, q: str, doc_id: str) -> list[int]:
        words = list(self.analyzer.iter_tokens(q))
        if not words:
            return []
        first = words[0]
        starts: list[int] = []
        posting = self.index.get_posting(first).get(doc_id, [])
        for s in posting:
            ok = True
            for i, w in enumerate(words):
                plist = self.index.get_posting(w).get(doc_id, [])
                if (s + i) not in plist:
                    ok = False
                    break
            if ok:
                starts.append(s)
        return starts


# -----------------------------------------------------------------------------
# FastAPI app and state
# -----------------------------------------------------------------------------
app = FastAPI(title="EviSearch-Py", version="2.0.2")

DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads"
DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_ANALYZER = Analyzer()
_INDEX_LOCK = threading.Lock()
_INDEX: InvertedIndex | None = None

# Cache for PDF path (for snapshot) and raw texts for rebuilds
_DOC_PATHS: dict[str, str] = {}
_DOC_TEXTS: dict[str, tuple[str, list[tuple[int, int]]]] = {}

# windows for snippets
LINES_WINDOW = 1  # text: show hit line ±N lines
COL_WINDOW = 1  # table: show hit col ±N columns

# Maximum hits to return
MAX_HITS_DEFAULT = 3  # Default number of hits to show
MAX_HITS_ALL = 50  # Maximum when showing all

# Optional Basic Auth
basic_user = os.getenv("BASIC_USER") or ""
basic_pass = os.getenv("BASIC_PASS") or ""
security = HTTPBasic()


def _needs_auth() -> bool:
    return bool(basic_user or basic_pass)


def _check_auth(creds: HTTPBasicCredentials = Depends(security)) -> None:  # noqa: B008
    if not _needs_auth():
        return
    ok = creds and creds.username == basic_user and creds.password == basic_pass
    if not ok:
        raise HTTPException(status_code=401, detail="Unauthorized")


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
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


class DeleteOut(BaseModel):
    ok: bool
    remaining: int


class StatusOut(BaseModel):
    app: str
    docs: int
    vocab: int
    auth: str


class SearchIn(BaseModel):
    q: str = Field(min_length=1)
    max_hits_per_doc: int = Field(default=3, ge=1, le=MAX_HITS_ALL)
    context_lines: int = Field(default=1, ge=0, le=8)
    doc_id: str | None = None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _extract_pdf_text_and_page_map(path: Path) -> tuple[str, list[tuple[int, int]]]:
    texts: list[str] = []
    page_map: list[tuple[int, int]] = []
    pos = 0
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text() or ""
            texts.append(txt)
            toks = list(_ANALYZER.iter_tokens(txt))
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
    page: int | None = None
    for start_pos, pg in pm:
        if start_pos <= pos:
            page = pg
        else:
            break
    return page


def _representative_pos(doc_id: str, terms: list[str]) -> int | None:
    if _INDEX is None:
        return None
    best: int | None = None
    for t in terms:
        posting = _INDEX.get_posting(t)
        plist = posting.get(doc_id, [])
        if plist:
            cand = plist[0]
            if best is None or cand < best:
                best = cand
    return best


def _all_term_positions(doc_id: str, terms: list[str]) -> list[int]:
    if _INDEX is None:
        return []
    seen: set[int] = set()
    out: list[int] = []
    for t in terms:
        posting = _INDEX.get_posting(t)
        for p in posting.get(doc_id, []):
            if p not in seen:
                seen.add(p)
                out.append(p)
    out.sort()
    return out


def _highlight_terms(text: str, terms: list[str]) -> str:
    if not terms:
        return html.escape(text)
    pats = sorted({t for t in terms if t}, key=len, reverse=True)
    rx = re.compile(r"\b(" + "|".join(re.escape(t) for t in pats) + r")\b", re.IGNORECASE)
    return rx.sub(lambda m: f"<mark>{html.escape(m.group(0))}</mark>", html.escape(text))


def _paragraph_snippet(
    doc_id: str,
    start_pos: int | None,
    terms: list[str],
    lwin: int = LINES_WINDOW,
) -> dict[str, Any]:
    if _INDEX is None or start_pos is None:
        return {"kind": "text", "snippet_html": "", "line": 0, "page": None}

    lines = _INDEX.doc_lines.get(doc_id, [])
    line_of_pos = _INDEX.line_of_pos.get(doc_id, [])
    if not lines or not line_of_pos or start_pos >= len(line_of_pos):
        return {"kind": "text", "snippet_html": "", "line": 0, "page": None}

    hit = line_of_pos[start_pos]
    s = max(0, hit - lwin)
    e = min(len(lines) - 1, hit + lwin)
    block = "\n".join(lines[s : e + 1])

    page = _page_of_pos(doc_id, start_pos)
    html_block = "<pre class='snippet'>" + _highlight_terms(block, terms) + "</pre>"
    return {"kind": "text", "snippet_html": html_block, "line": hit + 1, "page": page}


def _is_numericish(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return False
    return bool(re.search(r"\d", s))


def _table_snippet(
    doc_id: str,
    page_no: int,
    terms: list[str],
    cwin: int = COL_WINDOW,
    rwin: int = 1,
) -> dict[str, Any] | None:
    pdf_path = _DOC_PATHS.get(doc_id)
    if not pdf_path or not os.path.exists(pdf_path):
        return None
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
                1 for t in terms_l if re.search(rf"\b{re.escape(t)}\b", flat, re.IGNORECASE)
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
                if any(re.search(rf"\b{re.escape(t)}\b", val, re.IGNORECASE) for t in terms_l):
                    hit_rc.append((ri, ci))
        if not hit_rc:
            return None

        max_rows = len(grid)
        max_cols = len(grid[0]) if grid and grid[0] else 0
        r0 = min(r for r, _ in hit_rc)
        r1 = max(r for r, _ in hit_rc)
        c0 = min(c for _, c in hit_rc)
        c1 = max(c for _, c in hit_rc)
        r0 = max(0, r0 - rwin)
        r1 = min(max_rows - 1, r1 + rwin)
        c0 = max(0, c0 - cwin)
        c1 = min((max_cols - 1) if max_cols else 0, c1 + cwin)

        header_row_idx = 0
        first_row_empty = grid and not any((grid[0][ci] or "").strip() for ci in range(len(grid[0])))
        if first_row_empty:
            for rr in range(r0 - 1, -1, -1):
                row_not_empty = any((grid[rr][ci] or "").strip() for ci in range(len(grid[rr])))
                if row_not_empty:
                    header_row_idx = rr
                    break

        def tr(row: list[str], th: bool = False) -> str:
            tag = "th" if th else "td"
            cells: list[str] = []
            for c in row:
                v = (c or "").strip()
                if _is_numericish(v):
                    cells.append(f"<{tag} class='num'>{html.escape(v)}</{tag}>")
                else:
                    cells.append(f"<{tag}>{html.escape(v)}</{tag}>")
            return "<tr>" + "".join(cells) + "</tr>"

        # Build sub table
        sub_rows = [grid[i][c0 : c1 + 1] for i in range(r0, r1 + 1)]
        html_rows: list[str] = []
        if header_row_idx < len(grid):
            html_rows.append(tr(grid[header_row_idx], th=True))
        for r in sub_rows:
            html_rows.append(tr(r, th=False))

        bbox = getattr(tb, "bbox", None)
        if not bbox:
            return None
        x0, y0, x1, y1 = bbox
        table_html = "<table class='snippet-table'>" + "".join(html_rows) + "</table>"
        return {
            "kind": "table",
            "table_html": table_html,
            "bbox": [x0, y0, x1, y1],
            "page": page_no + 1,
        }


def _auto_mode(q: str) -> str:
    if '"' in q:
        return "phrase"
    if re.search(r"\b(AND|OR|NOT)\b|\(|\)", q, re.IGNORECASE):
        return "boolean"
    return "ranked"


def _rebuild_index() -> None:
    global _INDEX
    with _INDEX_LOCK:
        writer = IndexWriter(_ANALYZER)
        for did, (text, page_map) in _DOC_TEXTS.items():
            writer.add(did, text, page_map=page_map or None)
        _INDEX = writer.commit()


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/", response_model=StatusOut)
def root_status(_auth: None = Depends(_check_auth)) -> StatusOut:  # noqa: B008
    return StatusOut(
        app="EviSearch-Py",
        docs=len(_INDEX.doc_ids) if _INDEX else 0,
        vocab=_INDEX.vocabulary_size() if _INDEX else 0,
        auth="on" if _needs_auth() else "off",
    )


@app.get("/ui", response_class=HTMLResponse)
def ui_page() -> HTMLResponse:
    html_doc = (
        "<!doctype html>\n"
        "<html>\n"
        "<head>\n"
        '  <meta charset="utf-8">\n'
        '  <meta name="viewport" content="width=device-width,initial-scale=1">\n'
        "  <title>EviSearch-Py</title>\n"
        '  <link rel="stylesheet" href="/ui.css">\n'
        "</head>\n"
        "<body>\n"
        '  <header class="top">EviSearch-Py</header>\n'
        '  <main class="wrap">\n'
        '    <section class="uploader">\n'
        '      <div id="drop">Drop or choose files</div>\n'
        '      <input id="pick" type="file" multiple>\n'
        '      <button id="clearAll">Clear All</button>\n'
        '      <div id="list"></div>\n'
        "    </section>\n"
        '    <section class="search">\n'
        '      <input id="q" placeholder="Search terms">\n'
        '      <label>Max hits/doc <input id="maxhits" type="number" value="3"></label>\n'
        '      <label>Context lines <input id="ctx" type="number" value="1"></label>\n'
        '      <button id="go">Search</button>\n'
        '      <div id="out"></div>\n'
        "    </section>\n"
        "  </main>\n"
        '  <div id="lightbox" class="hide"><img id="big"></div>\n'
        '  <script src="/ui.js"></script>\n'
        "</body>\n"
        "</html>\n"
    )
    return HTMLResponse(html_doc)


@app.get("/ui.css", response_class=HTMLResponse)
def ui_css() -> HTMLResponse:
    css = (
        ":root{--bg:#0b0f18;--fg:#e7eaf2;--card:#131a27;--accent:#2ea043;}\n"
        "body{background:var(--bg);color:var(--fg);font-family:system-ui,Arial;}\n"
        ".wrap{max-width:980px;margin:20px auto;padding:12px;}\n"
        ".top{font-weight:700;padding:12px 0;text-align:center;}\n"
        ".uploader,.search{background:var(--card);padding:12px;border-radius:12px;}\n"
        "#drop{border:1px dashed #345;padding:24px;text-align:center;margin-bottom:8px;}\n"
        "#list .row{display:flex;gap:8px;align-items:center;margin:6px 0;}\n"
        "#list .name{flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}\n"
        ".btn{padding:4px 8px;border:1px solid #456;border-radius:8px;background:#182131;}\n"
        ".hit{border-top:1px solid #234;padding:8px 0;}\n"
        ".snippet{white-space:pre-wrap;line-height:1.35;}\n"
        "#lightbox{position:fixed;inset:0;background:rgba(0,0,0,.85);}\n"
        "#lightbox.hide{display:none;}\n"
        "#lightbox img{max-width:none;cursor:grab;transform-origin:0 0;}\n"
        "table.snippet-table{border-collapse:collapse;width:100%;}\n"
        "table.snippet-table td,table.snippet-table th{border:1px solid #223;padding:4px;}\n"
        "table.snippet-table th{background:#0f1726;}\n"
        "td.num{text-align:right;}\n"
    )
    return HTMLResponse(css, media_type="text/css")


@app.get("/ui.js", response_class=HTMLResponse)
def ui_js() -> HTMLResponse:
    js = (
        "(function(){\n"
        "const drop = document.getElementById('drop');\n"
        "const pick = document.getElementById('pick');\n"
        "const list = document.getElementById('list');\n"
        "const clearAll = document.getElementById('clearAll');\n"
        "const q = document.getElementById('q');\n"
        "const maxhits = document.getElementById('maxhits');\n"
        "const ctx = document.getElementById('ctx');\n"
        "const go = document.getElementById('go');\n"
        "const out = document.getElementById('out');\n"
        "const lb = document.getElementById('lightbox');\n"
        "const big = document.getElementById('big');\n"
        "function row(did){\n"
        "  const el = document.createElement('div'); el.className='row';\n"
        "  el.innerHTML = '<span class=\"name\"></span>'+\n"
        "    '<progress max=\"100\" value=\"0\"></progress>'+\n"
        "    '<button class=\"btn cancel\">×</button>'+\n"
        "    '<button class=\"btn del\">Delete</button>';\n"
        "  el.querySelector('.name').textContent = did; return el;\n"
        "}\n"
        "function upload(file){\n"
        "  const did = file.name; const r = row(did); list.appendChild(r);\n"
        "  const xhr = new XMLHttpRequest();\n"
        "  xhr.open('POST','/index-files');\n"
        "  const fd = new FormData(); fd.append('files', file, file.name);\n"
        "  xhr.upload.onprogress = e=>{ if(e.lengthComputable){\n"
        "    r.querySelector('progress').value = Math.round(e.loaded*100/e.total);} };\n"
        "  r.querySelector('.cancel').onclick = ()=>{ xhr.abort(); r.remove(); };\n"
        "  r.querySelector('.del').onclick = async ()=>{\n"
        "    await fetch('/files/'+encodeURIComponent(did),{method:'DELETE'});\n"
        "    r.remove(); };\n"
        "  xhr.onload = ()=>{ r.querySelector('progress').value=100; };\n"
        "  xhr.send(fd);\n"
        "}\n"
        "drop.ondragover = e=>{e.preventDefault();};\n"
        "drop.ondrop = e=>{e.preventDefault(); for(const f of e.dataTransfer.files) upload(f);};\n"
        "pick.onchange = ()=>{ for(const f of pick.files) upload(f); pick.value=''; };\n"
        "clearAll.onclick = async ()=>{ await fetch('/files',{method:'DELETE'}); list.innerHTML=''; };\n"
        "go.onclick = async ()=>{\n"
        "  const body = { q:q.value, max_hits_per_doc:+maxhits.value, context_lines:+ctx.value };\n"
        "  const r = await fetch('/search',{\n"
        "    method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});\n"
        "  const data = await r.json(); render(data.results);\n"
        "};\n"
        "function render(results){ out.innerHTML='';\n"
        "  for(const item of results){\n"
        "    const card = document.createElement('div'); card.className='card';\n"
        "    card.innerHTML = '<h4>'+item.doc_id+'</h4>';\n"
        "    for(const h of (item.hits||[])){\n"
        "      const wrap = document.createElement('div'); wrap.className='hit';\n"
        "      if(h.kind==='table' && h.snapshot_url){\n"
        "        const img = document.createElement('img'); img.src=h.snapshot_url; img.style.maxWidth='320px';\n"
        "        img.style.border='1px solid #223'; img.style.cursor='zoom-in';\n"
        "        img.onclick = ()=> openLightbox(img.src); wrap.appendChild(img);\n"
        "      } else { wrap.innerHTML += '<div class=\"snippet\">'+(h.snippet_html||'')+'</div>'; }\n"
        "      card.appendChild(wrap);\n"
        "    } out.appendChild(card);\n"
        "  }\n"
        "}\n"
        "function openLightbox(src){ lb.classList.remove('hide'); big.src = src; big.style.transform='scale(1)';\n"
        "  big.dataset.scale='1'; big.dataset.x='0'; big.dataset.y='0'; }\n"
        "lb.onclick = e=>{ if(e.target===lb) lb.classList.add('hide'); };\n"
        "document.addEventListener('keydown',e=>{ if(e.key==='Escape') lb.classList.add('hide'); });\n"
        "let drag=false, lastX=0, lastY=0;\n"
        "big.onmousedown = e=>{ drag=true; lastX=e.clientX; lastY=e.clientY; big.style.cursor='grabbing'; };\n"
        "document.onmouseup = ()=>{ drag=false; big.style.cursor='grab'; };\n"
        "document.onmousemove = e=>{ if(!drag) return;\n"
        "  const dx=e.clientX-lastX, dy=e.clientY-lastY; lastX=e.clientX; lastY=e.clientY;\n"
        "  const x=+big.dataset.x+dx, y=+big.dataset.y+dy; big.dataset.x=x; big.dataset.y=y; applyTransform(); };\n"
        "big.onwheel = e=>{ e.preventDefault(); const s=+big.dataset.scale;\n"
        "  const ns = Math.min(6, Math.max(0.5, s + (e.deltaY<0?0.2:-0.2))); big.dataset.scale = ns.toString(); applyTransform(); };\n"
        "function applyTransform(){ const s=big.dataset.scale, x=big.dataset.x, y=big.dataset.y;\n"
        "  big.style.transform = 'translate('+x+'px,'+y+'px) scale('+s+')'; }\n"
        "})();\n"
    )
    return HTMLResponse(js, media_type="application/javascript")


@app.post("/index", response_model=IndexOut)
def index_docs(
    payload: Annotated[IndexIn, Body(...)],
    _auth: None = Depends(_check_auth),  # noqa: B008
) -> IndexOut:
    global _DOC_TEXTS, _DOC_PATHS
    _DOC_TEXTS = {d.id: (d.text, d.page_map or []) for d in payload.docs}
    _DOC_PATHS = {}
    _rebuild_index()
    return IndexOut(
        ok=True,
        indexed=len(_INDEX.doc_ids) if _INDEX else 0,
        vocab=_INDEX.vocabulary_size() if _INDEX else 0,
    )


@app.post("/index-files", response_model=IndexOut)
async def index_files(
    files: list[UploadFile] = File(...),  # noqa: B008
    _auth: None = Depends(_check_auth),  # noqa: B008
) -> IndexOut:
    if not files:
        raise HTTPException(400, "No files")

    global _DOC_TEXTS, _DOC_PATHS

    for f in files:
        name = f.filename or "unnamed"
        doc_id = name  # keep extension
        suffix = os.path.splitext(name)[1].lower()
        raw = await f.read()
        if not raw:
            continue
        out_path = UPLOAD_DIR / name
        out_path.write_bytes(raw)

        if suffix == ".pdf":
            text, page_map = _extract_pdf_text_and_page_map(out_path)
            _DOC_TEXTS[doc_id] = (text, page_map)
            _DOC_PATHS[doc_id] = str(out_path)
        elif suffix in {".txt", ".md", ".csv"}:
            try:
                text = raw.decode("utf-8", errors="replace")
            except Exception:
                text = raw.decode("latin-1", errors="replace")
            _DOC_TEXTS[doc_id] = (text, [])
        else:
            # unsupported: keep file, but do not index
            pass

    _rebuild_index()
    return IndexOut(
        ok=True,
        indexed=len(_INDEX.doc_ids) if _INDEX else 0,
        vocab=_INDEX.vocabulary_size() if _INDEX else 0,
    )


@app.delete("/files/{doc_id}", response_model=DeleteOut)
def delete_file(doc_id: str, _auth: None = Depends(_check_auth)) -> DeleteOut:  # noqa: B008
    global _DOC_TEXTS, _DOC_PATHS
    if doc_id not in _DOC_TEXTS:
        raise HTTPException(404, f"doc_id not found: {doc_id}")
    _DOC_TEXTS.pop(doc_id, None)
    _DOC_PATHS.pop(doc_id, None)
    _rebuild_index()
    remain = len(_INDEX.doc_ids) if _INDEX else 0
    return DeleteOut(ok=True, remaining=remain)


@app.delete("/files", response_model=DeleteOut)
def clear_all(_auth: None = Depends(_check_auth)) -> DeleteOut:  # noqa: B008
    global _DOC_TEXTS, _DOC_PATHS, _INDEX
    _DOC_TEXTS.clear()
    _DOC_PATHS.clear()
    _INDEX = None
    return DeleteOut(ok=True, remaining=0)


# -----------------------------------------------------------------------------
# Searching
# -----------------------------------------------------------------------------
@app.get("/search", response_model=SearchOut)
def search(
    q: Annotated[str, Query(min_length=1)],
    all_hits: bool = Query(False, description="Return all hits"),
    doc_id: str | None = Query(None, description="Filter to doc_id"),
    _auth: None = Depends(_check_auth),  # noqa: B008
) -> SearchOut:
    if _INDEX is None:
        raise HTTPException(400, "Index empty. Upload first on /ui or /index.")

    mode = _auto_mode(q)
    results: list[dict[str, Any]] = []
    terms = list(_ANALYZER.iter_tokens(q))

    # doc iterator helper
    def each_doc(doc_iterable: Iterable[str]) -> Iterable[str]:
        for did in doc_iterable:
            if doc_id and did != doc_id:
                continue
            yield did

    if mode == "phrase":
        sch = Searcher(_INDEX, _ANALYZER)
        for did in each_doc(_INDEX.doc_ids):
            starts = sch.phrase_starts(q, did)
            if not starts:
                continue
            limit = MAX_HITS_ALL if all_hits else MAX_HITS_DEFAULT
            hits_to_show = starts[:limit]
            out_hits: list[dict[str, Any]] = []
            for st in hits_to_show:
                page = _page_of_pos(did, st) or 1
                table = _table_snippet(did, page - 1, terms)
                if table:
                    bbox = table["bbox"]
                    table["snapshot_url"] = (
                        f"/page-snapshot?doc_id={did}&page={table['page']}"
                        f"&x0={bbox[0]}&y0={bbox[1]}&x1={bbox[2]}&y1={bbox[3]}"
                    )
                    out_hits.append(table)
                else:
                    out_hits.append(_paragraph_snippet(did, st, terms))
            obj: dict[str, Any] = {"doc_id": did, "hits": out_hits, "total_hits": len(starts)}
            if not all_hits and len(starts) > MAX_HITS_DEFAULT:
                obj["has_more"] = True
            results.append(obj)

    elif mode == "boolean":
        sch = Searcher(_INDEX, _ANALYZER)
        docs = sch.boolean(q)
        for did in each_doc(docs):
            all_positions = _all_term_positions(did, terms)
            if not all_positions:
                continue
            limit = MAX_HITS_ALL if all_hits else MAX_HITS_DEFAULT
            positions_to_show = all_positions[:limit]
            out_hits = []
            for pos in positions_to_show:
                page = _page_of_pos(did, pos) or 1
                table = _table_snippet(did, page - 1, terms)
                hit = table or _paragraph_snippet(did, pos, terms)
                if table:
                    bbox = table["bbox"]
                    hit["snapshot_url"] = (
                        f"/page-snapshot?doc_id={did}&page={table['page']}"
                        f"&x0={bbox[0]}&y0={bbox[1]}&x1={bbox[2]}&y1={bbox[3]}"
                    )
                out_hits.append(hit)
            obj = {"doc_id": did, "hits": out_hits, "total_hits": len(all_positions)}
            if not all_hits and len(all_positions) > MAX_HITS_DEFAULT:
                obj["has_more"] = True
            results.append(obj)

    else:  # ranked
        sch = Searcher(_INDEX, _ANALYZER)
        for did, score in sch.ranked(q)[:20]:
            if doc_id and did != doc_id:
                continue
            all_positions = _all_term_positions(did, terms)
            if not all_positions:
                rep = _representative_pos(did, terms)
                if rep is None:
                    continue
                all_positions = [rep]
            limit = MAX_HITS_ALL if all_hits else MAX_HITS_DEFAULT
            positions_to_show = all_positions[:limit]
            out_hits = []
            for pos in positions_to_show:
                page = _page_of_pos(did, pos) or 1
                table = _table_snippet(did, page - 1, terms)
                hit = table or _paragraph_snippet(did, pos, terms)
                if table:
                    bbox = table["bbox"]
                    hit["snapshot_url"] = (
                        f"/page-snapshot?doc_id={did}&page={table['page']}"
                        f"&x0={bbox[0]}&y0={bbox[1]}&x1={bbox[2]}&y1={bbox[3]}"
                    )
                out_hits.append(hit)
            obj = {
                "doc_id": did,
                "score": float(score),
                "hits": out_hits,
                "total_hits": len(all_positions),
            }
            if not all_hits and len(all_positions) > MAX_HITS_DEFAULT:
                obj["has_more"] = True
            results.append(obj)

    return SearchOut(results=results)


@app.post("/search", response_model=SearchOut)
def search_post(
    body: SearchIn,
    _auth: None = Depends(_check_auth),  # noqa: B008
) -> SearchOut:
    global LINES_WINDOW
    old = LINES_WINDOW
    LINES_WINDOW = body.context_lines
    try:
        res = search(q=body.q, all_hits=False, doc_id=body.doc_id)
    finally:
        LINES_WINDOW = old

    trimmed: list[dict[str, Any]] = []
    for item in res.results:
        hits = item.get("hits", [])
        total = item.get("total_hits", len(hits))
        limit = max(1, min(body.max_hits_per_doc, MAX_HITS_ALL))
        new_hits = hits[:limit]
        obj = dict(item)
        obj["hits"] = new_hits
        obj["has_more"] = total > limit
        trimmed.append(obj)
    return SearchOut(results=trimmed)


# -----------------------------------------------------------------------------
# Page snapshot (PDF)
# -----------------------------------------------------------------------------
@app.get("/page-snapshot")
def page_snapshot(
    doc_id: str,
    page: int = Query(..., ge=1),
    x0: float = Query(...),
    y0: float = Query(...),
    x1: float = Query(...),
    y1: float = Query(...),
    _auth: None = Depends(_check_auth),  # noqa: B008
) -> StreamingResponse:
    if fitz is None:
        raise HTTPException(500, "PyMuPDF not available")
    pdf_path = _DOC_PATHS.get(doc_id)
    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(404, f"PDF not found for {doc_id}")
    doc = fitz.open(pdf_path)
    try:
        pg = doc.load_page(page - 1)
        pix = pg.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        rx, ry = pg.rect.width, pg.rect.height
        sx = pix.width / rx
        sy = pix.height / ry
        box = (int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy))

        dr = ImageDraw.Draw(img)
        dr.rectangle(box, outline=(255, 255, 0), width=3)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    finally:
        doc.close()
