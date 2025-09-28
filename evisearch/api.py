#ruff: noqa: E501
from __future__ import annotations

import html
import io
from pathlib import Path
import re
import threading
from typing import Annotated, Literal

from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import fitz  # PyMuPDF
import pdfplumber
from pydantic import BaseModel, Field

# =========================
# App & global state
# =========================

app = FastAPI(title="EviSearch-Py", version="1.5.1")

DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads"
DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_INDEX_LOCK = threading.Lock()

# tuning knobs
DEFAULT_MAX_PAGES_PER_DOC = 3
CROP_PAD_RATIO = 0.04  # 4% padding around union box
RENDER_SCALE = 2  # 2x rendering for snapshots

# =========================
# Indexing primitives
# =========================


class Analyzer:
    """Lightweight text analyzer used for tokenization and line splitting."""

    _word = re.compile(r"[A-Za-z0-9_]+")

    def tokenize(self, text: str) -> list[str]:
        return [m.group(0).lower() for m in self._word.finditer(text or "")]

    def split_lines(self, text: str) -> list[str]:
        return (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")


class InvertedIndex:
    """
    Minimal line-level inverted index.

    - texts[doc_id]: full text
    - lines[doc_id]: list of original lines
    - page_lines[doc_id]: list[(start_line, page_no)] for PDFs
    - line_postings[token][doc_id] -> list[line_numbers]
    """

    def __init__(self) -> None:
        self.texts: dict[str, str] = {}
        self.lines: dict[str, list[str]] = {}
        self.page_lines: dict[str, list[tuple[int, int]]] = {}
        self.line_postings: dict[str, dict[str, list[int]]] = {}

    # ---- basic ops

    def add_doc(
        self,
        doc_id: str,
        text: str,
        page_to_lines: list[int] | None = None,
    ) -> None:
        self.texts[doc_id] = text
        lines = ANALYZER.split_lines(text)
        self.lines[doc_id] = lines

        # Build page -> starting line mapping for PDFs.
        page_map: list[tuple[int, int]] = []
        if page_to_lines:
            start = 0
            for i, nlines in enumerate(page_to_lines, start=1):
                page_map.append((start, i))
                start += nlines
        self.page_lines[doc_id] = page_map

        # Build line postings.
        for ln, line in enumerate(lines):
            for tok in set(ANALYZER.tokenize(line)):
                self.line_postings.setdefault(tok, {}).setdefault(doc_id, []).append(ln)

    def remove_doc(self, doc_id: str) -> None:
        if doc_id not in self.texts:
            return

        # Clean postings.
        lines = self.lines.get(doc_id, [])
        for ln, line in enumerate(lines):
            for tok in set(ANALYZER.tokenize(line)):
                d = self.line_postings.get(tok)
                if not d:
                    continue
                lst = d.get(doc_id, [])
                try:
                    lst.remove(ln)
                except ValueError:
                    pass
                if not lst:
                    d.pop(doc_id, None)
                if not d:
                    self.line_postings.pop(tok, None)

        # Clean maps.
        self.texts.pop(doc_id, None)
        self.lines.pop(doc_id, None)
        self.page_lines.pop(doc_id, None)

    def clear(self) -> None:
        self.texts.clear()
        self.lines.clear()
        self.page_lines.clear()
        self.line_postings.clear()

    def docs(self) -> int:
        return len(self.texts)

    # ---- search

    def search_lines_any(
        self,
        q: str,
        doc_id_filter: str | None = None,
    ) -> dict[str, list[int]]:
        """Return doc_id -> line numbers where any term appears."""
        terms = [t for t in ANALYZER.tokenize(q) if t]
        if not terms:
            return {}
        hits: dict[str, list[int]] = {}
        for t in terms:
            pd = self.line_postings.get(t, {})
            for did, lns in pd.items():
                if doc_id_filter and did != doc_id_filter:
                    continue
                hits.setdefault(did, [])
                hits[did].extend(lns)
        for did in list(hits.keys()):
            hits[did] = sorted(set(hits[did]))
        return hits

    def search_lines_all(
        self,
        q: str,
        doc_id_filter: str | None = None,
    ) -> dict[str, list[int]]:
        """Return doc_id -> line numbers where all terms appear."""
        terms = [t for t in ANALYZER.tokenize(q) if t]
        if not terms:
            return {}
        base = self.line_postings.get(terms[0], {})
        res: dict[str, list[int]] = {}
        for did, lns in base.items():
            if doc_id_filter and did != doc_id_filter:
                continue
            inter = set(lns)
            ok = True
            for t in terms[1:]:
                nxt = set(self.line_postings.get(t, {}).get(did, []))
                inter &= nxt
                if not inter:
                    ok = False
                    break
            if ok and inter:
                res[did] = sorted(inter)
        return res

    def which_page(self, doc_id: str, line_no: int) -> int | None:
        """Map a 0-based line number to 1-based page number for PDFs (best effort)."""
        pairs = self.page_lines.get(doc_id) or []
        page = 1
        for start_ln, p in pairs:
            if line_no >= start_ln:
                page = p
            else:
                break
        return page if pairs else None


ANALYZER = Analyzer()
INDEX = InvertedIndex()
# doc_id (original filename with extension) -> file path on disk
_DOC_PATHS: dict[str, str] = {}

# =========================
# Pydantic models
# =========================


class HitText(BaseModel):
    # Pydantic v2: use Literal instead of Field(const=True)
    kind: Literal["image", "text"] = "image"
    page: int | None = None
    line: int | None = None
    snippet_html: str = ""
    snapshot_url: str | None = None


class SearchDocResult(BaseModel):
    doc_id: str
    hits: list[HitText]
    total_hits: int  # pages for PDFs; lines for non-PDFs
    has_more: bool | None = None


class SearchOut(BaseModel):
    ok: bool
    q: str
    results: list[SearchDocResult]


class IndexOut(BaseModel):
    ok: bool
    indexed: int


class DeleteOut(BaseModel):
    ok: bool
    deleted: int


# =========================
# Helpers
# =========================


def sanitize_filename(name: str) -> str:
    """Make a filesystem-safe filename while keeping the extension."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return safe or "file"


def highlight_html(text: str, terms: list[str]) -> str:
    """Wrap matched terms with <mark> tags in an HTML-safe string."""

    def repl(m: re.Match[str]) -> str:
        return f"<mark>{html.escape(m.group(0))}</mark>"

    out = text
    for t in sorted(set(terms), key=len, reverse=True):
        if not t:
            continue
        try:
            out = re.sub(rf"(?i)\b{re.escape(t)}\b", repl, out)
        except re.error:
            out = re.sub(re.escape(t), repl, out)
    return out


def _union_rects(rects: list[fitz.Rect], page_rect: fitz.Rect) -> fitz.Rect:
    if not rects:
        return page_rect
    x0 = min(r.x0 for r in rects)
    y0 = min(r.y0 for r in rects)
    x1 = max(r.x1 for r in rects)
    y1 = max(r.y1 for r in rects)
    # padding by ratio
    pad_x = CROP_PAD_RATIO * page_rect.width
    pad_y = CROP_PAD_RATIO * page_rect.height
    x0 = max(page_rect.x0, x0 - pad_x)
    y0 = max(page_rect.y0, y0 - pad_y)
    x1 = min(page_rect.x1, x1 + pad_x)
    y1 = min(page_rect.y1, y1 + pad_y)
    return fitz.Rect(x0, y0, x1, y1)


def make_text_snippet(
    doc_id: str,
    line_no: int,
    context_lines: int,
    terms: list[str],
) -> HitText:
    """Used for non-PDF files: return highlighted HTML snippet."""
    lines = INDEX.lines.get(doc_id, [])
    lo = max(0, line_no - context_lines)
    hi = min(len(lines), line_no + context_lines + 1)
    block = lines[lo:hi]
    snippet = "\n".join(block)

    html_block = html.escape(snippet).replace("\n", "<br/>")
    html_block = highlight_html(html_block, terms)

    return HitText(kind="text", page=None, line=line_no + 1, snippet_html=html_block)


# =========================
# Indexing & file mgmt
# =========================


@app.post("/index-files", response_model=IndexOut)
async def index_files(
    files: Annotated[list[UploadFile], File(description="Files to index")],
) -> IndexOut:
    """
    Upload and (re)build the index from provided files.
    doc_id == original filename (with extension); same-name overwrites.
    """
    if not files:
        raise HTTPException(400, "No files uploaded.")

    saved: list[tuple[str, Path, str, list[int] | None]] = []
    for uf in files:
        raw_name = uf.filename or "file"
        fname = sanitize_filename(raw_name)
        path_out = UPLOAD_DIR / fname
        content = await uf.read()
        path_out.write_bytes(content)

        text = ""
        page_lines: list[int] | None = None

        if fname.lower().endswith(".pdf"):
            text_parts: list[str] = []
            page_lines = []
            with pdfplumber.open(str(path_out)) as pdf:
                for page in pdf.pages:
                    t = page.extract_text() or ""
                    lines = ANALYZER.split_lines(t)
                    text_parts.append(t)
                    page_lines.append(len(lines))
            text = "\n".join(text_parts)
        else:
            try:
                text = content.decode("utf-8", errors="replace")
            except Exception:
                text = ""

        saved.append((fname, path_out, text, page_lines))

    # Rebuild index atomically.
    with _INDEX_LOCK:
        INDEX.clear()
        _DOC_PATHS.clear()
        for name, p, text, page_map in saved:
            INDEX.add_doc(name, text, page_map)
            _DOC_PATHS[name] = str(p)

    return IndexOut(ok=True, indexed=INDEX.docs())


@app.delete("/files/{doc_id}", response_model=DeleteOut)
def delete_file(doc_id: str) -> DeleteOut:
    """Delete a single file by doc_id (original filename with extension)."""
    with _INDEX_LOCK:
        removed = 0
        path = _DOC_PATHS.pop(doc_id, None)
        if path:
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                # Ignore filesystem errors but keep index consistent.
                pass
            INDEX.remove_doc(doc_id)
            removed = 1
    return DeleteOut(ok=True, deleted=removed)


@app.delete("/files", response_model=DeleteOut)
def delete_all_files() -> DeleteOut:
    """Clear all uploaded files and reset the index."""
    with _INDEX_LOCK:
        deleted = 0
        for _, p in list(_DOC_PATHS.items()):
            try:
                Path(p).unlink(missing_ok=True)
                deleted += 1
            except Exception:
                pass
        _DOC_PATHS.clear()
        INDEX.clear()
    return DeleteOut(ok=True, deleted=deleted)


# =========================
# Search
# =========================


def _pages_from_lines(doc_id: str, line_nos: list[int]) -> list[int]:
    pages = []
    seen = set()
    for ln in line_nos:
        p = INDEX.which_page(doc_id, ln)
        if p is None:
            continue
        if p not in seen:
            pages.append(p)
            seen.add(p)
    pages.sort()
    return pages


def _build_pdf_hits_for_pages(
    doc_id: str,
    pages: list[int],
    terms: list[str],
) -> list[HitText]:
    path = _DOC_PATHS.get(doc_id)
    if not path:
        return []
    hits: list[HitText] = []
    try:
        doc = fitz.open(path)
        for pno1 in pages:
            pno0 = pno1 - 1
            if pno0 < 0 or pno0 >= doc.page_count:
                continue
            pg = doc[pno0]
            rects: list[fitz.Rect] = []
            page_rect = pg.rect

            # case-insensitive search by trying a few variants
            variants: set[str] = set()
            for t in terms:
                if not t:
                    continue
                variants.add(t)
                variants.add(t.upper())
                variants.add(t.capitalize())

            for v in variants:
                try:
                    rects.extend(pg.search_for(v))
                except Exception:
                    pass

            box = _union_rects(rects, page_rect)
            x0, y0, x1, y1 = box
            url = (
                "/page-snapshot?"
                f"doc_id={html.escape(doc_id)}&page={pno1}"
                f"&x0={x0:.2f}&y0={y0:.2f}&x1={x1:.2f}&y1={y1:.2f}&crop=1"
            )
            hits.append(
                HitText(
                    kind="image",
                    page=pno1,
                    line=None,
                    snippet_html="",
                    snapshot_url=url,
                )
            )
        return hits
    finally:
        try:
            doc.close()
        except Exception:
            pass


def _search_impl(
    q: str,
    mode: str,
    doc_id: str | None,
    all_hits: bool,
    max_hits_per_doc: int,
    context_lines: int,
) -> SearchOut:
    if not q.strip():
        return SearchOut(ok=True, q=q, results=[])

    terms = [t for t in ANALYZER.tokenize(q) if t]

    # Collect hits per doc (line numbers).
    if mode == "boolean":
        doc_hits = INDEX.search_lines_all(q, doc_id)
    else:
        doc_hits = INDEX.search_lines_any(q, doc_id)

    results: list[SearchDocResult] = []

    for did, line_nos in doc_hits.items():
        is_pdf = (_DOC_PATHS.get(did, "").lower().endswith(".pdf"))
        if is_pdf:
            # dedupe to pages; each page -> one image hit
            pages = _pages_from_lines(did, line_nos)
            total = len(pages)
            limit = total if all_hits else min(max_hits_per_doc, total)
            show_pages = pages[:limit]
            hits = _build_pdf_hits_for_pages(did, show_pages, terms)
            res = SearchDocResult(doc_id=did, hits=hits, total_hits=total)
            if total > limit:
                res.has_more = True
            results.append(res)
        else:
            # non-PDF: keep textual snippets
            total = len(line_nos)
            limit = total if all_hits else min(max_hits_per_doc, total)
            show_lines = line_nos[:limit]
            hits = [
                make_text_snippet(did, ln, context_lines, terms) for ln in show_lines
            ]
            res = SearchDocResult(doc_id=did, hits=hits, total_hits=total)
            if total > limit:
                res.has_more = True
            results.append(res)

    return SearchOut(ok=True, q=q, results=results)


@app.get("/search", response_model=SearchOut)
def search_get(
    q: str = Query(..., description="Search query"),
    mode: str = Query("default", pattern="^(default|boolean)$"),
    doc_id: str | None = Query(None),
    all_hits: bool = Query(False),
) -> SearchOut:
    """Legacy GET search (kept for compatibility)."""
    return _search_impl(
        q=q,
        mode=mode,
        doc_id=doc_id,
        all_hits=all_hits,
        max_hits_per_doc=DEFAULT_MAX_PAGES_PER_DOC,
        context_lines=1,
    )


class SearchIn(BaseModel):
    q: str
    mode: str = Field(default="default", pattern="^(default|boolean)$")
    doc_id: str | None = None
    all_hits: bool = False
    max_hits_per_doc: int = Field(default=DEFAULT_MAX_PAGES_PER_DOC, ge=1, le=50)
    context_lines: int = Field(default=1, ge=0, le=10)


@app.post("/search", response_model=SearchOut)
def search_post(body: Annotated[SearchIn, Body(...)]) -> SearchOut:
    """POST search with per-doc hit limit and snippet context lines."""
    return _search_impl(
        q=body.q,
        mode=body.mode,
        doc_id=body.doc_id,
        all_hits=body.all_hits,
        max_hits_per_doc=body.max_hits_per_doc,
        context_lines=body.context_lines,
    )


# =========================
# Page snapshot (PDF -> PNG)
# =========================


@app.get("/page-snapshot")
def page_snapshot(
    doc_id: str = Query(...),
    page: int = Query(..., ge=1),
    x0: float | None = None,
    y0: float | None = None,
    x1: float | None = None,
    y1: float | None = None,
    crop: int = Query(0, description="1 to crop to bbox; else draw rectangle on full page"),
) -> StreamingResponse:
    """
    Render a PDF page (or a highlighted box if bbox provided) to PNG.
    - If crop=1 and bbox is provided, return the cropped region with a yellow border.
    - Else return the whole page and draw a yellow rectangle if bbox is provided.
    """
    path = _DOC_PATHS.get(doc_id)
    if not path or not path.lower().endswith(".pdf"):
        raise HTTPException(404, "PDF not found for snapshot.")

    try:
        doc = fitz.open(path)
        pno = page - 1
        if pno < 0 or pno >= doc.page_count:
            raise HTTPException(400, "Invalid page.")
        pg = doc[pno]

        pix = pg.get_pixmap(matrix=fitz.Matrix(RENDER_SCALE, RENDER_SCALE))
        img_bytes = pix.tobytes("png")

        from PIL import Image, ImageDraw  # lazy import inside function
        img = Image.open(io.BytesIO(img_bytes))

        if None not in (x0, y0, x1, y1):
            # map from page coords to pixel coords
            rx, ry = pg.rect.width, pg.rect.height
            sx = img.width / rx
            sy = img.height / ry
            box_px = (int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy))

            if crop:
                # crop region and draw a thin border for context
                crop_img = img.crop(box_px)
                draw = ImageDraw.Draw(crop_img)
                w = max(2, int(2 * (RENDER_SCALE / 2)))
                draw.rectangle(
                    (1, 1, crop_img.width - 2, crop_img.height - 2),
                    outline=(255, 204, 0),
                    width=w,
                )
                buf = io.BytesIO()
                crop_img.save(buf, format="PNG")
                buf.seek(0)
                return StreamingResponse(buf, media_type="image/png")
            else:
                draw = ImageDraw.Draw(img)
                draw.rectangle(box_px, outline=(255, 204, 0), width=4)

        buf2 = io.BytesIO()
        img.save(buf2, format="PNG")
        buf2.seek(0)
        return StreamingResponse(buf2, media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"render failed: {e}") from e


# =========================
# Status
# =========================


@app.get("/", response_class=JSONResponse)
def root_status() -> dict[str, int | str]:
    return {"app": "EviSearch-Py", "docs": INDEX.docs()}


# =========================
# Inline UI (style preserved)
# =========================


@app.get("/ui", response_class=HTMLResponse)
def ui_page() -> HTMLResponse:
    # Preserve dark theme and rounded style; enlarge drop zone; add robust DnD and zone-click.
    return HTMLResponse(
        """
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>EviSearch-Py</title>
<style>
:root{
  --bg:#0b0f18; --fg:#e7eaf2; --muted:#9aa3b2; --card:#131a27;
  --accent:#2ea043; --mark:#fff59d; --line:#2b3a55; --danger:#e94d5f;
}
html,body{background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,Segoe UI}
.wrap{max-width:960px;margin:32px auto;padding:0 16px}
h1{font-size:28px;margin:0 0 6px}
.sub{color:var(--muted);margin:0 0 20px}
.zone{
  border:2px dashed #30405c;border-radius:14px;background:var(--card);
  display:flex;gap:12px;align-items:center;flex-wrap:wrap;
  padding:24px;         /* larger padding */
  min-height:220px;     /* MUCH bigger drop target */
  transition:outline .15s ease, box-shadow .15s ease;
}
.zone.drag{outline:2px solid #4b6cb7;box-shadow:0 0 0 4px rgba(75,108,183,.15) inset}
.btn{background:#1f2937;color:var(--fg);border:1px solid var(--line);border-radius:10px;
  padding:8px 12px;cursor:pointer}
.btn.primary{background:var(--accent);border-color:var(--accent);color:#08110a}
.btn.danger{background:transparent;border:none;color:var(--danger);font-size:20px;line-height:1}
.btn.danger:hover{color:#ff6b7a}
.row{display:flex;align-items:center;gap:8px}
.file{display:grid;grid-template-columns:1fr 120px 90px 32px;gap:10px;
  padding:10px;border:1px solid var(--line);border-radius:10px;background:#0f1625}
.file-name{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
progress{width:120px;height:10px}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;border:1px solid var(--line);
  color:var(--muted);font-size:12px;margin-left:6px}
.input{flex:1;background:#0e1420;border:1px solid var(--line);color:var(--fg);
  padding:8px 10px;border-radius:10px}
.card{background:#0e1420;border:1px solid #20304e;border-radius:14px;padding:12px 14px;margin:12px 0}
.doc{font-weight:600;margin-bottom:4px; display:flex; align-items:center; gap:8px}
.meta{color:var(--muted)}
.hit{margin:12px 0;padding:12px;background:#0d1320;border-radius:10px}
.hit .meta{color:var(--muted);font-size:14px;margin-bottom:8px}
mark{background:var(--mark);color:#222}
img.thumb{max-width:420px;border:1px solid var(--line);border-radius:8px;cursor:zoom-in}
.modal{position:fixed;inset:0;background:rgba(0,0,0,.88);display:none;z-index:1000}
.modal.active{display:flex;align-items:center;justify-content:center}
.modal-content{max-width:90vw;max-height:90vh;position:relative;overflow:hidden}
.modal-img{user-select:none;cursor:grab;transform-origin:center center;touch-action:none}
.modal-close{position:absolute;top:14px;right:18px;background:transparent;border:none;
  color:#eee;font-size:28px;cursor:pointer}
.bar{display:flex;gap:10px;margin:10px 0}
.muted{color:var(--muted)}
.toggle{display:inline-flex;align-items:center;gap:6px;margin-left:auto}
</style>
</head>
<body>
<div class="wrap">
  <h1>EviSearch-Py</h1>
  <div class="sub">Multiple-file upload, multi-hit search (PDF returns image crops)</div>

  <div class="zone" id="zone">
    <button class="btn" id="pickBtn">Pick files</button>
    <input id="picker" type="file" multiple style="display:none"/>
    <button class="btn" id="clearAll">Clear All</button>
    <span id="state" class="badge">docs: --</span>
  </div>

  <div id="files"></div>

  <div class="bar">
    <input id="q" class="input" placeholder="Search phrase..."/>
    <button class="btn primary" id="go">Search</button>
  </div>

  <div id="results"></div>
</div>

<!-- Lightbox -->
<div class="modal" id="modal" aria-hidden="true">
  <div class="modal-content" id="modalContent">
    <button class="modal-close" id="modalClose" title="Close (Esc)">×</button>
    <img id="modalImg" class="modal-img" alt="preview"/>
  </div>
</div>

<script>
const zone = document.getElementById('zone');
const pickBtn = document.getElementById('pickBtn');
const picker = document.getElementById('picker');
const filesBox = document.getElementById('files');
const clearAllBtn = document.getElementById('clearAll');
const stateEl = document.getElementById('state');
const q = document.getElementById('q');
const go = document.getElementById('go');
const results = document.getElementById('results');

function refreshState() {
  fetch('/', {cache:'no-store'}).then(r => r.json()).then(j => {
    stateEl.textContent = `docs: ${j.docs}`;
  });
}

function createRow(name) {
  const row = document.createElement('div');
  row.className = 'file';
  row.innerHTML = `
    <div class="file-name">${name}</div>
    <progress value="0" max="100"></progress>
    <span class="muted">waiting</span>
    <button class="btn danger" title="Cancel / Remove">×</button>
  `;
  return row;
}

function uploadOne(file) {
  const row = createRow(file.name);
  filesBox.appendChild(row);
  const prog = row.querySelector('progress');
  const status = row.querySelector('.muted');
  const btn = row.querySelector('.btn.danger');

  const form = new FormData();
  form.append('files', file);

  const xhr = new XMLHttpRequest();
  row._xhr = xhr;

  xhr.open('POST', '/index-files', true);

  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) {
      prog.value = Math.round(100 * e.loaded / e.total);
      status.textContent = `uploading ${prog.value}%`;
    }
  };

  xhr.onreadystatechange = () => {
    if (xhr.readyState === 4) {
      if (xhr.status >= 200 && xhr.status < 300) {
        status.textContent = 'indexed';
        prog.value = 100;
        refreshState();
      } else if (xhr.status === 0) {
        status.textContent = 'canceled';
      } else {
        status.textContent = 'error';
      }
    }
  };

  btn.onclick = () => {
    if (row._xhr && row._xhr.readyState !== 4) {
      row._xhr.abort(); // true cancel
      status.textContent = 'canceled';
    } else {
      // after upload: try to remove from server
      fetch(`/files/${encodeURIComponent(file.name)}`, {method:'DELETE'})
        .then(() => {
          row.remove();
          refreshState();
        });
    }
  };

  xhr.send(form);
}

function handleFiles(fileList) {
  [...fileList].forEach(uploadOne);
}

// ---- Robust DnD (window + zone), with drag counter ----
let dragCounter = 0;
function prevent(e){ e.preventDefault(); e.stopPropagation(); }

['dragenter','dragover','dragleave','drop'].forEach(ev => {
  window.addEventListener(ev, prevent);
});
window.addEventListener('dragenter', () => { dragCounter++; zone.classList.add('drag'); });
window.addEventListener('dragleave', () => { dragCounter = Math.max(0, dragCounter-1); if (!dragCounter) zone.classList.remove('drag'); });
window.addEventListener('drop', e => {
  dragCounter = 0; zone.classList.remove('drag');
  const files = e.dataTransfer && e.dataTransfer.files;
  if (files && files.length) handleFiles(files);
});

// Zone-specific handlers (some browsers require a real drop target)
['dragenter','dragover','dragleave','drop'].forEach(ev => {
  zone.addEventListener(ev, prevent);
});
zone.addEventListener('dragenter', () => zone.classList.add('drag'));
zone.addEventListener('dragleave', () => zone.classList.remove('drag'));
zone.addEventListener('drop', e => {
  zone.classList.remove('drag');
  const files = e.dataTransfer && e.dataTransfer.files;
  if (files && files.length) handleFiles(files);
});

// Zone click opens file picker (except when clicking buttons)
zone.addEventListener('click', (e) => {
  const t = e.target;
  if (t.closest('button') || t.tagName === 'INPUT') return;
  picker.click();
});

pickBtn.onclick = () => picker.click();
picker.onchange = () => picker.files && handleFiles(picker.files);

clearAllBtn.onclick = () => {
  if (!confirm('Clear all files and index?')) return;
  fetch('/files', {method:'DELETE'}).then(() => {
    filesBox.innerHTML = '';
    refreshState();
  });
};

function openModal(src) {
  const modal = document.getElementById('modal');
  const img = document.getElementById('modalImg');
  img.src = src;
  img.style.transform = 'translate(0px,0px) scale(1)';
  img.dataset.scale = '1';
  img.dataset.x = '0';
  img.dataset.y = '0';
  modal.classList.add('active');
  modal.setAttribute('aria-hidden', 'false');
}

function closeModal() {
  const modal = document.getElementById('modal');
  modal.classList.remove('active');
  modal.setAttribute('aria-hidden', 'true');
}

document.getElementById('modalClose').onclick = closeModal;
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeModal();
});

// Zoom & Pan
(function(){
  const img = document.getElementById('modalImg');
  let dragging = false, sx = 0, sy = 0, ox = 0, oy = 0;

  function apply() {
    img.style.transform =
      `translate(${ox}px,${oy}px) scale(${parseFloat(img.dataset.scale||'1')})`;
  }
  img.addEventListener('wheel', e => {
    e.preventDefault();
    let s = parseFloat(img.dataset.scale || '1');
    s *= (e.deltaY < 0) ? 1.1 : 0.9;
    s = Math.min(8, Math.max(0.2, s));
    img.dataset.scale = String(s);
    apply();
  }, {passive:false});

  img.addEventListener('mousedown', e => {
    dragging = true; img.style.cursor = 'grabbing';
    sx = e.clientX; sy = e.clientY;
  });
  window.addEventListener('mouseup', () => {
    dragging = false; img.style.cursor = 'grab';
  });
  window.addEventListener('mousemove', e => {
    if (!dragging) return;
    ox += (e.clientX - sx);
    oy += (e.clientY - sy);
    sx = e.clientX; sy = e.clientY;
    apply();
  });
})();

// ---- Search (per-doc Show all hits) ----

function renderDocCard(result) {
  // result: SearchDocResult
  const card = document.createElement('div');
  card.className = 'card';
  card.dataset.docId = result.doc_id;

  const head = document.createElement('div');
  head.className = 'doc';
  head.innerHTML = `
    <span>${result.doc_id}</span>
    <span class="badge">${result.total_hits} hit${result.total_hits>1?'s':''}</span>
  `;
  const toggle = document.createElement('label');
  toggle.className = 'toggle';
  toggle.innerHTML = `
    <input type="checkbox" ${result.has_more ? '' : 'checked'} />
    <span class="muted">Show all hits</span>
  `;
  const cb = toggle.querySelector('input');
  cb.checked = !result.has_more; // if no more -> treated as all shown
  cb.onchange = async () => {
    const wantAll = cb.checked;
    const payload = {
      q: window.__last_q || '',
      mode: 'default',
      doc_id: result.doc_id,
      all_hits: wantAll,
      max_hits_per_doc: ${DEFAULT_MAX_PAGES_PER_DOC},
      context_lines: 1
    };
    const r = await fetch('/search', {
      method: 'POST',
      headers: {'content-type': 'application/json'},
      body: JSON.stringify(payload)
    });
    const j = await r.json();
    if (j && j.results && j.results.length) {
      const fresh = j.results[0];
      const newCard = renderDocCard(fresh);
      card.replaceWith(newCard);
    }
  };
  head.appendChild(toggle);
  card.appendChild(head);

  for (const h of result.hits) {
    const div = document.createElement('div');
    div.className = 'hit';

    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.textContent = h.page ? `Page ${h.page}` : '-';
    div.appendChild(meta);

    const body = document.createElement('div');
    if (h.snapshot_url) {
      const img = new Image();
      img.className = 'thumb';
      img.src = h.snapshot_url;
      img.alt = 'snapshot';
      img.onclick = () => openModal(h.snapshot_url);
      body.appendChild(img);
    } else if (h.snippet_html) {
      // non-PDF textual fallback
      const box = document.createElement('div');
      box.innerHTML = h.snippet_html;
      body.appendChild(box);
    }
    div.appendChild(body);
    card.appendChild(div);
  }
  return card;
}

function renderResults(payload) {
  const arr = payload.results || [];
  results.innerHTML = '';
  if (!arr.length) {
    results.innerHTML = '<div class="muted">No results</div>';
    return;
  }
  for (const r of arr) {
    results.appendChild(renderDocCard(r));
  }
}

go.onclick = async () => {
  const body = {
    q: q.value || '',
    mode: 'default',
    all_hits: false, // initial: show first N pages per doc
    max_hits_per_doc: ${DEFAULT_MAX_PAGES_PER_DOC},
    context_lines: 1
  };
  window.__last_q = body.q;
  const r = await fetch('/search', {
    method: 'POST',
    headers: {'content-type': 'application/json'},
    body: JSON.stringify(body),
  });
  const j = await r.json();
  renderResults(j);
};

refreshState();
</script>
</body>
</html>
        """
    )
