from __future__ import annotations

import io
import os
import re
import threading
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

import ocrmypdf

from .analyzer import Analyzer
from .celery_app import celery_app  # import the Celery app instance
from .indexer import IndexWriter
from .storage import save_index     # import save_index function

# variables shared by tasks
DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_FILE = DATA_DIR / "index.json"

# lock to prevent concurrent index writes
_INDEX_WRITE_LOCK = threading.Lock()

# shared analyzer instance
_ANALYZER = Analyzer()


# moved from api.py

def _safe_doc_id(name: str) -> str:
    """Create safe document ID from filename."""
    base = os.path.basename(name or "doc")
    base = re.sub(r'\s+', '_', base)
    base = base.replace('/', '_').replace('\\', '_').replace('\0', '_')
    return os.path.splitext(base)[0]


def _rebuild_index_and_save() -> int:
    """Rebuild the search index from documents in UPLOAD_DIR and save to INDEX_FILE."""
    writer = IndexWriter(_ANALYZER, index_stopwords=True)
    
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    doc_paths = list(UPLOAD_DIR.glob("*.pdf")) + list(UPLOAD_DIR.glob("*.txt"))
    if not doc_paths:
        print("INFO: [Worker] No documents found in UPLOAD_DIR. Saving empty index.")

    print(f"INFO: [Worker] Rebuilding index from {len(doc_paths)} documents...")

    for path in doc_paths:
        doc_id = _safe_doc_id(path.name)
        if not doc_id:
            print(f"WARN: [Worker] Skipping file with invalid name: {path.name}")
            continue
            
        try:
            text = ""
            page_map = []
            
            if path.suffix.lower() == ".pdf":
                print(f"INFO: [Worker] Running OCR (ocrmypdf) on {path.name}...")
                try:
                    # 这会运行 OCR 并*覆盖*原始文件，
                    # 创造一个“三明治 PDF”
                    ocrmypdf.ocr(
                        path,             # input_file
                        path,             # output_file (覆盖它自己)
                        skip_text=True, 
                        output_type="pdf",
                        language='eng',   # 假设是英语
                        jobs=1,           # 限制为1个核心，以免 Celery worker 过载
                        progress_bar=False,)
                    print(f"INFO: [Worker] OCR complete for {path.name}.")
                except Exception as ocr_error:
                    # 如果 ocrmypdf 失败 (例如，PDF 损坏或受密码保护)
                    print(f"ERROR: [Worker] ocrmypdf failed for {path.name}: {ocr_error}")
                    # 我们仍然尝试用 fitz 提取文本（以防万一）
                    pass
                
                texts: list[str] = []
                pos = 0
                with fitz.open(str(path)) as doc:
                    for i, page in enumerate(doc, start=1):
                        txt = page.get_text() or ""
                        texts.append(txt)
                        # 我们仍然需要构建 page_map 以便索引器工作
                        toks = list(_ANALYZER.iter_tokens(txt, keep_stopwords=True))
                        page_map.append((pos, i))
                        pos += len(toks)
                text = "\n\n".join(texts)
                
            else: # .txt
                text = path.read_text(encoding="utf-8", errors="ignore")
                toks = list(_ANALYZER.iter_tokens(text, keep_stopwords=True))
                page_map = [(0, 1)] if toks else []
            
            if text:
                writer.add(doc_id, text, page_map=page_map)
                print(f"INFO: [Worker] Added '{doc_id}' to index.")
            else:
                print(f"WARN: [Worker] No text extracted for {doc_id} ({path.name})")
                
        except Exception as e:
            print(f"ERROR: [Worker] Failed to process {doc_id} ({path.name}): {e}")

    index = writer.commit()
    try:
        save_index(index, INDEX_FILE)
        print(f"INFO: [Worker] Index successfully saved to {INDEX_FILE} (Vocab: {index.vocabulary_size()})")
        return index.vocabulary_size()
    except Exception as e:
        print(f"ERROR: [Worker] Failed to save index to {INDEX_FILE}: {e}")
        return 0


# --- Celery ---

@celery_app.task(name="evisearch.tasks.trigger_reindex")
def trigger_reindex() -> dict[str, Any]:

    have_lock = _INDEX_WRITE_LOCK.acquire(timeout=10)
    
    if not have_lock:
        print("INFO: [Worker] Skipping re-index: another index task is already running.")
        return {"status": "skipped", "message": "Index build already in progress."}

    print("INFO: [Worker] Lock acquired. Starting full re-index...")
    try:
        vocab_size = _rebuild_index_and_save()
        return {"status": "success", "vocab_size": vocab_size}
    except Exception as e:
        print(f"ERROR: [Worker] Re-index failed: {e}")
        return {"status": "error", "error": str(e)}
    finally:
        _INDEX_WRITE_LOCK.release()
        print("INFO: [Worker] Lock released.")