from __future__ import annotations

import io
import os
import re
import threading
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import ocrmypdf
import requests  # <-- *** 1. 新增导入 ***
from requests.auth import HTTPBasicAuth  # <-- *** 2. 新增导入 ***

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

                if is_pdf_scanned(path):
                    # 1. 只有在文件是扫描件时，才运行 ocrmypdf
                    print(f"INFO: [Worker] PDF {path.name} seems scanned. Running OCR (ocrmypdf)...")
                    try:
                        ocrmypdf.ocr(
                            path,
                            path,
                            redo_ocr=True,
                            output_type="pdf",
                            language='eng',
                            jobs=1,
                            progress_bar=False,
                        )
                        print(f"INFO: [Worker] OCR complete for {path.name}.")
                    except Exception as ocr_error:
                        print(f"ERROR: [Worker] ocrmypdf failed for {path.name}: {ocr_error}")
                        continue # 跳到下一个文件
                else:
                    # 2. 如果是文本 PDF，打印日志并跳过 OCR
                    print(f"INFO: [Worker] PDF {path.name} is text-based. Skipping OCR.")

                texts: list[str] = []
                pos = 0
                with fitz.open(str(path)) as doc:
                    for i, page in enumerate(doc, start=1):
                        txt = page.get_text() or ""
                        texts.append(txt)
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
    
    # *** 3. 修改：将结果存储在变量中 ***
    result = {} 
    
    try:
        vocab_size = _rebuild_index_and_save()
        result = {"status": "success", "vocab_size": vocab_size} # <-- 修改
    except Exception as e:
        print(f"ERROR: [Worker] Re-index failed: {e}")
        result = {"status": "error", "error": str(e)} # <-- 修改
    finally:
        _INDEX_WRITE_LOCK.release()
        print("INFO: [Worker] Lock released.")

    # *** 4. 新增：如果成功，通知 API 进程重载缓存 ***
    if result.get("status") == "success":
        try:
            print("INFO: [Worker] Index built. Notifying API server to reload cache...")
            # 从环境变量获取认证信息
            user = os.getenv("BASIC_USER") or ""
            pwd = os.getenv("BASIC_PASS") or ""
            
            # Gunicorn 运行在 10000 端口 (根据你的 start.sh)
            # 我们从 worker 内部调用 API
            response = requests.post(
                "http://localhost:10000/reload-index",
                auth=HTTPBasicAuth(user, pwd) if user and pwd else None,
                timeout=5
            )
            if response.status_code == 202:
                print("INFO: [Worker] API server successfully triggered for cache reload.")
            else:
                print(f"WARN: [Worker] API server returned {response.status_code} on cache reload.")
        except Exception as e:
            print(f"ERROR: [Worker] Failed to notify API server to reload cache: {e}")

    return result # <-- 修改


def is_pdf_scanned(path: Path) -> bool:
    """
    Check if a PDF is scanned (image-based) by sampling the first few pages.
    Returns True if it's likely scanned, False if it's text-based.
    """
    try:
        with fitz.open(str(path)) as doc:
            # 检查前 5 页（或所有页面，如果少于 5 页）
            num_pages_to_check = min(len(doc), 5)
            if num_pages_to_check == 0:
                return False  # 空 PDF

            total_text_length = 0
            for i in range(num_pages_to_check):
                page = doc.load_page(i)
                total_text_length += len(page.get_text())

            # 如果平均每页的字符数少于 100，我们*假设*它是扫描件
            avg_chars = total_text_length / num_pages_to_check
            return avg_chars < 100
    except Exception as e:
        print(f"WARN: [Worker] is_pdf_scanned check failed for {path.name}: {e}")
        return False # 出错时，默认为文本 PDF
