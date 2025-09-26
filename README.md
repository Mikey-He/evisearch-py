
# EviSearch-Py

Evidence-level mini search engine. Returns the **original line** and **1-based line number**.  
Supports **Boolean**, **exact phrase**, and **BM25 ranked** search. Ships with CLI, minimal REST API, and small benchmark scripts.

---

## Requirements
- Python 3.11+ (tested on 3.12)
- Windows/macOS/Linux
- Dependencies installed via `pip` (see commands below)

---

## Install
```bash
# inside your project venv
pip install -U pip
pip install fastapi uvicorn matplotlib pytest pytest-cov mypy ruff black
