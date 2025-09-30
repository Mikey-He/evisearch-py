
# EviSearch-Py
A simple, evidence-level full-text search engine written in Python.

## Features
-   **Fast, In-Memory Indexing**: Builds an inverted index for quick searching.
-   **Multiple File Types**: Supports PDF and simple TXT files.
-   **OCR Support**: Automatically extracts text from scanned PDFs using Tesseract.
-   **Auto Search Modes**:
    -   Ranked search (BM25)
    -   Boolean search (`AND`, `OR`, `NOT`)
    -   Exact phrase search
-   **Rich Search Results**: Generates highlighted text snippets and page snapshot images.
-   **Dual Interfaces**:
    -   A modern web UI built with FastAPI.
    -   A full-featured Command-Line Interface (CLI).
-   **Performance Benchmarking**: Includes scripts to measure QPS, latency, and index size.
