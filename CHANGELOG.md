# Changelog

## [1.0.0] - 2025-09-25
### Added
- Positional inverted index with line mapping (evidence line + 1-based line number).
- Boolean (AND/OR/NOT), exact phrase, BM25 ranked retrieval.
- CLI (`index`, `search`) and minimal REST API (`/index`, `/search`).
- Benchmarks (QPS/P95/index size) and quick charts.
- Tests with coverage gate (>=70%), `mypy --strict`, `ruff/black`.

### Changed
- Unified tool config in `pyproject.toml` (ruff/black/mypy/pytest/coverage).

### Fixed
- Phrase matching alignment when indexing with/without stopwords.
