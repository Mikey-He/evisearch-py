from __future__ import annotations

from dataclasses import dataclass

from .indexer import InvertedIndex


@dataclass(frozen=True)
class Context:
    """A readable evidence block with location info."""
    snippet: str
    line: int           # 1-based line index in the document
    page: int | None    # 1-based page index if available (None otherwise)


class Highlighter:
    """
    Builds readable evidence from token hit positions:
      - build_snippet: single-line evidence (compatibility)
      - build_context: paragraph or ±window lines context
    Requires the index to expose:
      - doc_lines[doc_id]: list[str]
      - line_of_pos[doc_id]: list[int]  (maps token-position -> 0-based line)
    Optionally:
      - page_map[doc_id]: list[tuple[int, int]]  (sorted by token-pos; values are (pos0, page1))
    """

    def __init__(self, index: InvertedIndex) -> None:
        self.index = index

    def _page_of_pos(self, doc_id: str, pos: int) -> int | None:
        """
        If page_map is available, return page number for the given token position.
        page_map format: [(pos0, page1), (posN, pageM), ...], sorted by pos0.
        """
        page_map = getattr(self.index, "page_map", {}).get(doc_id)
        if not page_map:
            return None
        lo, hi = 0, len(page_map) - 1
        ans: int | None = None
        while lo <= hi:
            mid = (lo + hi) // 2
            p0, pg = page_map[mid]
            if p0 <= pos:
                ans = pg
                lo = mid + 1
            else:
                hi = mid - 1
        return ans

    def build_snippet(self, doc_id: str, start_pos: int | None) -> tuple[str, int]:
        """
        Return a single evidence line and its 1-based line number.
        """
        if start_pos is None:
            return ("", 0)
        lines = self.index.doc_lines[doc_id]
        line_of_pos = self.index.line_of_pos[doc_id]
        line0 = line_of_pos[start_pos] if start_pos < len(line_of_pos) else 0
        return (lines[line0].strip(), line0 + 1)

    def build_context(
        self,
        doc_id: str,
        start_pos: int,
        *,
        mode: str = "paragraph",  # "paragraph" | "window"
        window: int = 2,          # only used when mode == "window"
    ) -> Context:
        """
        Build a multi-line snippet around the given token position.
        - paragraph: expand to surrounding non-empty lines
        - window:    include ±window lines around the hit line
        """
        lines = self.index.doc_lines[doc_id]
        line_of_pos = self.index.line_of_pos[doc_id]
        line0 = line_of_pos[start_pos] if start_pos < len(line_of_pos) else 0

        if mode == "window":
            lo = max(0, line0 - window)
            hi = min(len(lines) - 1, line0 + window)
            snippet = "\n".join(s.strip() for s in lines[lo : hi + 1])
        else:
            # paragraph mode: expand to blank-line boundaries
            lo = line0
            while lo > 0 and lines[lo - 1].strip() != "":
                lo -= 1
            hi = line0
            while hi + 1 < len(lines) and lines[hi + 1].strip() != "":
                hi += 1
            snippet = "\n".join(s.strip() for s in lines[lo : hi + 1])

        page = self._page_of_pos(doc_id, start_pos)
        return Context(snippet=snippet, line=line0 + 1, page=page)
