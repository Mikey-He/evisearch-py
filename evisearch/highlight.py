from __future__ import annotations

from evisearch.indexer import InvertedIndex


class Highlighter:
    """
    Minimal snippet builder: returns the original line text and its 1-based line number.
    Later we can extend to page mapping (for PDFs) and inline term highlighting.
    """

    def __init__(self, index: InvertedIndex) -> None:
        self.index = index

    def build_snippet(self, doc_id: str, start_pos: int) -> tuple[str, int]:
        """
        Parameters
        ----------
        doc_id : str
            Document id.
        start_pos : int
            Phrase start token position in the document's global token stream.

        Returns
        -------
        (snippet, line_no) : tuple[str, int]
            Original line text and its 1-based line number.
        """
        pos2line = self.index.pos_to_line.get(doc_id)
        if not pos2line:
            return ("", 0)
        line_no = pos2line.get(start_pos, 0)
        if line_no <= 0:
            return ("", 0)

        lines = self.index.doc_lines.get(doc_id, [])
        if 1 <= line_no <= len(lines):
            return (lines[line_no - 1], line_no)
        return ("", 0)
