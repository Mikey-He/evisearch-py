from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .analyzer import Analyzer


@dataclass(frozen=True)
class InvertedIndex:
    """
    Immutable-ish inverted index snapshot.

    postings:
        term -> {doc_id -> [positions...]}
        (positions are 0-based token indices within the document)

    doc_lengths:
        doc_id -> number of **non-stopword** tokens (used by BM25 length norm)

    doc_ids:
        list of doc ids (in insertion order)

    doc_lines:
        doc_id -> list[str]  (original text split into lines, for highlighting/context)

    line_of_pos:
        doc_id -> list[int]  (maps token position -> 0-based line index)

    page_map (optional):
        doc_id -> list[tuple[int, int]]
        Each tuple is (token_pos_offset, page_number), sorted by token_pos_offset.
    """

    postings: dict[str, dict[str, list[int]]]
    doc_lengths: dict[str, int]
    doc_ids: list[str]
    doc_lines: dict[str, list[str]]
    line_of_pos: dict[str, list[int]]
    page_map: dict[str, list[tuple[int, int]]] = field(default_factory=dict)

    def get_posting(self, term: str) -> dict[str, list[int]] | None:
        return self.postings.get(term)

    def vocabulary_size(self) -> int:
        return len(self.postings)

    @property
    def pos_to_line(self) -> dict[str, dict[int, int]]:
        """Compatibility shim: build {doc_id: {pos: line}} from line_of_pos."""
        return {
            doc: {i: ln for i, ln in enumerate(lst)}
            for doc, lst in self.line_of_pos.items()
        }


class IndexWriter:
    """
    Mutable builder; call `commit()` to freeze into an InvertedIndex.

    If `index_stopwords=True`, stopwords are kept in postings/positions
    (useful for phrase matching). **Doc length** always counts **without**
    stopwords for BM25 normalization.
    """

    def __init__(self, analyzer: Analyzer, *, index_stopwords: bool = False) -> None:
        self.analyzer = analyzer
        self.index_stopwords = index_stopwords

        # term -> doc_id -> [positions...]
        self._postings: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._doc_lengths: dict[str, int] = {}
        self._doc_ids: list[str] = []

        # for highlighting/context
        self._doc_lines: dict[str, list[str]] = {}
        self._line_of_pos: dict[str, list[int]] = {}

        # optional: token-pos -> page-number map per doc
        self._page_map: dict[str, list[tuple[int, int]]] = {}

    def add(
        self,
        doc_id: str,
        text: str,
        *,
        page_map: list[tuple[int, int]] | None = None,
    ) -> None:
        """
        Add a single document. `doc_id` must be unique and contain no whitespace.

        If `page_map` is provided, it must be a sorted list of (pos, page).
        """
        if not doc_id or any(c.isspace() for c in doc_id):
            raise ValueError("doc_id must be non-empty and contain no whitespace")
        if doc_id in self._doc_lengths:
            raise ValueError(f"Duplicate doc_id: {doc_id!r}")

        lines = text.splitlines()
        self._doc_lines[doc_id] = lines

        # Build postings + line_of_pos in one pass.
        pos = 0
        line_map: list[int] = []
        by_term = self._postings

        # Doc length counts tokens **without** stopwords regardless of index_stopwords.
        doc_len_no_stop = 0

        for line_idx, line in enumerate(lines):
            # tokens for postings (may include stopwords)
            for tok in self.analyzer.iter_tokens(
                line, keep_stopwords=self.index_stopwords
            ):
                by_term[tok][doc_id].append(pos)
                line_map.append(line_idx)
                pos += 1

            # tokens for doc length (never include stopwords)
            for _tok in self.analyzer.iter_tokens(line, keep_stopwords=False):
                doc_len_no_stop += 1

        self._line_of_pos[doc_id] = line_map
        self._doc_lengths[doc_id] = doc_len_no_stop
        self._doc_ids.append(doc_id)

        if page_map:
            # Ensure sorted and unique by pos
            pm_sorted = sorted(page_map, key=lambda t: t[0])
            self._page_map[doc_id] = pm_sorted

    def commit(self) -> InvertedIndex:
        """Finalize the index: sort & de-dup each posting list, then freeze."""
        for by_doc in self._postings.values():
            for did, pos_list in by_doc.items():
                if not pos_list:
                    continue
                pos_list.sort()
                deduped: list[int] = []
                last = None
                for p in pos_list:
                    if p != last:
                        deduped.append(p)
                        last = p
                by_doc[did] = deduped

        return InvertedIndex(
            postings={t: dict(d) for t, d in self._postings.items()},
            doc_lengths=dict(self._doc_lengths),
            doc_ids=list(self._doc_ids),
            doc_lines=dict(self._doc_lines),
            line_of_pos=dict(self._line_of_pos),
            page_map=dict(self._page_map),
        )
