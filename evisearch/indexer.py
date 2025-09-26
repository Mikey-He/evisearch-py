from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from evisearch.analyzer import Analyzer

# -------------------------------
# Data structures
# -------------------------------
Posting = dict[str, list[int]]  # doc_id -> sorted list of positions


@dataclass(slots=True)
class InvertedIndex:
    """In-memory inverted index with positional postings and line mapping."""

    postings: dict[str, Posting] = field(default_factory=lambda: defaultdict(dict))
    doc_lengths: dict[str, int] = field(default_factory=dict)  # length after stopword filtering
    doc_ids: list[str] = field(default_factory=list)

    # --- additions for highlighting ---
    doc_lines: dict[str, list[str]] = field(default_factory=dict)  # original lines
    pos_to_line: dict[str, dict[int, int]] = field(
        default_factory=dict
    )  # token pos -> 1-based line no
    line_token_offsets: dict[str, list[int]] = field(
        default_factory=dict
    )  # per line, first token's global pos

    def get_posting(self, term: str) -> Posting:
        return self.postings.get(term, {})

    def vocabulary_size(self) -> int:
        return len(self.postings)

    def __contains__(self, term: str) -> bool:  # convenience
        return term in self.postings


class IndexWriter:
    """
    Build an in-memory inverted index.

    Parameters
    ----------
    analyzer : Analyzer
        Tokenizer/normalizer to use.
    index_stopwords : bool
        If True, index positions for *all* tokens (including stopwords).
        If False, index with stopwords removed (smaller index). Phrase queries
        across stopwords will be harder in that mode.
    """

    def __init__(self, analyzer: Analyzer, *, index_stopwords: bool = True) -> None:
        self.analyzer = analyzer
        self.index_stopwords = index_stopwords
        self._postings: dict[str, Posting] = defaultdict(lambda: defaultdict(list))
        self._doc_lengths: dict[str, int] = {}
        self._doc_ids: list[str] = []

        # for highlighting
        self._doc_lines: dict[str, list[str]] = {}
        self._pos_to_line: dict[str, dict[int, int]] = {}
        self._line_token_offsets: dict[str, list[int]] = {}

    def add(self, doc_id: str, text: str) -> None:
        """
        Tokenize `text` and add positional postings for `doc_id`.

        - Positions are token indices *in the emitted stream* (respecting index_stopwords).
        - Doc length is computed with stopwords removed (useful for BM25 later).
        - Also compute mapping: token-position -> line number,
          so we can highlight the original line.
        """
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError("doc_id must be a non-empty string")

        # 1) Build postings from the global token stream (for correctness)
        pairs = self.analyzer.tokenize_with_positions(text, keep_stopwords=self.index_stopwords)
        for tok, pos in pairs:
            self._postings[tok][doc_id].append(pos)

        # Dedup/sort just for this doc (robustness)
        for _tok, by_doc in self._postings.items():
            if doc_id in by_doc and by_doc[doc_id]:
                by_doc[doc_id] = _dedup_sorted(by_doc[doc_id])

        # 2) Doc length for scoring (stopwords removed)
        self._doc_lengths[doc_id] = len(self.analyzer.tokenize(text, keep_stopwords=False))

        # 3) Build line-level metadata (for highlighting)
        lines = text.splitlines()
        self._doc_lines[doc_id] = lines

        pos_to_line: dict[int, int] = {}
        line_offsets: list[int] = []
        pos = 0
        for lineno, line in enumerate(lines, start=1):
            line_offsets.append(pos)
            # Use the same keep_stopwords policy as postings to keep positions aligned
            toks = self.analyzer.tokenize(line, keep_stopwords=self.index_stopwords)
            for _ in toks:
                pos_to_line[pos] = lineno  # map this token position to its original 1-based line no
                pos += 1

        self._pos_to_line[doc_id] = pos_to_line
        self._line_token_offsets[doc_id] = line_offsets

        if doc_id not in self._doc_ids:
            self._doc_ids.append(doc_id)

    def commit(self) -> InvertedIndex:
        """Finalize and return an immutable-ish view of the index."""
        # Ensure every posting list is sorted (safety for multi-add use cases)
        for _tok, by_doc in self._postings.items():
            for did, pos_list in by_doc.items():
                by_doc[did] = _dedup_sorted(pos_list)

        idx = InvertedIndex(
            postings={t: dict(d) for t, d in self._postings.items()},
            doc_lengths=dict(self._doc_lengths),
            doc_ids=list(self._doc_ids),
            doc_lines=dict(self._doc_lines),
            pos_to_line=dict(self._pos_to_line),
            line_token_offsets=dict(self._line_token_offsets),
        )
        return idx


# -------------------------------
# Helpers
# -------------------------------
def _dedup_sorted(nums: list[int]) -> list[int]:
    if not nums:
        return []
    nums.sort()
    out: list[int] = [nums[0]]
    for x in nums[1:]:
        if x != out[-1]:
            out.append(x)
    return out
