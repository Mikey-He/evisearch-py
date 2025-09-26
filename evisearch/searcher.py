from __future__ import annotations

import heapq
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass

from .analyzer import Analyzer
from .indexer import InvertedIndex


# =========================
# Boolean query AST
# =========================
@dataclass(frozen=True)
class Node:
    pass


@dataclass(frozen=True)
class Term(Node):
    value: str


@dataclass(frozen=True)
class Not(Node):
    child: Node


@dataclass(frozen=True)
class And(Node):
    left: Node
    right: Node


@dataclass(frozen=True)
class Or(Node):
    left: Node
    right: Node


# =========================
# Parser (NOT > AND > OR; parentheses)
# =========================
_TOKEN = re.compile(r'\(|\)|AND|OR|NOT|"[^"]+"|\'[^\']+\'|[^\s()]+', re.IGNORECASE)


class QueryParser:
    """
    Very small boolean parser with precedence:
        NOT > AND > OR
    Supports parentheses. Terms are normalized via Analyzer.normalize().
    """

    def __init__(self, analyzer: Analyzer) -> None:
        self.analyzer = analyzer
        self._tokens: list[str] = []
        self._i: int = 0

    def parse(self, q: str) -> Node:
        tokens: list[str] = [t for t in _TOKEN.findall(q) if t.strip()]
        self._i = 0
        self._tokens = tokens
        node = self._parse_or()
        if self._i != len(self._tokens):
            raise ValueError(f"Unexpected token near: {self._tokens[self._i]!r}")
        return node

    # OR level
    def _parse_or(self) -> Node:
        node = self._parse_and()
        while self._peek_upper() == "OR":
            self._consume()
            node = Or(node, self._parse_and())
        return node

    # AND level
    def _parse_and(self) -> Node:
        node = self._parse_not()
        while self._peek_upper() == "AND":
            self._consume()
            node = And(node, self._parse_not())
        return node

    # NOT level
    def _parse_not(self) -> Node:
        if self._peek_upper() == "NOT":
            self._consume()
            return Not(self._parse_not())
        return self._parse_term()

    # term / ( ... )
    def _parse_term(self) -> Node:
        tok = self._peek()
        if tok is None:
            # empty → empty term (matches nothing)
            return Term("__EMPTY__")
        if tok == "(":
            self._consume()
            node = self._parse_or()
            if self._peek() != ")":
                raise ValueError("Unclosed '('")
            self._consume()
            return node

        self._consume()
        # strip quotes if any
        if (tok.startswith('"') and tok.endswith('"')) or (
            tok.startswith("'") and tok.endswith("'")
        ):
            tok = tok[1:-1]

        # normalize using analyzer; keep_stopwords=False
        norm = self.analyzer.normalize(tok)
        value = norm.strip()
        if not value:
            value = "__EMPTY__"  # will evaluate to ∅
        return Term(value)

    # peek helpers
    def _peek(self) -> str | None:
        return self._tokens[self._i] if self._i < len(self._tokens) else None

    def _peek_upper(self) -> str | None:
        t = self._peek()
        return t.upper() if t is not None else None

    def _consume(self) -> str:
        t: str = self._tokens[self._i]
        self._i += 1
        return t


# =========================
# Boolean helpers (sorted-list merge ops)
# =========================
def _sorted_unique(xs: Iterable[str]) -> list[str]:
    return sorted(set(xs))


def _intersect_sorted(a: list[str], b: list[str]) -> list[str]:
    """Linear-time merge intersection on two sorted lists."""
    i = j = 0
    out: list[str] = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return out


def _union_sorted(a: list[str], b: list[str]) -> list[str]:
    """Linear-time merge union on two sorted lists."""
    i = j = 0
    out: list[str] = []
    while i < len(a) or j < len(b):
        if j >= len(b) or (i < len(a) and a[i] < b[j]):
            if not out or out[-1] != a[i]:
                out.append(a[i])
            i += 1
        elif i >= len(a) or b[j] < a[i]:
            if not out or out[-1] != b[j]:
                out.append(b[j])
            j += 1
        else:  # equal
            if not out or out[-1] != a[i]:
                out.append(a[i])
            i += 1
            j += 1
    return out


def _difference_sorted(a: list[str], b: list[str]) -> list[str]:
    """Linear-time a - b using merge walk."""
    i = j = 0
    out: list[str] = []
    while i < len(a):
        if j >= len(b):
            out.extend(a[i:])
            break
        if a[i] == b[j]:
            i += 1
            j += 1
        elif a[i] < b[j]:
            out.append(a[i])
            i += 1
        else:
            j += 1
    return out


# =========================
# Phrase matcher (exact contiguous tokens)
# =========================
class PhraseMatcher:
    def __init__(self, index: InvertedIndex, analyzer: Analyzer) -> None:
        self.index = index
        self.analyzer = analyzer

    def match(self, phrase: str, *, keep_stopwords: bool = True) -> dict[str, list[int]]:
        terms = self.analyzer.tokenize(phrase, keep_stopwords=keep_stopwords)
        if not terms:
            return {}
        first_post = self.index.get_posting(terms[0])
        if not first_post:
            return {}

        # Candidate docs: intersection of docs containing all terms
        candidate_docs = set(first_post.keys())
        for t in terms[1:]:
            posting = self.index.get_posting(t)
            if not posting:
                return {}
            candidate_docs &= set(posting.keys())
            if not candidate_docs:
                return {}

        hits: dict[str, list[int]] = {}
        for doc in candidate_docs:
            positions_seq = [self.index.get_posting(t)[doc] for t in terms]
            sets_follow = [set(ps) for ps in positions_seq[1:]]
            starts: list[int] = []
            for p0 in positions_seq[0]:
                ok = True
                for i, s in enumerate(sets_follow, start=1):
                    if (p0 + i) not in s:
                        ok = False
                        break
                if ok:
                    starts.append(p0)
            if starts:
                hits[doc] = starts
        return hits


# =========================
# BM25 scorer (Okapi)
# =========================
class BM25Scorer:
    def __init__(self, index: InvertedIndex, k1: float = 1.2, b: float = 0.75) -> None:
        self.index = index
        self.N = max(1, len(index.doc_ids))
        self.avgdl = sum(index.doc_lengths.get(d, 0) for d in index.doc_ids) / float(self.N)
        self.k1 = k1
        self.b = b
        self._idf_cache: dict[str, float] = {}

    def _idf(self, term: str) -> float:
        v = self._idf_cache.get(term)
        if v is not None:
            return v
        df = len(self.index.get_posting(term))
        # log with +1 to keep positive; robust for small corpora
        v = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
        self._idf_cache[term] = v
        return v

    def score(self, doc_id: str, q_terms: list[str]) -> float:
        dl = max(1, self.index.doc_lengths.get(doc_id, 0))
        score = 0.0
        # term frequency in this doc via posting lengths
        for t in q_terms:
            posting = self.index.get_posting(t)
            tf = len(posting.get(doc_id, []))
            if tf == 0:
                continue
            idf = self._idf(t)
            denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
            score += idf * (tf * (self.k1 + 1.0)) / denom
        return score


# =========================
# Searcher (boolean + ranked)
# =========================
class Searcher:
    """Boolean retrieval and BM25 ranking over an InvertedIndex."""

    def __init__(self, index: InvertedIndex, analyzer: Analyzer) -> None:
        self.index = index
        self.analyzer = analyzer
        self.parser = QueryParser(analyzer)
        self._all_docs: list[str] = sorted(self.index.doc_ids)

    # --- boolean ---
    def search_boolean(self, q: str) -> list[str]:
        """Return sorted doc IDs that satisfy the boolean query."""
        ast = self.parser.parse(q)
        return self._eval(ast)

    def _posting_docs(self, term: str) -> list[str]:
        if term == "__EMPTY__":
            return []
        posting = self.index.get_posting(term)
        if not posting:
            return []
        return _sorted_unique(posting.keys())

    def _eval(self, node: Node) -> list[str]:
        if isinstance(node, Term):
            return self._posting_docs(node.value)
        if isinstance(node, Not):
            child = self._eval(node.child)
            return _difference_sorted(self._all_docs, child)
        if isinstance(node, And):
            left = self._eval(node.left)
            right = self._eval(node.right)
            return _intersect_sorted(left, right)
        if isinstance(node, Or):
            left = self._eval(node.left)
            right = self._eval(node.right)
            return _union_sorted(left, right)
        raise TypeError(f"Unknown node: {node!r}")

        # --- ranked (BM25) ---

    def search_ranked(self, q: str, k: int = 10) -> list[tuple[str, float]]:
        terms = self.analyzer.tokenize(q, keep_stopwords=False)
        if not terms:
            return []

        # unqiue terms to avoid redundant posting fetch
        unique_terms = list(dict.fromkeys(terms))  # preserve order, dedup
        postings = [self.index.get_posting(t) for t in unique_terms]
        nonempty_postings = [p for p in postings if p]

        # no terms exist in any doc
        if not nonempty_postings:
            return []

        # satisfying docs = intersection of all non-empty term postings
        candidates: set[str] = set(nonempty_postings[0].keys())
        for p in nonempty_postings[1:]:
            candidates &= set(p.keys())
            if not candidates:
                return []

        scorer = BM25Scorer(self.index)
        heap: list[tuple[float, str]] = []  # (neg_score, doc_id)
        for d in candidates:
            s = scorer.score(d, terms)
            if s <= 0.0:
                continue
            if len(heap) < k:
                heapq.heappush(heap, (-s, d))
            else:
                if -s > heap[0][0]:
                    heapq.heapreplace(heap, (-s, d))

        out = [(d, -neg) for (neg, d) in heap]
        out.sort(key=lambda x: x[1], reverse=True)
        return out
