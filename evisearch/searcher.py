from __future__ import annotations

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
_TOKEN = re.compile(
    r"\(|\)|AND|OR|NOT|\"[^\"]+\"|\'[^\']+\'|[^\s()]+",
    re.IGNORECASE,
)


class QueryParser:
    """
    Very small boolean parser with precedence:
        NOT > AND > OR
    Supports parentheses. Terms are normalized via Analyzer.normalize().
    """

    def __init__(self, analyzer: Analyzer) -> None:
        self.analyzer = analyzer
        self._tokens: list[str] = []
        self._i = 0

    def parse(self, q: str) -> Node:
        tokens = [t for t in _TOKEN.findall(q) if t.strip()]
        self._tokens = tokens
        self._i = 0
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

        norm = self.analyzer.normalize(tok)
        value = norm.strip()
        if not value:
            value = "__EMPTY__"
        return Term(value)

    # peek helpers
    def _peek(self) -> str | None:
        return self._tokens[self._i] if self._i < len(self._tokens) else None

    def _peek_upper(self) -> str | None:
        t = self._peek()
        return t.upper() if t is not None else None

    def _consume(self) -> str:
        t = self._tokens[self._i]
        self._i += 1
        return t


# =========================
# Boolean helpers
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
        else:
            if not out or out[-1] != a[i]:
                out.append(a[i])
            i += 1
            j += 1
    return out


def _difference_sorted(a: list[str], b: list[str]) -> list[str]:
    """Linear-time a - b using merge walk (a,b are sorted unique)."""
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
# Phrase matcher
# =========================
class PhraseMatcher:
    """Exact phrase matching using positional postings."""

    def __init__(self, index: InvertedIndex, analyzer: Analyzer) -> None:
        self.index = index
        self.analyzer = analyzer

    def match(self, phrase: str, *, keep_stopwords: bool = True) -> dict[str, list[int]]:
        """
        Return {doc_id: [start_pos, ...]} for documents containing the phrase.
        When keep_stopwords=True, the query tokens include stopwords to match
        positions exactly against an index built with stopwords.
        """
        tokens = self.analyzer.tokenize(phrase, keep_stopwords=keep_stopwords)
        if not tokens:
            return {}

        postings = [self.index.get_posting(t) for t in tokens]
        if any(p is None for p in postings):
            return {}

        # candidate docs = docs that contain all terms
        docs_sets = [set(p.keys()) for p in postings if p is not None]
        cand_docs = sorted(set.intersection(*docs_sets)) if docs_sets else []

        out: dict[str, list[int]] = {}
        for d in cand_docs:
            pos_lists = [postings[i][d] for i in range(len(tokens))]  # type: ignore[index]
            if any(not ps for ps in pos_lists):
                continue
            # Find p such that p+j in pos_lists[j] for all j
            sets = [set(lst) for lst in pos_lists[1:]]
            starts: list[int] = []
            for p0 in pos_lists[0]:
                ok = True
                for j, s in enumerate(sets, start=1):
                    if (p0 + j) not in s:
                        ok = False
                        break
                if ok:
                    starts.append(p0)
            if starts:
                out[d] = starts
        return out


# =========================
# BM25 scorer
# =========================
class BM25Scorer:
    def __init__(self, index: InvertedIndex, k1: float = 1.2, b: float = 0.75) -> None:
        self.index = index
        self.k1 = float(k1)
        self.b = float(b)

    def score(self, doc_id: str, q_terms: list[str]) -> float:
        # document length (non-stopword count) with safety
        dl = float(max(1, self.index.doc_lengths.get(doc_id, 0)))

        # average doc length
        n_docs = max(1, len(self.index.doc_ids))
        total_len = sum(self.index.doc_lengths.get(d, 0) for d in self.index.doc_ids)
        avgdl = max(1.0, float(total_len) / float(n_docs))

        score = 0.0
        for t in q_terms:
            posting = self.index.get_posting(t)
            if not posting:
                # unknown term contributes nothing
                continue

            tf = len(posting.get(doc_id, []))
            if tf == 0:
                continue

            df = len(posting)  # document frequency for this term
            # Robertson–Sparck Jones IDF with +0.5 smoothing; keep non-negative
            idf = max(0.0, math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0))

            denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / avgdl))
            score += idf * ((tf * (self.k1 + 1.0)) / denom)

        return score


# =========================
# Searcher (boolean + ranked)
# =========================
class Searcher:
    """Boolean retrieval and BM25 ranked retrieval."""

    def __init__(self, index: InvertedIndex, analyzer: Analyzer) -> None:
        self.index = index
        self.analyzer = analyzer
        self.parser = QueryParser(analyzer)
        self._all_docs: list[str] = sorted(self.index.doc_ids)

    # ------- Boolean -------
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

    # ------- Ranked (BM25) -------
    def search_ranked(self, q: str, *, k: int = 10) -> list[tuple[str, float]]:
        """
        BM25 Top-K over docs containing at least one query term.
        Returns list of (doc_id, score) sorted by score desc.
        """
        terms = self.analyzer.tokenize(q, keep_stopwords=False)
        if not terms:
            return []

        # candidate docs = union of postings of known terms
        cand: set[str] = set()
        for t in terms:
            posting = self.index.get_posting(t)
            if posting:
                cand.update(posting.keys())
        if not cand:
            return []

        scorer = BM25Scorer(self.index)
        scores: list[tuple[str, float]] = []
        for d in cand:
            s = scorer.score(d, terms)
            if s > 0.0:
                scores.append((d, s))

        # sort by score desc, then doc_id asc for stability
        scores.sort(key=lambda x: (-x[1], x[0]))
        return scores[:k]
