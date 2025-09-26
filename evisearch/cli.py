from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from evisearch.analyzer import Analyzer
from evisearch.highlight import Highlighter
from evisearch.indexer import IndexWriter, InvertedIndex
from evisearch.searcher import PhraseMatcher, Searcher
from evisearch.storage import load_index, save_index


def build_index_from_dir(directory: str | Path, *, index_stopwords: bool = True) -> InvertedIndex:
    """Index all .txt files in a directory (non-recursive)."""
    az = Analyzer()
    writer = IndexWriter(az, index_stopwords=index_stopwords)

    folder = Path(directory)
    count = 0
    for p in sorted(folder.glob("*.txt")):
        text = p.read_text(encoding="utf-8", errors="ignore")
        writer.add(p.stem, text)
        count += 1
    if count == 0:
        raise SystemExit(f"No .txt files found in: {folder}")

    return writer.commit()


def first_hit_pos(index: InvertedIndex, terms: Iterable[str], doc_id: str) -> int | None:
    """Return the first global token position in doc where ANY term occurs."""
    best: int | None = None
    for t in terms:
        posting = index.get_posting(t)
        if not posting:
            continue
        positions = posting.get(doc_id)
        if not positions:
            continue
        pos0 = positions[0]
        if best is None or pos0 < best:
            best = pos0
    return best


def cmd_index(args: argparse.Namespace) -> None:
    idx = build_index_from_dir(args.directory, index_stopwords=not args.no_stopwords)
    save_index(idx, args.out)
    print(f"Indexed {len(idx.doc_ids)} docs → {args.out}")
    print(f"Vocabulary size: {idx.vocabulary_size()}")


def cmd_search(args: argparse.Namespace) -> None:
    idx = load_index(args.index)
    az = Analyzer()
    s = Searcher(idx, az)
    hi = Highlighter(idx)

    mode = "ranked"
    if args.phrase:
        mode = "phrase"
    elif args.boolean:
        mode = "boolean"

    if mode == "boolean":
        docs = s.search_boolean(args.query)
        if not docs:
            print("No matches.")
            return
        print(f"[boolean] {len(docs)} docs:")
        for d in docs:
            # try to show a small snippet line using first query term occurrence
            terms = az.tokenize(args.query, keep_stopwords=False)
            pos = first_hit_pos(idx, terms, d)
            if pos is not None:
                snippet, line_no = hi.build_snippet(d, pos)
                print(f"- {d}\t(line {line_no})\t{snippet}")
            else:
                print(f"- {d}")

    elif mode == "phrase":
        pm = PhraseMatcher(idx, az)
        hits = pm.match(args.query, keep_stopwords=True)
        if not hits:
            print("No phrase matches.")
            return
        print(f"[phrase] {len(hits)} docs:")
        for d in sorted(hits.keys()):
            for start in hits[d]:
                snippet, line_no = hi.build_snippet(d, start)
                print(f"- {d}\t(line {line_no})\t{snippet}")

    else:  # ranked (BM25)
        results = s.search_ranked(args.query, k=args.k)
        if not results:
            print("No results.")
            return
        print(f"[ranked] top-{len(results)}:")
        terms = az.tokenize(args.query, keep_stopwords=False)
        for d, score in results:
            pos = first_hit_pos(idx, terms, d)
            if pos is not None:
                snippet, line_no = hi.build_snippet(d, pos)
                print(f"- {d}\t{score:.3f}\t(line {line_no})\t{snippet}")
            else:
                print(f"- {d}\t{score:.3f}")


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evisearch",
        description="EviSearch-Py: evidence-level mini search (CLI)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="Index all .txt files in a directory")
    p_index.add_argument("directory", help="Folder containing .txt files")
    p_index.add_argument("--out", default="index.json", help="Output index file (JSON)")
    p_index.add_argument("--no-stopwords", action="store_true", help="Do not index stopwords")
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="Search the saved index")
    p_search.add_argument("query", help="Query string")
    p_search.add_argument("-k", type=int, default=10, help="Top-K for ranked mode")
    p_search.add_argument("--index", default="index.json", help="Path to saved index JSON")
    mode = p_search.add_mutually_exclusive_group()
    mode.add_argument("--boolean", action="store_true", help="Boolean mode (AND/OR/NOT)")
    mode.add_argument("--phrase", action="store_true", help="Exact phrase mode")
    p_search.set_defaults(func=cmd_search)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = make_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
