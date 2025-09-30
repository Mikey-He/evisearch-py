from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import random
import statistics
import time

from evisearch.analyzer import Analyzer
from evisearch.indexer import IndexWriter, InvertedIndex
from evisearch.searcher import Searcher
from evisearch.storage import save_index


# ----------------------------
# Utilities
# ----------------------------
def load_docs_from_dir(directory: str | Path) -> list[tuple[str, str]]:
    """Load all .txt files (non-recursive) as (doc_id, text)."""
    folder = Path(directory)
    files = sorted(folder.glob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt files found in: {folder.resolve()}")
    return [(p.stem, p.read_text(encoding="utf-8", errors="ignore")) for p in files]


def build_index(
    docs: list[tuple[str, str]], *, index_stopwords: bool = True
) -> tuple[InvertedIndex, Analyzer]:
    az = Analyzer()
    w = IndexWriter(az, index_stopwords=index_stopwords)
    for did, text in docs:
        w.add(did, text)
    return w.commit(), az


def top_terms_for_queries(az: Analyzer, texts: list[str], *, topn: int = 200) -> list[str]:
    """Return most-common *content* terms (stopwords already removed)."""
    cnt: Counter[str] = Counter()
    for t in texts:
        cnt.update(az.tokenize(t, keep_stopwords=False))
    # unique, non-empty
    terms = [t for t, _ in cnt.most_common(topn) if t]
    # keep order but dedup (in case)
    return list(dict.fromkeys(terms))


def sample_queries(
    terms: list[str], *, n: int = 20, terms_per_query: int = 2, seed: int = 42
) -> list[str]:
    if not terms:
        return []
    rng = random.Random(seed)
    out: list[str] = []
    for _ in range(n):
        if len(terms) >= terms_per_query:
            parts = rng.sample(terms, terms_per_query)
        else:
            parts = [rng.choice(terms)]
        out.append(" ".join(parts))
    return out


def percentile(values_ms: list[float], pct: float) -> float:
    """Nearest-rank percentile. values_ms must be non-empty."""
    xs = sorted(values_ms)
    if not xs:
        return 0.0
    if len(xs) == 1:
        return xs[0]
    k = int(round((pct / 100.0) * (len(xs) - 1)))
    return xs[max(0, min(k, len(xs) - 1))]


def approx_index_size_bytes(index: InvertedIndex, save_to: Path) -> int:
    """Serialize to JSON and return file size in bytes."""
    save_to.parent.mkdir(parents=True, exist_ok=True)
    save_index(index, save_to)
    return save_to.stat().st_size


# ----------------------------
# Benchmark core
# ----------------------------
def run_benchmark(
    index: InvertedIndex,
    analyzer: Analyzer,
    queries: list[str],
    *,
    mode: str = "ranked",
    k: int = 10,
    warmup: int = 5,
) -> tuple[list[float], dict[str, float]]:
    """Run queries and return (per-query durations, aggregate stats)."""
    s = Searcher(index, analyzer)

    # warmup
    for q in queries[:warmup]:
        if mode == "boolean":
            s.search_boolean(q.replace(" ", " AND "))
        else:
            s.search_ranked(q, k=k)

    durations_ms: list[float] = []
    total_results = 0

    for q in queries:
        t0 = time.perf_counter_ns()
        if mode == "boolean":
            docs = s.search_boolean(q.replace(" ", " AND "))
            total_results += len(docs)
        else:
            items = s.search_ranked(q, k=k)
            total_results += len(items)
        t1 = time.perf_counter_ns()
        durations_ms.append((t1 - t0) / 1e6)

    total_s = sum(durations_ms) / 1000.0 if durations_ms else 0.0
    qps = (len(queries) / total_s) if total_s > 0 else 0.0
    p50 = percentile(durations_ms, 50)
    p95 = percentile(durations_ms, 95)
    p99 = percentile(durations_ms, 99)
    avg = statistics.fmean(durations_ms) if durations_ms else 0.0

    stats = {
        "mode": mode,
        "queries": float(len(queries)),
        "total_results": float(total_results),
        "total_ms": sum(durations_ms),
        "qps": qps,
        "avg_ms": avg,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
    }
    return durations_ms, stats


def write_csv(durations_ms: list[float], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = "idx,duration_ms\n"
    rows = "\n".join(f"{i},{v:.3f}" for i, v in enumerate(durations_ms))
    out_csv.write_text(header + rows, encoding="utf-8")


# ----------------------------
# CLI
# ----------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark EviSearch-Py (QPS, P95, index size)")
    parser.add_argument(
        "--docs",
        default="data/docs",
        help="Directory with .txt files",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=20,
        help="Number of random queries",
    )
    parser.add_argument(
        "--terms-per-query",
        type=int,
        default=2,
        help="Terms per query",
    )
    parser.add_argument(
        "--mode",
        choices=["ranked", "boolean"],
        default="ranked",
        help="Search mode",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=10,
        help="Top-K for ranked mode",
    )
    parser.add_argument(
        "--index-stopwords",
        action="store_true",
        help="Index stopwords (phrase-friendly)",
    )
    parser.add_argument(
        "--csv",
        default="bench/last_run_durations.csv",
        help="Write per-query durations CSV",
    )
    parser.add_argument(
        "--index-file",
        default="bench/index_snapshot.json",
        help="Path to save index for size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args(argv)

    docs = load_docs_from_dir(args.docs)
    idx, az = build_index(docs, index_stopwords=args.index_stopwords)

    # query generation
    texts = [t for _, t in docs]
    vocab_terms = top_terms_for_queries(az, texts, topn=200)
    queries = sample_queries(
        vocab_terms,
        n=args.queries,
        terms_per_query=args.terms_per_query,
        seed=args.seed,
    )
    if not queries:
        raise SystemExit("No queries could be generated; need more content terms.")

    # run benchmark
    durations_ms, stats = run_benchmark(
        index=idx,
        analyzer=az,
        queries=queries,
        mode=args.mode,
        k=args.k,
        warmup=5,
    )

    # index size and CSV
    size_bytes = approx_index_size_bytes(idx, Path(args.index_file))
    write_csv(durations_ms, Path(args.csv))

    # outputs (避免超长行)
    docs_path = Path(args.docs).resolve()
    index_path = Path(args.index_file).resolve()
    csv_path = Path(args.csv).resolve()

    print("=== EviSearch Bench ===")
    print(f"docs_dir         : {docs_path}")
    print(f"docs_indexed     : {len(idx.doc_ids)}")
    print(f"mode             : {args.mode}")
    print(f"queries          : {int(stats['queries'])} (terms/query={args.terms_per_query})")
    print(f"QPS              : {stats['qps']:.2f}")
    print(
        "P50 / P95 / P99  : "
        f"{stats['p50_ms']:.2f} / {stats['p95_ms']:.2f} / {stats['p99_ms']:.2f} ms"
    )
    print(f"Avg latency      : {stats['avg_ms']:.2f} ms")
    print(f"Index size (JSON): {size_bytes} bytes -> {index_path}")
    print(f"Durations CSV    : {csv_path}")
