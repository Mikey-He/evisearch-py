from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore

from bench.benchmark import (
    approx_index_size_bytes,
    build_index,
    load_docs_from_dir,
    run_benchmark,
    sample_queries,
    top_terms_for_queries,
)
from evisearch.searcher import Searcher


def load_gold_jsonl(path: Path) -> list[tuple[str, set[str]]]:
    """
    Read a JSONL file where each line looks like:
      {"q": "ap credit policy", "relevant": ["doc1", "doc3"]}
    Returns a list of (query, set_of_relevant_doc_ids).
    """
    out: list[tuple[str, set[str]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        q = str(obj["q"])
        rel = {str(x) for x in obj.get("relevant", [])}
        out.append((q, rel))
    return out


def precision_at_10(
    index_dir_docs: str | Path,
    index_stopwords: bool,
    gold_path: Path,
) -> float:
    """
    Compute mean Precision@10 against a small gold set.
    """
    docs = load_docs_from_dir(index_dir_docs)
    idx, az = build_index(docs, index_stopwords=index_stopwords)
    s = Searcher(idx, az)

    gold = load_gold_jsonl(gold_path)
    if not gold:
        return 0.0

    total = 0.0
    for q, relevant in gold:
        results = s.search_ranked(q, k=10)
        retrieved = [d for d, _ in results]
        hits = sum(1 for d in retrieved if d in relevant)
        total += hits / 10.0
    return total / float(len(gold))


# Plot helpers
def plot_qps_p95(
    stats_ranked: dict[str, float],
    stats_boolean: dict[str, float],
    out: Path,
) -> None:
    modes = ["ranked", "boolean"]
    qps_vals = [stats_ranked["qps"], stats_boolean["qps"]]
    p95_vals = [stats_ranked["p95_ms"], stats_boolean["p95_ms"]]

    x = range(len(modes))
    width = 0.35

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.bar([i - width / 2 for i in x], qps_vals, width, label="QPS")
    ax.bar([i + width / 2 for i in x], p95_vals, width, label="P95 (ms)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(modes)
    ax.set_ylabel("QPS / ms")
    ax.set_title("QPS & P95 by Mode")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def plot_index_size(size_bytes: int, out: Path) -> None:
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    ax.bar(["index.json"], [size_bytes])
    ax.set_ylabel("Bytes")
    ax.set_title("Index Size (JSON)")
    for i, v in enumerate([size_bytes]):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def maybe_plot_precision_at_10(
    docs_dir: str | Path,
    index_stopwords: bool,
    gold: Path | None,
    out: Path,
) -> None:
    if gold is None or not gold.exists():
        return
    p = precision_at_10(docs_dir, index_stopwords, gold)
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    ax.bar(["Precision@10"], [p])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Precision@10")
    ax.text(0, p, f"{p:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out, dpi=150)

# CLI

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run EviSearch bench and plot charts")
    parser.add_argument("--docs", default="data/docs", help="Directory with .txt files")
    parser.add_argument("--queries", type=int, default=40, help="Number of random queries")
    parser.add_argument("--terms-per-query", type=int, default=2, help="Terms per query")
    parser.add_argument("-k", type=int, default=10, help="Top-K for ranked mode")
    parser.add_argument("--index-stopwords", action="store_true", help="Index stopwords")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--outdir", default="bench/plots", help="Output directory for PNGs")
    parser.add_argument("--gold", default="", help="Optional JSONL for Precision@10")
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Build index once
    docs = load_docs_from_dir(args.docs)
    idx, az = build_index(docs, index_stopwords=args.index_stopwords)

    # Generate queries
    texts = [t for _, t in docs]
    vocab_terms = top_terms_for_queries(az, texts, topn=200)
    queries = sample_queries(
        vocab_terms,
        n=args.queries,
        terms_per_query=args.terms_per_query,
        seed=args.seed,
    )
    if not queries:
        raise SystemExit("No queries generated; need more content terms.")

    # Run both modes
    _dur_r, stats_r = run_benchmark(idx, az, queries, mode="ranked", k=args.k, warmup=5)
    _dur_b, stats_b = run_benchmark(idx, az, queries, mode="boolean", k=args.k, warmup=5)

    # Index size (single index)
    size_bytes = approx_index_size_bytes(idx, outdir / "index_snapshot.json")

    # Plots
    plot_qps_p95(stats_r, stats_b, outdir / "qps_p95.png")
    plot_index_size(size_bytes, outdir / "index_size.png")

    gold_path = Path(args.gold) if args.gold else None
    maybe_plot_precision_at_10(
        args.docs,
        args.index_stopwords,
        gold_path,
        outdir / "precision_at_10.png",
    )

    # Console summary
    print("=== Charts saved ===")
    print(f"QPS & P95 : {outdir / 'qps_p95.png'}")
    print(f"Index size: {outdir / 'index_size.png'}")
    if gold_path and gold_path.exists():
        print(f"Precision@10: {outdir / 'precision_at_10.png'}")


if __name__ == "__main__":
    main()
