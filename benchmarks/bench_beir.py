#!/usr/bin/env python3
"""Benchmark: BEIR retrieval quality with and without pyturboquant compression.

For each (dataset, bits) combination we report:

- Relevance metrics vs qrels: nDCG@10, Recall@100, MAP@10 (via BEIR's
  ``EvaluateRetrieval`` using pytrec_eval).
- Self-Recall@10 / @100: overlap between the compressed top-k and the fp32
  top-k for the same embedding model. This isolates quantizer error from
  model error.
- Resource cost: index memory (MB), index wall time (ms), search latency
  per query (ms).

The fp32 baseline is an exact chunked matmul + topk over the L2-normalized
embeddings, so ``metric="ip"`` on the :class:`TurboQuantIndex` equals cosine
similarity. Encoded corpora and queries are cached to disk in
``--cache-dir`` so sweeping bits or re-running does not re-encode.

Examples:
    # Small suite (all fit under 60k docs), BGE-base, default bits 2/3/4
    python benchmarks/bench_beir.py

    # Fast iteration on a single dataset
    python benchmarks/bench_beir.py --datasets scifact --model minilm

    # Subset MSMARCO for a quick scale check, dump JSON
    python benchmarks/bench_beir.py --datasets msmarco \\
        --max-corpus-size 500000 --output-json msmarco.json

Usage:
    pip install -e ".[bench]"
    python benchmarks/bench_beir.py [flags]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

# Ensure ``python benchmarks/bench_beir.py`` works without installing the
# benchmarks directory as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _beir_helpers import (
    DATASETS,
    MODELS,
    SMALL_SUITE,
    Timer,
    corpus_to_texts,
    download_and_load,
    encode_with_cache,
    env_hf_offline_hint,
    format_number,
    fp32_search,
    parse_dataset_list,
    pick_device,
    queries_to_texts,
    self_recall_at_k,
    to_trec_results,
)

from pyturboquant.search.index import TurboQuantIndex


@dataclass
class RowResult:
    """One row in the per-dataset results table."""

    dataset: str
    model: str
    mode: str  # "fp32" or "b=<bits>"
    bits: int | None
    n_corpus: int
    n_queries: int
    dim: int
    ndcg_at_10: float
    recall_at_100: float
    map_at_10: float
    self_recall_at_10: float | None
    self_recall_at_100: float | None
    mem_mb: float
    index_ms: float | None
    search_ms_per_query: float


def parse_bits(raw: str) -> list[int]:
    """Parse ``"2,3,4"`` into ``[2, 3, 4]``, validating IP-quantizer range."""
    bits = [int(x.strip()) for x in raw.split(",") if x.strip()]
    for b in bits:
        if b < 2:
            raise ValueError(
                f"bits must be >= 2 for the inner-product quantizer, got {b}"
            )
    return bits


def fp32_mem_mb(n: int, dim: int) -> float:
    """Return the fp32 storage footprint of an ``(n, dim)`` matrix in MB."""
    return n * dim * 4 / (1024 * 1024)


def evaluate_trec(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
) -> tuple[float, float, float]:
    """Run BEIR's pytrec_eval wrapper; return ``(nDCG@10, Recall@100, MAP@10)``."""
    from beir.retrieval.evaluation import EvaluateRetrieval

    ndcg, _map, recall, _precision = EvaluateRetrieval.evaluate(
        qrels, results, k_values=[1, 10, 100]
    )
    return (
        float(ndcg.get("NDCG@10", 0.0)),
        float(recall.get("Recall@100", 0.0)),
        float(_map.get("MAP@10", 0.0)),
    )


def time_search(
    search_fn: Callable[[torch.Tensor], object],
    warmup: int,
    queries: torch.Tensor,
) -> float:
    """Return average search latency in ms/query (with a small warmup)."""
    if warmup > 0 and queries.shape[0] > 0:
        _ = search_fn(queries[: min(warmup, queries.shape[0])])
    t0 = time.perf_counter()
    _ = search_fn(queries)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return elapsed_ms / max(1, queries.shape[0])


def run_dataset(
    *,
    dataset_key: str,
    model_alias: str,
    bits_list: list[int],
    device: str,
    cache_dir: Path,
    encode_batch_size: int,
    max_corpus_size: int | None,
    search_batch_size: int,
) -> list[RowResult]:
    """Run fp32 baseline + all bit settings on one dataset; return rows."""
    spec = DATASETS[dataset_key]
    model_spec = MODELS[model_alias]

    print(f"\n>>> Loading {spec.display_name} (split={spec.split})")
    corpus, queries, qrels = download_and_load(dataset_key, cache_dir)
    doc_ids, corpus_texts = corpus_to_texts(corpus, max_size=max_corpus_size)
    query_ids, query_texts = queries_to_texts(queries)

    if max_corpus_size is not None and len(doc_ids) < len(corpus):
        # Subsampled: drop qrels entries whose positive docs are now missing.
        kept = set(doc_ids)
        qrels = {
            qid: {d: r for d, r in docs.items() if d in kept}
            for qid, docs in qrels.items()
        }
        qrels = {qid: docs for qid, docs in qrels.items() if docs}
        query_ids = [q for q in query_ids if q in qrels]
        query_texts = [queries[q] for q in query_ids]

    print(
        f"    n_corpus={len(doc_ids):,}  n_queries={len(query_ids):,}  "
        f"dim={model_spec.dim}"
    )

    print(f">>> Encoding corpus with {model_spec.display_name}")
    corpus_emb = encode_with_cache(
        model_alias=model_alias,
        dataset_key=dataset_key,
        kind="corpus" if max_corpus_size is None else f"corpus_n{max_corpus_size}",
        texts=corpus_texts,
        ids=doc_ids,
        cache_dir=cache_dir,
        device=device,
        batch_size=encode_batch_size,
    )
    print(f">>> Encoding queries with {model_spec.display_name}")
    queries_emb = encode_with_cache(
        model_alias=model_alias,
        dataset_key=dataset_key,
        kind="queries" if max_corpus_size is None else f"queries_n{max_corpus_size}",
        texts=query_texts,
        ids=query_ids,
        cache_dir=cache_dir,
        device=device,
        batch_size=encode_batch_size,
    )

    k_max = 100
    dim = corpus_emb.shape[1]
    n_corpus = corpus_emb.shape[0]
    n_queries = queries_emb.shape[0]

    print(">>> Running fp32 baseline (exact chunked matmul)")
    base_scores, base_indices = fp32_search(
        corpus_emb, queries_emb, k=k_max, chunk_size=search_batch_size, device=device
    )
    base_results = to_trec_results(base_scores, base_indices, doc_ids, query_ids)
    base_ndcg, base_recall, base_map = evaluate_trec(qrels, base_results)
    base_latency_ms_per_q = time_search(
        lambda q: fp32_search(
            corpus_emb, q, k=k_max, chunk_size=search_batch_size, device=device
        ),
        warmup=min(4, n_queries),
        queries=queries_emb,
    )

    rows: list[RowResult] = [
        RowResult(
            dataset=spec.display_name,
            model=model_spec.display_name,
            mode="fp32",
            bits=None,
            n_corpus=n_corpus,
            n_queries=n_queries,
            dim=dim,
            ndcg_at_10=base_ndcg,
            recall_at_100=base_recall,
            map_at_10=base_map,
            self_recall_at_10=1.0,
            self_recall_at_100=1.0,
            mem_mb=fp32_mem_mb(n_corpus, dim),
            index_ms=None,
            search_ms_per_query=base_latency_ms_per_q,
        )
    ]

    for bits in bits_list:
        print(f">>> Indexing with TurboQuant bits={bits}")
        index = TurboQuantIndex(
            dim=dim,
            bits=bits,
            metric="ip",
            seed=0,
            device=device,
            search_batch_size=search_batch_size,
        )
        with Timer() as idx_timer:
            index.add(corpus_emb)
            index.consolidate()
        tq_scores, tq_indices = index.search(queries_emb, k=k_max)
        tq_scores_cpu = tq_scores.cpu()
        tq_indices_cpu = tq_indices.cpu()
        tq_results = to_trec_results(
            tq_scores_cpu, tq_indices_cpu, doc_ids, query_ids
        )
        tq_ndcg, tq_recall, tq_map = evaluate_trec(qrels, tq_results)
        sr10 = self_recall_at_k(tq_indices_cpu, base_indices, k=10)
        sr100 = self_recall_at_k(tq_indices_cpu, base_indices, k=100)
        tq_latency_ms_per_q = time_search(
            lambda q, idx=index: idx.search(q, k=k_max),
            warmup=min(4, n_queries),
            queries=queries_emb,
        )
        rows.append(
            RowResult(
                dataset=spec.display_name,
                model=model_spec.display_name,
                mode=f"b={bits}",
                bits=bits,
                n_corpus=n_corpus,
                n_queries=n_queries,
                dim=dim,
                ndcg_at_10=tq_ndcg,
                recall_at_100=tq_recall,
                map_at_10=tq_map,
                self_recall_at_10=sr10,
                self_recall_at_100=sr100,
                mem_mb=index.memory_usage_mb,
                index_ms=idx_timer.elapsed_ms,
                search_ms_per_query=tq_latency_ms_per_q,
            )
        )

    return rows


def print_dataset_table(rows: list[RowResult]) -> None:
    """Pretty-print the per-dataset block."""
    if not rows:
        return
    head = rows[0]
    title = (
        f"\n{head.dataset} ({head.model}, d={head.dim}, "
        f"n_corpus={head.n_corpus:,}, n_queries={head.n_queries:,})"
    )
    print(title)
    print("-" * len(title.strip()))
    header = (
        f"  {'mode':<6} {'nDCG@10':>8} {'Recall@100':>11} {'MAP@10':>8} "
        f"{'SelfR@10':>9} {'SelfR@100':>10} {'mem_MB':>9} "
        f"{'idx_ms':>8} {'q_ms':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        print(
            f"  {r.mode:<6} "
            f"{format_number(r.ndcg_at_10, '>8.4f')} "
            f"{format_number(r.recall_at_100, '>11.4f')} "
            f"{format_number(r.map_at_10, '>8.4f')} "
            f"{format_number(r.self_recall_at_10, '>9.4f')} "
            f"{format_number(r.self_recall_at_100, '>10.4f')} "
            f"{format_number(r.mem_mb, '>9.2f')} "
            f"{format_number(r.index_ms, '>8.0f')} "
            f"{format_number(r.search_ms_per_query, '>7.2f')}"
        )


def print_aggregate(all_rows: list[RowResult]) -> None:
    """Average primary metrics across the small suite per (model, mode)."""
    small_display = {DATASETS[d].display_name for d in SMALL_SUITE}
    small_rows = [r for r in all_rows if r.dataset in small_display]
    if not small_rows:
        return

    groups: dict[tuple[str, str], list[RowResult]] = {}
    for r in small_rows:
        groups.setdefault((r.model, r.mode), []).append(r)

    title = "\nAverage across small suite"
    print(title)
    print("-" * len(title.strip()))
    header = (
        f"  {'model':<24} {'mode':<6} {'nDCG@10':>8} {'Recall@100':>11} "
        f"{'MAP@10':>8} {'SelfR@10':>9} {'SelfR@100':>10}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    ordered_modes = [
        "fp32",
        *sorted(
            {m for (_, m) in groups if m != "fp32"},
            key=lambda m: int(m.split("=")[1]),
            reverse=True,
        ),
    ]
    for (model_name, mode), rows in sorted(
        groups.items(),
        key=lambda kv: (kv[0][0], ordered_modes.index(kv[0][1])),
    ):
        n = len(rows)
        avg_ndcg = sum(r.ndcg_at_10 for r in rows) / n
        avg_recall = sum(r.recall_at_100 for r in rows) / n
        avg_map = sum(r.map_at_10 for r in rows) / n
        sr10_vals = [r.self_recall_at_10 for r in rows if r.self_recall_at_10 is not None]
        sr100_vals = [
            r.self_recall_at_100 for r in rows if r.self_recall_at_100 is not None
        ]
        avg_sr10 = sum(sr10_vals) / len(sr10_vals) if sr10_vals else None
        avg_sr100 = sum(sr100_vals) / len(sr100_vals) if sr100_vals else None
        print(
            f"  {model_name:<24} {mode:<6} "
            f"{avg_ndcg:>8.4f} {avg_recall:>11.4f} {avg_map:>8.4f} "
            f"{format_number(avg_sr10, '>9.4f')} "
            f"{format_number(avg_sr100, '>10.4f')}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--datasets",
        default="small",
        help=(
            "Comma-separated dataset keys, or the special tokens "
            "'small' (default: scifact,nfcorpus,fiqa,scidocs,arguana) or "
            "'all' (small + msmarco). Valid keys: "
            + ", ".join(sorted(DATASETS))
        ),
    )
    p.add_argument(
        "--model",
        default="bge-base",
        choices=sorted(MODELS),
        help="Embedding model alias (default: bge-base).",
    )
    p.add_argument(
        "--bits",
        default="2,3,4",
        help="Comma-separated bit budgets for TurboQuant (default: 2,3,4).",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="Torch device: 'auto', 'cpu', 'cuda', 'mps', or 'cuda:N'.",
    )
    p.add_argument(
        "--cache-dir",
        default="data/beir_cache",
        help="Directory for BEIR dataset downloads and embedding caches.",
    )
    p.add_argument(
        "--output-json",
        default=None,
        help="Optional path to dump all rows as JSON.",
    )
    p.add_argument(
        "--max-corpus-size",
        type=int,
        default=None,
        help=(
            "Truncate each corpus to the first N documents after loading. "
            "Useful for MSMARCO (~8.8M) on limited hardware."
        ),
    )
    p.add_argument(
        "--encode-batch-size",
        type=int,
        default=256,
        help="SentenceTransformer encoding batch size (default: 256).",
    )
    p.add_argument(
        "--search-batch-size",
        type=int,
        default=65_536,
        help=(
            "Chunk size for both the fp32 baseline matmul and TurboQuantIndex "
            "search reconstruction window (default: 65536)."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_keys = parse_dataset_list(args.datasets)
    bits_list = parse_bits(args.bits)
    device = pick_device(args.device)
    cache_dir = Path(args.cache_dir).resolve()

    bar = "=" * 96
    print(bar)
    print("pyturboquant BEIR benchmark")
    print(f"  datasets : {', '.join(dataset_keys)}")
    print(f"  model    : {args.model} ({MODELS[args.model].hf_id})")
    print(f"  bits     : {bits_list}")
    print(f"  device   : {device}{env_hf_offline_hint()}")
    print(f"  cache    : {cache_dir}")
    print(bar)

    all_rows: list[RowResult] = []
    for dk in dataset_keys:
        try:
            rows = run_dataset(
                dataset_key=dk,
                model_alias=args.model,
                bits_list=bits_list,
                device=device,
                cache_dir=cache_dir,
                encode_batch_size=args.encode_batch_size,
                max_corpus_size=args.max_corpus_size,
                search_batch_size=args.search_batch_size,
            )
        except Exception as exc:
            print(f"!!! {dk} failed: {exc.__class__.__name__}: {exc}")
            continue
        print_dataset_table(rows)
        all_rows.extend(rows)

    if len(dataset_keys) > 1:
        print_aggregate(all_rows)

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump([asdict(r) for r in all_rows], f, indent=2)
        print(f"\nWrote {len(all_rows)} rows to {out}")

    print("\nBenchmark complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
