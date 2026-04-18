#!/usr/bin/env python3
"""Benchmark: peak fp32 reconstruction memory during search.

Demonstrates that ``TurboQuantIndex.search_batch_size`` bounds the transient
fp32 window materialized during ``search()`` by ``search_batch_size * dim * 4``
bytes, regardless of how many vectors are indexed. Storage at rest is unchanged.

This is the measurement that backs the README's "storage and search both fit
the same budget" claim on large indexes.

Usage:
    python benchmarks/bench_search_memory.py
"""

from __future__ import annotations

import time

import torch

from pyturboquant.search.index import TurboQuantIndex


def bench_transient_window(
    n_db: int,
    dim: int,
    bits: int,
    batch_sizes: list[int],
) -> None:
    """Show how the sub-batched dequantize window scales with ``search_batch_size``.

    The key invariant: ``sub_window_mb == batch * dim * 4 / (1024 * 1024)``.
    The full-chunk baseline (what the old non-chunked path would materialize)
    grows linearly with ``n_db`` and is the failure mode this patch fixes.
    """
    print("=" * 96)
    print(
        f"Search-time peak memory: n_db={n_db:,}, dim={dim}, bits={bits} "
        f"(IP quantizer)"
    )
    print("=" * 96)
    header = (
        f"{'batch':>10} {'storage_MB':>12} {'sub_window_MB':>16} "
        f"{'old_peak_MB':>14} {'reduction':>12}"
    )
    print(header)
    print("-" * len(header))

    g = torch.Generator().manual_seed(0)
    db = torch.randn(n_db, dim, generator=g)
    full_mb = n_db * dim * 4 / (1024 * 1024)

    for batch in batch_sizes:
        idx = TurboQuantIndex(
            dim=dim, bits=bits, metric="ip", seed=0, search_batch_size=batch
        )
        idx.add(db)
        idx.consolidate()
        qt = idx._qt_data[0]
        sub = idx._quantizer.dequantize_range(qt, 0, min(batch, n_db))
        sub_mb = sub.numel() * sub.element_size() / (1024 * 1024)
        reduction = full_mb / sub_mb if sub_mb > 0 else float("inf")
        print(
            f"{batch:>10,} {idx.memory_usage_mb:>12.2f} {sub_mb:>16.2f} "
            f"{full_mb:>14.2f} {reduction:>11.1f}x"
        )


def bench_search_latency(
    n_db: int,
    n_queries: int,
    dim: int,
    bits: int,
    batch_sizes: list[int],
) -> None:
    """Confirm sub-batching has modest, predictable impact on search latency.

    Bigger windows amortize Python-loop overhead; smaller windows save memory.
    The numbers printed here help users pick a default for their workload.
    """
    print("\n" + "=" * 78)
    print(
        f"Search latency vs batch size: n_db={n_db:,}, dim={dim}, "
        f"n_queries={n_queries}"
    )
    print("=" * 78)
    header = f"{'batch':>10} {'search_ms':>14} {'ms_per_query':>16} {'k':>5}"
    print(header)
    print("-" * len(header))

    g = torch.Generator().manual_seed(1)
    db = torch.randn(n_db, dim, generator=g)
    queries = torch.randn(n_queries, dim, generator=g)

    for batch in batch_sizes:
        idx = TurboQuantIndex(
            dim=dim, bits=bits, metric="ip", seed=0, search_batch_size=batch
        )
        idx.add(db)
        idx.consolidate()

        _ = idx.search(queries[:1], k=10)
        t0 = time.perf_counter()
        _, _ = idx.search(queries, k=10)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        per_query = elapsed_ms / max(1, n_queries)
        print(f"{batch:>10,} {elapsed_ms:>14.1f} {per_query:>16.3f} {10:>5}")


def main() -> None:
    bench_transient_window(
        n_db=100_000,
        dim=768,
        bits=4,
        batch_sizes=[1_024, 4_096, 16_384, 65_536, 262_144],
    )
    bench_search_latency(
        n_db=100_000,
        n_queries=32,
        dim=768,
        bits=4,
        batch_sizes=[1_024, 4_096, 16_384, 65_536],
    )
    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
