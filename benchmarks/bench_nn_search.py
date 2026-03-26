#!/usr/bin/env python3
"""Benchmark: Nearest Neighbor search recall and indexing time.

Reports:
- Recall@1, @10, @100 on synthetic data
- Indexing time (wall-clock for add())
- Search time per query
- Comparison vs brute-force exact search

Usage:
    python benchmarks/bench_nn_search.py
"""

from __future__ import annotations

import time

import torch

from pyturboquant.search.index import TurboQuantIndex


def compute_recall(
    pred_indices: torch.Tensor, true_indices: torch.Tensor, k: int
) -> float:
    """Compute recall@k: fraction of true top-k that appear in predicted top-k."""
    nq = pred_indices.shape[0]
    hits = 0
    total = 0
    for i in range(nq):
        true_set = set(true_indices[i, :k].tolist())
        pred_set = set(pred_indices[i, :k].tolist())
        hits += len(true_set & pred_set)
        total += k
    return hits / total if total > 0 else 0.0


def bench_recall(
    n_db: int,
    n_queries: int,
    dim: int,
    bits_list: list[int],
    ks: list[int],
) -> None:
    """Benchmark recall at various k values."""
    print("=" * 90)
    print(f"Recall Benchmark: n_db={n_db}, n_queries={n_queries}, dim={dim}")
    print("=" * 90)

    g = torch.Generator().manual_seed(42)
    db = torch.randn(n_db, dim, generator=g)
    queries = torch.randn(n_queries, dim, generator=g)

    # Brute-force exact IP
    true_scores = queries @ db.T
    max_k = max(ks)
    true_topk = torch.topk(true_scores, min(max_k, n_db), dim=-1)
    true_indices = true_topk.indices

    for bits in bits_list:
        print(f"\n  bits={bits}:")
        idx = TurboQuantIndex(dim=dim, bits=bits, metric="ip", seed=0)

        t0 = time.perf_counter()
        idx.add(db)
        add_time = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        _, pred_indices = idx.search(queries, k=max_k)
        search_time = (time.perf_counter() - t0) * 1000

        print(f"    Index time:  {add_time:.1f} ms")
        print(f"    Search time: {search_time:.1f} ms ({search_time / n_queries:.2f} ms/query)")
        print(f"    Memory:      {idx.memory_usage_mb:.3f} MB")

        for k in ks:
            if k > n_db:
                continue
            recall = compute_recall(pred_indices, true_indices, k)
            print(f"    Recall@{k:<4}: {recall:.4f}")


def bench_indexing_time(dims: list[int], n_db: int, bits: int) -> None:
    """Benchmark indexing time across dimensions."""
    print("\n" + "=" * 70)
    print(f"Indexing Time Benchmark: n_db={n_db}, bits={bits}")
    print("=" * 70)
    print(f"{'dim':>6} {'Time (ms)':>12} {'Throughput (vec/s)':>20}")
    print("-" * 45)

    for d in dims:
        g = torch.Generator().manual_seed(0)
        db = torch.randn(n_db, d, generator=g)

        idx = TurboQuantIndex(dim=d, bits=bits, metric="ip", seed=0)
        t0 = time.perf_counter()
        idx.add(db)
        elapsed = (time.perf_counter() - t0) * 1000

        throughput = n_db / (elapsed / 1000) if elapsed > 0 else 0
        print(f"{d:>6} {elapsed:>12.1f} {throughput:>20,.0f}")


def bench_memory_savings(dim: int, n_db: int, bits_list: list[int]) -> None:
    """Compare memory usage: quantized vs full precision."""
    print("\n" + "=" * 70)
    print(f"Memory Savings: dim={dim}, n_db={n_db}")
    print("=" * 70)

    fp32_mb = n_db * dim * 4 / (1024 * 1024)
    print(f"  FP32 baseline: {fp32_mb:.2f} MB")

    for bits in bits_list:
        idx = TurboQuantIndex(dim=dim, bits=bits, metric="ip", seed=0)
        g = torch.Generator().manual_seed(0)
        db = torch.randn(n_db, dim, generator=g)
        idx.add(db)
        quant_mb = idx.memory_usage_mb
        ratio = fp32_mb / quant_mb if quant_mb > 0 else float("inf")
        print(f"  bits={bits}: {quant_mb:.2f} MB  ({ratio:.1f}x compression)")


def main() -> None:
    bench_recall(
        n_db=1000,
        n_queries=50,
        dim=128,
        bits_list=[2, 3, 4],
        ks=[1, 10, 100],
    )

    bench_indexing_time(
        dims=[64, 128, 256, 512],
        n_db=1000,
        bits=4,
    )

    bench_memory_savings(
        dim=128,
        n_db=10000,
        bits_list=[2, 3, 4],
    )

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
