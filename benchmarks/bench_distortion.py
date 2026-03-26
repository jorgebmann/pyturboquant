#!/usr/bin/env python3
"""Benchmark: empirical MSE and IP error vs Shannon lower bound.

Sweeps bit-widths b=1..8 and dimensions d=128..3072, reporting:
- Normalized MSE distortion for MSE quantizer
- Shannon lower bound comparison
- Inner product estimation error for IP quantizer

Usage:
    python benchmarks/bench_distortion.py
"""

from __future__ import annotations

import time

import torch

from pyturboquant.core.mse_quantizer import MSEQuantizer
from pyturboquant.core.prod_quantizer import InnerProductQuantizer
from pyturboquant.utils.metrics import mse_distortion, shannon_lower_bound


def bench_mse_distortion(
    dims: list[int], bits_range: list[int], n_vectors: int = 5000
) -> None:
    """Measure empirical MSE distortion vs Shannon lower bound."""
    print("=" * 80)
    print("MSE Distortion Benchmark")
    print("=" * 80)
    print(f"{'dim':>6} {'bits':>5} {'Empirical MSE':>15} {'Shannon LB':>12} {'Ratio':>8} {'Time (ms)':>10}")
    print("-" * 80)

    for d in dims:
        for b in bits_range:
            g = torch.Generator().manual_seed(42)
            x = torch.randn(n_vectors, d, generator=g)
            x = x / torch.linalg.norm(x, dim=-1, keepdim=True)

            q = MSEQuantizer(dim=d, bits=b, seed=0)

            t0 = time.perf_counter()
            qt = q.quantize(x)
            x_hat = q.dequantize(qt)
            elapsed = (time.perf_counter() - t0) * 1000

            emp_mse = mse_distortion(x, x_hat).item()
            lb = shannon_lower_bound(b)
            ratio = emp_mse / lb if lb > 0 else float("inf")

            print(f"{d:>6} {b:>5} {emp_mse:>15.8f} {lb:>12.8f} {ratio:>8.2f}x {elapsed:>10.1f}")
        print()


def bench_ip_error(
    dims: list[int], bits_range: list[int], n_vectors: int = 2000
) -> None:
    """Measure inner product estimation error."""
    print("=" * 80)
    print("Inner Product Estimation Error Benchmark")
    print("=" * 80)
    print(f"{'dim':>6} {'bits':>5} {'Mean |err|':>12} {'Rel err':>10} {'Time (ms)':>10}")
    print("-" * 80)

    for d in dims:
        for b in bits_range:
            if b < 2:
                continue
            g = torch.Generator().manual_seed(42)
            x = torch.randn(n_vectors, d, generator=g)
            y = torch.randn(n_vectors, d, generator=g)

            q = InnerProductQuantizer(dim=d, bits=b, seed=0)

            t0 = time.perf_counter()
            qt = q.quantize(x)
            est_ip = q.estimate_inner_product(qt, y)
            elapsed = (time.perf_counter() - t0) * 1000

            true_ip = (x * y).sum(dim=-1)
            abs_err = (est_ip - true_ip).abs().mean().item()
            rel_err = abs_err / (true_ip.abs().mean().item() + 1e-8)

            print(f"{d:>6} {b:>5} {abs_err:>12.6f} {rel_err:>10.4f} {elapsed:>10.1f}")
        print()


def main() -> None:
    dims = [128, 256, 512, 1024]
    bits_range = [1, 2, 3, 4]

    bench_mse_distortion(dims, bits_range)
    bench_ip_error(dims, bits_range)

    print("Benchmark complete.")


if __name__ == "__main__":
    main()
