#!/usr/bin/env python3
"""Precompute Lloyd-Max codebooks for standard Gaussian N(0,1) at bit-widths 1-8.

Saves results as .pt files under data/codebooks/. Requires scipy.

Usage:
    python scripts/precompute_codebooks.py
"""

from __future__ import annotations

from pathlib import Path

import torch

from pyturboquant.core.codebook import _compute_gaussian_codebook


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "data" / "codebooks"
    out_dir.mkdir(parents=True, exist_ok=True)

    for bits in range(1, 9):
        print(f"Computing codebook for b={bits} ({1 << bits} levels)...")
        cb = _compute_gaussian_codebook(bits)
        path = out_dir / f"gaussian_b{bits}.pt"
        torch.save(
            {
                "centroids": cb.centroids,
                "boundaries": cb.boundaries,
                "mse_cost": cb.mse_cost,
                "bits": cb.bits,
            },
            path,
        )
        print(f"  -> {path}  (MSE cost = {cb.mse_cost:.8f})")

    print("Done.")


if __name__ == "__main__":
    main()
