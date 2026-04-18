"""Lloyd-Max codebook solver and precomputed codebook loading.

The codebooks are precomputed for a standard Gaussian N(0,1) and scaled
by 1/sqrt(d) at load time to match the sphere coordinate distribution.

Shipped ``pyturboquant/data/codebooks/*.pt`` files cover common bit-widths so runtime
never needs SciPy. SciPy is only used when those files are missing and
``bits > 4`` (iterative Lloyd--Max) or when computing ``mse_cost`` without
cached values.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path

import torch

from pyturboquant.core.types import Codebook

_CODEBOOK_CACHE: dict[int, Codebook] = {}
_SCALED_CODEBOOK_CACHE: OrderedDict[tuple[int, int, str], Codebook] = OrderedDict()
_MAX_SCALED_CODEBOOK_CACHE = 256

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "codebooks"


def _compute_gaussian_codebook(bits: int) -> Codebook:
    """Compute the Lloyd-Max optimal codebook for N(0,1) at a given bit-width.

    Falls back to a closed-form/hardcoded solution for common bit-widths
    and uses iterative Lloyd-Max for higher bit-widths.
    """
    n_levels = 1 << bits

    # Hardcoded optimal centroids for N(0,1) from the paper / numerical computation
    known: dict[int, list[float]] = {
        1: [-0.7978845608, 0.7978845608],
        2: [-1.5104176088, -0.4527800398, 0.4527800398, 1.5104176088],
        3: [
            -2.1519742685, -1.3439092613, -0.7560052489, -0.2451209529,
            0.2451209529, 0.7560052489, 1.3439092613, 2.1519742685,
        ],
        4: [
            -2.7326368225, -2.0690571770, -1.6180378132, -1.2561836443,
            -0.9423401764, -0.6567588956, -0.3880170670, -0.1284042432,
            0.1284042432, 0.3880170670, 0.6567588956, 0.9423401764,
            1.2561836443, 1.6180378132, 2.0690571770, 2.7326368225,
        ],
    }

    if bits in known:
        centroids = torch.tensor(known[bits], dtype=torch.float64)
    else:
        centroids = _lloyd_max_gaussian(n_levels)

    boundaries = (centroids[:-1] + centroids[1:]) / 2.0

    # Compute MSE cost for N(0,1)
    mse = _compute_mse_cost(centroids.tolist(), boundaries.tolist())

    return Codebook(
        centroids=centroids.float(),
        boundaries=boundaries.float(),
        bits=bits,
        mse_cost=mse,
    )


def _lloyd_max_gaussian(n_levels: int, max_iter: int = 500, tol: float = 1e-14) -> torch.Tensor:
    """Iterative Lloyd-Max for N(0,1). Used for bit-widths > 4."""
    try:
        from scipy import integrate
        from scipy.stats import norm
    except ImportError as e:
        raise ImportError(
            "scipy is required for computing codebooks with bits > 4. "
            "Install with: pip install pyturboquant[dev]"
        ) from e

    # Initialize centroids uniformly over [-4, 4]
    centroids = [(-4.0 + (2 * i + 1) * 8.0 / n_levels) for i in range(n_levels)]

    for _ in range(max_iter):
        boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
        edges = [float("-inf"), *boundaries, float("inf")]

        new_centroids = []
        for i in range(n_levels):
            lo, hi = edges[i], edges[i + 1]
            num, _ = integrate.quad(lambda x: x * norm.pdf(x), lo, hi)
            den, _ = integrate.quad(lambda x: norm.pdf(x), lo, hi)
            new_centroids.append(num / den if den > 1e-30 else centroids[i])

        delta = max(abs(a - b) for a, b in zip(centroids, new_centroids, strict=True))
        centroids = new_centroids
        if delta < tol:
            break

    return torch.tensor(centroids, dtype=torch.float64)


def _compute_mse_cost(centroids: list[float], boundaries: list[float]) -> float:
    """Compute the MSE cost of a codebook against N(0,1)."""
    try:
        from scipy import integrate
        from scipy.stats import norm
    except ImportError:
        # Return a placeholder; exact MSE requires scipy
        return float("nan")

    n = len(centroids)
    edges = [float("-inf"), *list(boundaries), float("inf")]
    total_mse = 0.0
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        c = centroids[i]
        val, _ = integrate.quad(lambda x, _c=c: (x - _c) ** 2 * norm.pdf(x), lo, hi)
        total_mse += val
    return total_mse


def get_codebook(dim: int, bits: int, device: torch.device | None = None) -> Codebook:
    """Load or compute a codebook scaled for dimension d.

    Centroids and boundaries are scaled by 1/sqrt(d) to match the variance
    of coordinates on S^{d-1}.

    Args:
        dim: Vector dimension d.
        bits: Bit-width b (1-8).
        device: Target torch device.

    Returns:
        Codebook with centroids/boundaries scaled to dimension d.
    """
    if bits < 1 or bits > 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")

    if bits not in _CODEBOOK_CACHE:
        pt_path = _DATA_DIR / f"gaussian_b{bits}.pt"
        if pt_path.exists():
            data = torch.load(pt_path, weights_only=True)
            _CODEBOOK_CACHE[bits] = Codebook(
                centroids=data["centroids"].float(),
                boundaries=data["boundaries"].float(),
                bits=bits,
                mse_cost=float(data.get("mse_cost", float("nan"))),
            )
        else:
            _CODEBOOK_CACHE[bits] = _compute_gaussian_codebook(bits)

    dev = device or torch.device("cpu")
    scaled_key = (bits, dim, str(dev))
    if scaled_key in _SCALED_CODEBOOK_CACHE:
        _SCALED_CODEBOOK_CACHE.move_to_end(scaled_key)
        return _SCALED_CODEBOOK_CACHE[scaled_key]

    base = _CODEBOOK_CACHE[bits]
    scale = 1.0 / math.sqrt(dim)
    scaled = Codebook(
        centroids=(base.centroids * scale).to(dev),
        boundaries=(base.boundaries * scale).to(dev),
        bits=bits,
        mse_cost=base.mse_cost / dim,
    )
    _SCALED_CODEBOOK_CACHE[scaled_key] = scaled
    while len(_SCALED_CODEBOOK_CACHE) > _MAX_SCALED_CODEBOOK_CACHE:
        _SCALED_CODEBOOK_CACHE.popitem(last=False)
    return scaled
