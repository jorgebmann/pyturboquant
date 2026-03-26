"""Beta distribution of coordinates on the unit hypersphere (Lemma 1 of TurboQuant paper).

For a random point uniformly distributed on S^{d-1}, each coordinate follows:
    f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)
which converges to N(0, 1/d) as d -> infinity.
"""

from __future__ import annotations

import math

import torch


def sphere_coordinate_pdf(x: torch.Tensor, d: int) -> torch.Tensor:
    """Evaluate the PDF of a single coordinate of a uniform random point on S^{d-1}.

    Args:
        x: Points in [-1, 1] at which to evaluate the PDF.
        d: Ambient dimension (must be >= 3).

    Returns:
        PDF values, same shape as x.
    """
    if d < 3:
        raise ValueError(f"Dimension must be >= 3, got {d}")
    log_norm = (
        math.lgamma(d / 2) - 0.5 * math.log(math.pi) - math.lgamma((d - 1) / 2)
    )
    exponent = (d - 3) / 2
    log_body = exponent * torch.log1p(-x * x)
    return torch.exp(log_norm + log_body)


def sphere_coordinate_pdf_numpy(x: float, d: int) -> float:
    """Scalar version for scipy integration during codebook precomputation."""
    if abs(x) >= 1.0:
        return 0.0
    log_norm = (
        math.lgamma(d / 2) - 0.5 * math.log(math.pi) - math.lgamma((d - 1) / 2)
    )
    exponent = (d - 3) / 2
    log_body = exponent * math.log(1.0 - x * x)
    return math.exp(log_norm + log_body)


def sphere_coordinate_variance(d: int) -> float:
    """Variance of a single coordinate on S^{d-1}, equals 1/d."""
    return 1.0 / d


def gaussian_pdf(x: torch.Tensor, variance: float = 1.0) -> torch.Tensor:
    """Standard Gaussian PDF N(0, variance)."""
    return torch.exp(-x * x / (2 * variance)) / math.sqrt(2 * math.pi * variance)


def gaussian_pdf_numpy(x: float, variance: float = 1.0) -> float:
    """Scalar Gaussian PDF for scipy integration."""
    return math.exp(-x * x / (2 * variance)) / math.sqrt(2 * math.pi * variance)
