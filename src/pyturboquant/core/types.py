"""Dataclass types for quantized representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch


class Codebook(NamedTuple):
    """Precomputed Lloyd-Max codebook for a given bit-width."""

    centroids: torch.Tensor  # shape (2^b,), sorted ascending
    boundaries: torch.Tensor  # shape (2^b - 1,), midpoints between centroids
    bits: int
    mse_cost: float  # optimal MSE per coordinate for this codebook


@dataclass
class QuantizedMSE:
    """Quantized representation from TurboQuant_mse (Algorithm 1)."""

    packed_indices: torch.Tensor  # bit-packed centroid indices
    norms: torch.Tensor  # original L2 norms, shape (*, 1) or scalar
    dim: int
    bits: int
    seed: int
    device: torch.device


@dataclass
class QuantizedIP:
    """Quantized representation from TurboQuant_prod (Algorithm 2)."""

    mse_data: QuantizedMSE  # MSE quantization at (bits - 1)
    qjl_bits: torch.Tensor  # packed sign bits from QJL, shape (*, d_packed)
    residual_norms: torch.Tensor  # ||r||_2, shape (*,)
    dim: int
    bits: int
    seed: int
    device: torch.device
