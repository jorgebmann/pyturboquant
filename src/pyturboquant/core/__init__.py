"""pyturboquant.core -- Pure PyTorch building blocks for TurboQuant quantization.

Public API re-exports and convenience functional wrappers.
"""

from __future__ import annotations

import torch

from pyturboquant.core.mse_quantizer import MSEQuantizer
from pyturboquant.core.prod_quantizer import InnerProductQuantizer
from pyturboquant.core.qjl import QJLTransform
from pyturboquant.core.rotation import RandomRotation, random_rotate, random_rotate_inverse
from pyturboquant.core.types import Codebook, QuantizedIP, QuantizedMSE

__all__ = [
    "Codebook",
    "InnerProductQuantizer",
    "MSEQuantizer",
    "QJLTransform",
    "QuantizedIP",
    "QuantizedMSE",
    "RandomRotation",
    "estimate_inner_product",
    "ip_dequantize",
    "ip_quantize",
    "mse_dequantize",
    "mse_quantize",
    "random_rotate",
    "random_rotate_inverse",
]

_MSE_CACHE: dict[tuple[int, int, int, str], MSEQuantizer] = {}
_IP_CACHE: dict[tuple[int, int, int, str], InnerProductQuantizer] = {}


def _get_mse(dim: int, bits: int, seed: int, device: torch.device) -> MSEQuantizer:
    key = (dim, bits, seed, str(device))
    if key not in _MSE_CACHE:
        _MSE_CACHE[key] = MSEQuantizer(dim, bits, seed, device)
    return _MSE_CACHE[key]


def _get_ip(dim: int, bits: int, seed: int, device: torch.device) -> InnerProductQuantizer:
    key = (dim, bits, seed, str(device))
    if key not in _IP_CACHE:
        _IP_CACHE[key] = InnerProductQuantizer(dim, bits, seed, device)
    return _IP_CACHE[key]


def mse_quantize(
    x: torch.Tensor, bits: int = 4, seed: int = 0
) -> QuantizedMSE:
    """Functional API: MSE-optimal quantization (Algorithm 1).

    Args:
        x: Input tensor of shape (*, d).
        bits: Bit-width (1-8).
        seed: Deterministic seed.

    Returns:
        QuantizedMSE representation.
    """
    q = _get_mse(x.shape[-1], bits, seed, x.device)
    return q.quantize(x)


def mse_dequantize(qt: QuantizedMSE) -> torch.Tensor:
    """Functional API: reconstruct from MSE-quantized representation.

    Args:
        qt: QuantizedMSE from mse_quantize().

    Returns:
        Reconstructed tensor.
    """
    q = _get_mse(qt.dim, qt.bits, qt.seed, qt.device)
    return q.dequantize(qt)


def ip_quantize(
    x: torch.Tensor, bits: int = 4, seed: int = 0
) -> QuantizedIP:
    """Functional API: inner-product-preserving quantization (Algorithm 2).

    Args:
        x: Input tensor of shape (*, d).
        bits: Total bit budget (>= 2).
        seed: Deterministic seed.

    Returns:
        QuantizedIP representation.
    """
    q = _get_ip(x.shape[-1], bits, seed, x.device)
    return q.quantize(x)


def ip_dequantize(qt: QuantizedIP) -> torch.Tensor:
    """Functional API: reconstruct from IP-quantized representation (MSE component).

    Args:
        qt: QuantizedIP from ip_quantize().

    Returns:
        Reconstructed tensor.
    """
    q = _get_ip(qt.dim, qt.bits, qt.seed, qt.device)
    return q.dequantize(qt)


def estimate_inner_product(qt: QuantizedIP, y: torch.Tensor) -> torch.Tensor:
    """Functional API: estimate <x, y> from quantized representation.

    Args:
        qt: QuantizedIP from ip_quantize().
        y: Query vectors of shape (*, d).

    Returns:
        Estimated inner products.
    """
    q = _get_ip(qt.dim, qt.bits, qt.seed, qt.device)
    return q.estimate_inner_product(qt, y)
