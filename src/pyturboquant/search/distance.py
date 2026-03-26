"""Asymmetric distance computations for quantized vector search.

The query is kept in full precision while database vectors are quantized,
enabling fast approximate distance computation.
"""

from __future__ import annotations

import torch

from pyturboquant.core.mse_quantizer import MSEQuantizer
from pyturboquant.core.qjl import QJLTransform
from pyturboquant.core.types import QuantizedIP


def asymmetric_inner_product(
    query: torch.Tensor,
    qt_list: list[QuantizedIP],
    mse_quantizer: MSEQuantizer,
    qjl_transform: QJLTransform,
) -> torch.Tensor:
    """Compute asymmetric inner products between a query and quantized database.

    <x_i, q> ≈ <x_hat_mse_i, q> + QJL_estimate(<r_i, q>)

    Args:
        query: Query vector of shape (d,).
        qt_list: List of QuantizedIP entries (one per database vector or batch).
        mse_quantizer: The MSEQuantizer used for the database.
        qjl_transform: The QJLTransform used for the database.

    Returns:
        Inner product estimates, shape (n,).
    """
    scores = []
    for qt in qt_list:
        x_hat = mse_quantizer.dequantize(qt.mse_data)
        flat_x_hat = x_hat.reshape(-1, query.shape[-1])
        mse_ip = (flat_x_hat * query.unsqueeze(0)).sum(dim=-1)

        qjl_ip = qjl_transform.estimate_inner_product_batch(
            qt.qjl_bits.reshape(-1, qt.qjl_bits.shape[-1]),
            query,
            qt.residual_norms.reshape(-1),
        )
        scores.append(mse_ip + qjl_ip)
    return torch.cat(scores)


def asymmetric_l2(
    query: torch.Tensor,
    qt_list: list[QuantizedIP],
    mse_quantizer: MSEQuantizer,
    norms_sq: torch.Tensor,
) -> torch.Tensor:
    """Compute asymmetric L2 distances using the identity:
    ||x - q||^2 = ||x||^2 - 2<x, q> + ||q||^2

    For ranking, only the IP term matters (norms are constant per db vector).

    Args:
        query: Query vector of shape (d,).
        qt_list: List of QuantizedIP entries.
        mse_quantizer: MSEQuantizer for the database.
        norms_sq: Precomputed ||x_i||^2 for each database vector, shape (n,).

    Returns:
        Approximate L2 distances, shape (n,).
    """
    ip_scores = []
    for qt in qt_list:
        x_hat = mse_quantizer.dequantize(qt.mse_data)
        flat_x_hat = x_hat.reshape(-1, query.shape[-1])
        ip_scores.append((flat_x_hat * query.unsqueeze(0)).sum(dim=-1))
    ip = torch.cat(ip_scores)
    q_norm_sq = (query * query).sum()
    return norms_sq - 2.0 * ip + q_norm_sq
