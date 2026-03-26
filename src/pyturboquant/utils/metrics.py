"""Distortion and quality metrics for quantizer evaluation."""

from __future__ import annotations

import torch


def mse_distortion(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Mean squared error between original and reconstructed vectors.

    Computes: mean(||x_i - x_hat_i||^2 / ||x_i||^2) over the batch,
    i.e. the normalized MSE distortion as defined in the paper.

    Args:
        x: Original vectors of shape (n, d).
        x_hat: Reconstructed vectors of shape (n, d).

    Returns:
        Scalar tensor with the mean normalized MSE distortion.
    """
    sq_err = ((x - x_hat) ** 2).sum(dim=-1)
    sq_norm = (x**2).sum(dim=-1).clamp(min=1e-12)
    return (sq_err / sq_norm).mean()


def inner_product_error(
    x: torch.Tensor, y: torch.Tensor, x_hat: torch.Tensor
) -> torch.Tensor:
    """Mean absolute error of inner product estimation.

    Computes: mean(|<x_i, y_i> - <x_hat_i, y_i>|) over the batch.

    Args:
        x: Original vectors of shape (n, d).
        y: Query vectors of shape (n, d).
        x_hat: Reconstructed vectors of shape (n, d).

    Returns:
        Scalar tensor with the mean absolute inner-product error.
    """
    true_ip = (x * y).sum(dim=-1)
    est_ip = (x_hat * y).sum(dim=-1)
    return (true_ip - est_ip).abs().mean()


def shannon_lower_bound(bits: int) -> float:
    """Shannon rate-distortion lower bound: D >= 1/4^b per coordinate.

    For d-dimensional vectors, the total normalized MSE lower bound is 1/(d * 4^b).
    This function returns the per-coordinate value 1/4^b.

    Args:
        bits: Bit-width b.

    Returns:
        Lower bound on MSE per coordinate.
    """
    return 1.0 / (4**bits)
