"""Inner Product TurboQuant quantizer (Algorithm 2).

Two-stage quantization: MSE quantize at (bits-1), then apply 1-bit QJL
to the residual.  This enables unbiased inner product estimation.
"""

from __future__ import annotations

import torch

from pyturboquant.core.mse_quantizer import MSEQuantizer
from pyturboquant.core.qjl import QJLTransform
from pyturboquant.core.types import QuantizedIP


class InnerProductQuantizer:
    """Two-stage inner-product-preserving quantizer.

    Stage 1: MSE-optimal quantization at (bits - 1) bits.
    Stage 2: 1-bit QJL transform on the quantization residual.

    The total budget is `bits` bits per coordinate: (bits-1) for MSE + 1 for QJL.

    Each ``quantize`` performs one random rotation and Lloyd--Max assignment for
    the MSE stage, reconstructs ``x_hat`` from centroids (no pack/unpack round
    trip for the residual), then runs QJL on the normalized residual. Cost is
    dominated by rotation and QJL matrix multiplies; batch large ``add`` calls
    when indexing many vectors.

    Args:
        dim: Vector dimension d.
        bits: Total bit budget per coordinate (>= 2).
        seed: Deterministic seed.
        device: Torch device.
    """

    def __init__(
        self,
        dim: int,
        bits: int = 4,
        seed: int = 0,
        device: torch.device | None = None,
    ) -> None:
        if bits < 2:
            raise ValueError(f"InnerProductQuantizer needs bits >= 2, got {bits}")
        self.dim = dim
        self.bits = bits
        self.seed = seed
        self.device = device or torch.device("cpu")
        self._mse = MSEQuantizer(dim, bits=bits - 1, seed=seed, device=self.device)
        self._qjl = QJLTransform(dim, seed=seed + 1, device=self.device)

    @property
    def mse_quantizer(self) -> MSEQuantizer:
        return self._mse

    @property
    def qjl_transform(self) -> QJLTransform:
        return self._qjl

    def quantize(self, x: torch.Tensor) -> QuantizedIP:
        """Quantize vectors using Algorithm 2 (Inner Product TurboQuant).

        Args:
            x: Input tensor of shape (*, d).

        Returns:
            QuantizedIP holding MSE data and QJL sign bits.
        """
        leading = x.shape[:-1]
        flat = x.reshape(-1, self.dim)

        mse_qt, x_hat = self._mse.quantize_with_reconstruction(flat)

        residual = flat - x_hat
        residual_norms = torch.linalg.norm(residual, dim=-1)
        safe_norms = residual_norms.clamp(min=1e-12)
        residual_normalized = residual / safe_norms.unsqueeze(-1)

        qjl_bits = self._qjl.quantize(residual_normalized)

        return QuantizedIP(
            mse_data=mse_qt,
            qjl_bits=qjl_bits,
            residual_norms=residual_norms.reshape(leading),
            dim=self.dim,
            bits=self.bits,
            seed=self.seed,
            device=self.device,
        )

    def dequantize(self, qt: QuantizedIP) -> torch.Tensor:
        """Reconstruct vectors (MSE component only; QJL is lossy).

        The QJL bits only aid inner-product estimation and cannot reconstruct
        the residual direction, so dequantize returns the MSE reconstruction.

        Args:
            qt: Quantized representation from quantize().

        Returns:
            Reconstructed tensor of shape matching the original input.
        """
        return self._mse.dequantize(qt.mse_data)

    def dequantize_range(
        self, qt: QuantizedIP, start: int, end: int
    ) -> torch.Tensor:
        """Reconstruct vectors in the index range ``[start, end)`` from a chunk.

        Thin wrapper around :meth:`MSEQuantizer.dequantize_range` -- only the
        MSE component is reconstructed. Used by ``TurboQuantIndex`` to bound
        peak fp32 memory during search independently of chunk size.

        Args:
            qt: Quantized representation from quantize().
            start: Inclusive start vector index.
            end: Exclusive end vector index.

        Returns:
            Reconstructed tensor of shape ``(end - start, dim)``.
        """
        return self._mse.dequantize_range(qt.mse_data, start, end)

    def estimate_inner_product(self, qt: QuantizedIP, y: torch.Tensor) -> torch.Tensor:
        """Estimate <x, y> from the quantized representation.

        Uses: <x, y> ≈ <x_hat_mse, y> + QJL_estimate(<r, y>)

        Args:
            qt: Quantized data for the database vectors.
            y: Query vectors of shape (*, d).

        Returns:
            Estimated inner products, shape (*,).
        """
        x_hat = self._mse.dequantize(qt.mse_data)
        flat_x_hat = x_hat.reshape(-1, self.dim)
        flat_y = y.reshape(-1, self.dim)

        mse_ip = (flat_x_hat * flat_y).sum(dim=-1)

        qjl_ip = self._qjl.estimate_inner_product(
            qt.qjl_bits.reshape(-1, qt.qjl_bits.shape[-1]),
            flat_y,
            qt.residual_norms.reshape(-1),
        )

        return (mse_ip + qjl_ip).reshape(qt.residual_norms.shape)
