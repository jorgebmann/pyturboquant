"""MSE-optimal TurboQuant quantizer (Algorithm 1 of the paper).

Performs: normalize -> random rotate -> per-coordinate scalar quantize -> pack.
Dequantize reverses the process.
"""

from __future__ import annotations

import torch

from pyturboquant.core.codebook import get_codebook
from pyturboquant.core.packed import pack_indices, unpack_indices
from pyturboquant.core.rotation import RandomRotation
from pyturboquant.core.types import Codebook, QuantizedMSE


class MSEQuantizer:
    """Data-oblivious MSE-optimal vector quantizer.

    Applies a seeded random rotation then per-coordinate Lloyd-Max quantization
    at the given bit-width.  The distortion approaches the Shannon lower bound
    1/(d * 4^b) for large d (Theorem 1).

    Args:
        dim: Vector dimension d.
        bits: Bit-width b (1-8).
        seed: Deterministic seed for the rotation matrix.
        device: Torch device.
    """

    def __init__(
        self,
        dim: int,
        bits: int,
        seed: int = 0,
        device: torch.device | None = None,
    ) -> None:
        self.dim = dim
        self.bits = bits
        self.seed = seed
        self.device = device or torch.device("cpu")
        self._rotation = RandomRotation(dim, seed=seed, device=self.device)
        self._codebook: Codebook = get_codebook(dim, bits, device=self.device)

    @property
    def codebook(self) -> Codebook:
        return self._codebook

    def quantize(self, x: torch.Tensor) -> QuantizedMSE:
        """Quantize vectors using Algorithm 1 (MSE TurboQuant).

        Args:
            x: Input tensor of shape (*, d).

        Returns:
            QuantizedMSE holding packed indices and norms.
        """
        leading = x.shape[:-1]
        d = x.shape[-1]
        if d != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {d}")

        flat = x.reshape(-1, d)
        norms = torch.linalg.norm(flat, dim=-1, keepdim=True)
        safe_norms = norms.clamp(min=1e-12)
        normalized = flat / safe_norms

        rotated = self._rotation.forward(normalized)

        boundaries = self._codebook.boundaries
        indices = torch.searchsorted(boundaries, rotated)
        indices = indices.clamp(0, (1 << self.bits) - 1).to(torch.int32)

        packed = pack_indices(indices.reshape(-1), self.bits)

        return QuantizedMSE(
            packed_indices=packed,
            norms=norms.reshape(*leading, 1),
            dim=self.dim,
            bits=self.bits,
            seed=self.seed,
            device=self.device,
        )

    def quantize_with_reconstruction(
        self, x: torch.Tensor
    ) -> tuple[QuantizedMSE, torch.Tensor]:
        """Same as ``quantize``, but also returns the MSE reconstruction without pack/unpack.

        Avoids an extra dequantize pass when the caller needs ``x_hat`` immediately
        (for example inner-product quantization of the residual).
        """
        leading = x.shape[:-1]
        d = x.shape[-1]
        if d != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {d}")

        flat = x.reshape(-1, d)
        norms = torch.linalg.norm(flat, dim=-1, keepdim=True)
        safe_norms = norms.clamp(min=1e-12)
        normalized = flat / safe_norms

        rotated = self._rotation.forward(normalized)

        boundaries = self._codebook.boundaries
        indices = torch.searchsorted(boundaries, rotated)
        indices = indices.clamp(0, (1 << self.bits) - 1).to(torch.int32)

        centroids = self._codebook.centroids[indices]
        rotated_back = self._rotation.inverse(centroids)
        x_hat = (rotated_back * safe_norms).reshape(*leading, d)

        packed = pack_indices(indices.reshape(-1), self.bits)
        qt = QuantizedMSE(
            packed_indices=packed,
            norms=norms.reshape(*leading, 1),
            dim=self.dim,
            bits=self.bits,
            seed=self.seed,
            device=self.device,
        )
        return qt, x_hat

    def dequantize(self, qt: QuantizedMSE) -> torch.Tensor:
        """Reconstruct vectors from quantized representation.

        Args:
            qt: Quantized representation from quantize().

        Returns:
            Reconstructed tensor of shape matching the original input.
        """
        n_vectors = qt.norms.reshape(-1).shape[0]
        count = n_vectors * self.dim

        indices = unpack_indices(qt.packed_indices, qt.bits, count)
        centroids = self._codebook.centroids
        reconstructed = centroids[indices.long()].reshape(n_vectors, self.dim)

        rotated_back = self._rotation.inverse(reconstructed)

        norms_flat = qt.norms.reshape(n_vectors, 1)
        return (rotated_back * norms_flat).reshape(*qt.norms.shape[:-1], self.dim)
