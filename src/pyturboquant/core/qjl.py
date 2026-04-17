"""Quantized Johnson-Lindenstrauss (QJL) transform (Definition 1).

Projects vectors via a random Gaussian matrix and takes the sign,
producing 1-bit sketches that allow unbiased inner product estimation.
"""

from __future__ import annotations

import math

import torch

from pyturboquant.core.packed import pack_bits_batch, unpack_bits_batch


class QJLTransform:
    """1-bit Quantized Johnson-Lindenstrauss transform.

    Given a random Gaussian matrix S of shape (m, d), the QJL sketch of
    a vector r is sign(S @ r).  The inner product <x, y> can be estimated
    unbiasedly from the sketches.

    Args:
        dim: Input dimension d.
        m: Number of projection rows (sketch dimension). Defaults to dim.
        seed: Deterministic seed for the projection matrix.
        device: Torch device.
    """

    def __init__(
        self,
        dim: int,
        m: int | None = None,
        seed: int = 0,
        device: torch.device | None = None,
    ) -> None:
        self.dim = dim
        self.m = m or dim
        self.seed = seed
        self.device = device or torch.device("cpu")
        self._S = self._generate()

    def _generate(self) -> torch.Tensor:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed)
        S = torch.randn(self.m, self.dim, generator=g, dtype=torch.float32)
        return S.to(device=self.device)

    @property
    def projection_matrix(self) -> torch.Tensor:
        return self._S

    def quantize(self, r: torch.Tensor) -> torch.Tensor:
        """Compute the QJL sketch: packed sign bits of S @ r.

        Args:
            r: Residual vectors of shape (*, d).

        Returns:
            Packed uint8 tensor of sign bits.
        """
        leading = r.shape[:-1]
        flat = r.reshape(-1, self.dim)  # (n, d)
        proj = flat @ self._S.T  # (n, m)
        signs = (proj >= 0).to(torch.uint8)
        packed = pack_bits_batch(signs)
        return packed.reshape(*leading, -1)

    def estimate_inner_product(
        self,
        packed_z: torch.Tensor,
        y: torch.Tensor,
        residual_norms: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate <r, y> from the QJL sketch.

        Uses the unbiased estimator:
            <r, y>_est = sqrt(pi/2) / m * ||r|| * sum_j z_j * (S_j . y)

        where z_j ∈ {-1, +1} are the unpacked sign bits.

        Args:
            packed_z: Packed sign bits from quantize(), shape (*, n_bytes).
            y: Query vectors of shape (*, d).
            residual_norms: L2 norms ||r||, shape (*,).

        Returns:
            Estimated inner products, shape (*,).
        """
        leading = packed_z.shape[:-1]
        flat_z = packed_z.reshape(-1, packed_z.shape[-1])
        flat_y = y.reshape(-1, self.dim)
        flat_norms = residual_norms.reshape(-1)
        n = flat_z.shape[0]

        Sy = flat_y @ self._S.T  # (n, m)
        signs = unpack_bits_batch(flat_z, self.m)
        scale = math.sqrt(math.pi / 2.0) / self.m
        ip = (signs * Sy).sum(dim=-1)
        return (scale * flat_norms * ip).reshape(leading)

    def estimate_inner_product_batch(
        self,
        packed_z: torch.Tensor,
        y: torch.Tensor,
        residual_norms: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized version for batch estimation (query y against n database entries).

        Args:
            packed_z: Packed sign bits, shape (n, n_bytes).
            y: Single query vector of shape (d,).
            residual_norms: Norms of shape (n,).

        Returns:
            Estimated inner products, shape (n,).
        """
        n = packed_z.shape[0]
        signs = unpack_bits_batch(packed_z, self.m)
        Sy = y @ self._S.T  # (m,)
        scale = math.sqrt(math.pi / 2.0) / self.m
        return scale * residual_norms * (signs * Sy.unsqueeze(0)).sum(dim=-1)

    def estimate_inner_product_batch_queries(
        self,
        packed_z: torch.Tensor,
        queries: torch.Tensor,
        residual_norms: torch.Tensor,
    ) -> torch.Tensor:
        """All-queries variant: ``queries`` of shape (nq, d), database rows ``n``.

        Returns:
            Tensor of shape (nq, n) with ``[j, i]`` = estimate for DB row i vs query j.
        """
        signs = unpack_bits_batch(packed_z, self.m)  # (n, m)
        Sy = queries @ self._S.T  # (nq, m)
        scale = math.sqrt(math.pi / 2.0) / self.m
        ip = signs @ Sy.T  # (n, nq)
        return scale * residual_norms.unsqueeze(0) * ip.T
