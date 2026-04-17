"""Random rotation matrices for TurboQuant.

Generates d x d random orthogonal matrices via QR decomposition of
a Gaussian random matrix, ensuring uniform distribution over O(d).
"""

from __future__ import annotations

from collections import OrderedDict

import torch

_MAX_ROTATION_CACHE = 64
_ROTATION_CACHE: OrderedDict[tuple[int, int, str, str], RandomRotation] = OrderedDict()


def _rotation_cache_key(
    dim: int, seed: int, device: torch.device, dtype: torch.dtype
) -> tuple[int, int, str, str]:
    return (dim, seed, str(device), str(dtype))


def _get_cached_rotation(
    dim: int, seed: int, device: torch.device, dtype: torch.dtype
) -> RandomRotation:
    key = _rotation_cache_key(dim, seed, device, dtype)
    if key in _ROTATION_CACHE:
        _ROTATION_CACHE.move_to_end(key)
        return _ROTATION_CACHE[key]
    rot = RandomRotation(dim, seed=seed, device=device, dtype=dtype)
    _ROTATION_CACHE[key] = rot
    while len(_ROTATION_CACHE) > _MAX_ROTATION_CACHE:
        _ROTATION_CACHE.popitem(last=False)
    return rot


class RandomRotation:
    """Deterministic random orthogonal rotation seeded by an integer.

    After rotation, each coordinate of a unit vector on S^{d-1}
    follows a Beta distribution (Lemma 1), enabling per-coordinate
    scalar quantization.

    Args:
        dim: Vector dimension d.
        seed: Deterministic seed for reproducibility.
        device: Torch device.
        dtype: Torch dtype for the rotation matrix.
    """

    def __init__(
        self,
        dim: int,
        seed: int = 0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.dim = dim
        self.seed = seed
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self._matrix = self._generate()

    def _generate(self) -> torch.Tensor:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed)
        gauss = torch.randn(self.dim, self.dim, generator=g, dtype=self.dtype)
        q, r = torch.linalg.qr(gauss)
        # Ensure uniform Haar measure by fixing the sign of the diagonal of R
        diag_sign = torch.sign(torch.diag(r))
        diag_sign[diag_sign == 0] = 1.0
        q = q * diag_sign.unsqueeze(0)
        return q.to(device=self.device)

    @property
    def matrix(self) -> torch.Tensor:
        """The d x d orthogonal matrix Pi."""
        return self._matrix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotation: y = Pi @ x.

        Args:
            x: Tensor of shape (*, d).

        Returns:
            Rotated tensor of shape (*, d).
        """
        return x @ self._matrix.T

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Apply inverse rotation: x = Pi^T @ y.

        Args:
            y: Tensor of shape (*, d).

        Returns:
            Tensor of shape (*, d).
        """
        return y @ self._matrix


def random_rotate(x: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Functional API: apply a seeded random rotation to x.

    Args:
        x: Tensor of shape (*, d).
        seed: Deterministic seed.

    Returns:
        Rotated tensor of shape (*, d).
    """
    rot = _get_cached_rotation(
        dim=x.shape[-1], seed=seed, device=x.device, dtype=x.dtype
    )
    return rot.forward(x)


def random_rotate_inverse(y: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Functional API: apply the inverse of a seeded random rotation.

    Args:
        y: Tensor of shape (*, d).
        seed: Must match the seed used in random_rotate.

    Returns:
        Original tensor of shape (*, d).
    """
    rot = _get_cached_rotation(
        dim=y.shape[-1], seed=seed, device=y.device, dtype=y.dtype
    )
    return rot.inverse(y)
