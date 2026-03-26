"""Tests for Lloyd-Max codebooks."""

from __future__ import annotations

import pytest
import torch

from pyturboquant.core.codebook import _compute_gaussian_codebook, get_codebook

PAPER_MSE_VALUES = {
    1: 0.3634,
    2: 0.1175,
    3: 0.03454,
    4: 0.009497,
}


class TestGaussianCodebook:
    """Tests for raw N(0,1) codebook computation."""

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_centroids_sorted(self, bits: int) -> None:
        cb = _compute_gaussian_codebook(bits)
        diffs = cb.centroids[1:] - cb.centroids[:-1]
        assert (diffs > 0).all(), "Centroids must be strictly ascending"

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_boundaries_between_centroids(self, bits: int) -> None:
        cb = _compute_gaussian_codebook(bits)
        for i in range(len(cb.boundaries)):
            assert cb.boundaries[i] > cb.centroids[i]
            assert cb.boundaries[i] < cb.centroids[i + 1]

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_mse_matches_paper(self, bits: int) -> None:
        cb = _compute_gaussian_codebook(bits)
        expected = PAPER_MSE_VALUES[bits]
        assert abs(cb.mse_cost - expected) / expected < 0.05, (
            f"b={bits}: MSE {cb.mse_cost:.6f} vs expected {expected:.6f} (>5% off)"
        )

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_symmetry(self, bits: int) -> None:
        cb = _compute_gaussian_codebook(bits)
        n = len(cb.centroids)
        for i in range(n // 2):
            torch.testing.assert_close(
                cb.centroids[i], -cb.centroids[n - 1 - i], atol=1e-5, rtol=1e-5
            )


class TestGetCodebook:
    """Tests for the scaled codebook loader."""

    @pytest.mark.parametrize("dim", [64, 128, 256])
    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_scaling(self, dim: int, bits: int) -> None:
        cb = get_codebook(dim=dim, bits=bits)
        cb_base = _compute_gaussian_codebook(bits)
        import math
        scale = 1.0 / math.sqrt(dim)
        torch.testing.assert_close(
            cb.centroids, cb_base.centroids * scale, atol=1e-5, rtol=1e-5
        )

    def test_invalid_bits_raises(self) -> None:
        with pytest.raises(ValueError, match="bits must be"):
            get_codebook(dim=64, bits=0)
        with pytest.raises(ValueError, match="bits must be"):
            get_codebook(dim=64, bits=9)

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_caching(self, bits: int) -> None:
        cb1 = get_codebook(dim=64, bits=bits)
        cb2 = get_codebook(dim=64, bits=bits)
        torch.testing.assert_close(cb1.centroids, cb2.centroids)
