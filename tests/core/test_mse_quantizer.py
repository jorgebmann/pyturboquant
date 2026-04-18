"""Tests for the MSE quantizer (Algorithm 1)."""

from __future__ import annotations

import pytest
import torch

from pyturboquant.core.mse_quantizer import MSEQuantizer
from pyturboquant.utils.metrics import mse_distortion, shannon_lower_bound

THEOREM1_BOUNDS = {
    1: 0.3634,
    2: 0.1175,
    3: 0.03454,
    4: 0.009497,
}


class TestMSEQuantizer:
    """Tests for MSEQuantizer round-trip and correctness."""

    @pytest.mark.parametrize("d", [64, 128, 256])
    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_round_trip_shape(self, d: int, bits: int) -> None:
        q = MSEQuantizer(dim=d, bits=bits, seed=0)
        g = torch.Generator().manual_seed(42)
        x = torch.randn(32, d, generator=g)
        qt = q.quantize(x)
        x_hat = q.dequantize(qt)
        assert x_hat.shape == x.shape

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_norms_preserved(self, bits: int) -> None:
        d = 128
        q = MSEQuantizer(dim=d, bits=bits, seed=0)
        g = torch.Generator().manual_seed(42)
        x = torch.randn(64, d, generator=g)
        qt = q.quantize(x)
        expected_norms = torch.linalg.norm(x, dim=-1, keepdim=True)
        torch.testing.assert_close(qt.norms, expected_norms, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_distortion_below_theorem1(self, bits: int) -> None:
        """Empirical normalized MSE should be close to paper's Theorem 1 values.

        For unit vectors: E[||x - Q(x)||^2] ≈ MSE_cost (per-coordinate MSE
        summed over d coordinates, each with variance 1/d, giving total ≈ MSE_cost).
        """
        d = 256
        n = 10000
        q = MSEQuantizer(dim=d, bits=bits, seed=0)
        g = torch.Generator().manual_seed(99)
        x = torch.randn(n, d, generator=g)
        x = x / torch.linalg.norm(x, dim=-1, keepdim=True)
        qt = q.quantize(x)
        x_hat = q.dequantize(qt)
        empirical = mse_distortion(x, x_hat).item()
        bound = THEOREM1_BOUNDS[bits] * 1.3
        assert empirical < bound, (
            f"b={bits}: empirical MSE {empirical:.6f} > bound {bound:.6f}"
        )

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_distortion_above_shannon(self, bits: int) -> None:
        """Distortion should be above the Shannon lower bound."""
        d = 256
        n = 5000
        q = MSEQuantizer(dim=d, bits=bits, seed=0)
        g = torch.Generator().manual_seed(0)
        x = torch.randn(n, d, generator=g)
        x = x / torch.linalg.norm(x, dim=-1, keepdim=True)
        qt = q.quantize(x)
        x_hat = q.dequantize(qt)
        empirical = mse_distortion(x, x_hat).item()
        lb = shannon_lower_bound(bits)
        assert empirical >= lb * 0.5, (
            f"b={bits}: empirical MSE {empirical:.6f} way below Shannon bound {lb:.6f}"
        )

    def test_wrong_dim_raises(self) -> None:
        q = MSEQuantizer(dim=64, bits=2, seed=0)
        x = torch.randn(10, 32)
        with pytest.raises(ValueError, match="Expected dim"):
            q.quantize(x)

    def test_batch_dimensions(self) -> None:
        d = 64
        q = MSEQuantizer(dim=d, bits=2, seed=0)
        g = torch.Generator().manual_seed(0)
        x = torch.randn(4, 8, d, generator=g)
        qt = q.quantize(x)
        x_hat = q.dequantize(qt)
        assert x_hat.shape == x.shape

    def test_deterministic(self) -> None:
        d = 64
        q = MSEQuantizer(dim=d, bits=3, seed=42)
        g = torch.Generator().manual_seed(0)
        x = torch.randn(16, d, generator=g)
        qt1 = q.quantize(x)
        qt2 = q.quantize(x)
        torch.testing.assert_close(qt1.packed_indices, qt2.packed_indices)

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_dequantize_range_matches_slice(self, bits: int) -> None:
        """Byte-aligned range dequantize must match a slice of the full dequantize."""
        d = 64
        q = MSEQuantizer(dim=d, bits=bits, seed=0)
        g = torch.Generator().manual_seed(0)
        x = torch.randn(32, d, generator=g)
        qt = q.quantize(x)
        full = q.dequantize(qt)
        sub = q.dequantize_range(qt, 8, 24)
        torch.testing.assert_close(sub, full[8:24], atol=1e-6, rtol=1e-6)

    def test_dequantize_range_unaligned_dim(self) -> None:
        """Fallback path (dim * bits not divisible by 8) still produces correct slice."""
        d = 100
        bits = 3
        q = MSEQuantizer(dim=d, bits=bits, seed=0)
        g = torch.Generator().manual_seed(0)
        x = torch.randn(20, d, generator=g)
        qt = q.quantize(x)
        full = q.dequantize(qt)
        sub = q.dequantize_range(qt, 5, 15)
        torch.testing.assert_close(sub, full[5:15], atol=1e-6, rtol=1e-6)

    def test_dequantize_range_empty_slice(self) -> None:
        d = 64
        q = MSEQuantizer(dim=d, bits=3, seed=0)
        g = torch.Generator().manual_seed(0)
        x = torch.randn(10, d, generator=g)
        qt = q.quantize(x)
        sub = q.dequantize_range(qt, 5, 5)
        assert sub.shape == (0, d)

    def test_dequantize_range_full_span(self) -> None:
        """Range covering the entire chunk must equal the full dequantize."""
        d = 64
        q = MSEQuantizer(dim=d, bits=4, seed=0)
        g = torch.Generator().manual_seed(0)
        x = torch.randn(16, d, generator=g)
        qt = q.quantize(x)
        full = q.dequantize(qt)
        sub = q.dequantize_range(qt, 0, 16)
        torch.testing.assert_close(sub, full, atol=1e-6, rtol=1e-6)

    def test_dequantize_range_invalid_bounds(self) -> None:
        d = 64
        q = MSEQuantizer(dim=d, bits=3, seed=0)
        g = torch.Generator().manual_seed(0)
        x = torch.randn(10, d, generator=g)
        qt = q.quantize(x)
        with pytest.raises(ValueError, match="Invalid range"):
            q.dequantize_range(qt, -1, 5)
        with pytest.raises(ValueError, match="Invalid range"):
            q.dequantize_range(qt, 5, 20)
        with pytest.raises(ValueError, match="Invalid range"):
            q.dequantize_range(qt, 6, 4)
