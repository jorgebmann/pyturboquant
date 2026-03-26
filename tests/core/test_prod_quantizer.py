"""Tests for the Inner Product quantizer (Algorithm 2)."""

from __future__ import annotations

import pytest
import torch

from pyturboquant.core.prod_quantizer import InnerProductQuantizer


class TestInnerProductQuantizer:
    """Tests for InnerProductQuantizer."""

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_round_trip_shape(self, bits: int) -> None:
        d = 64
        q = InnerProductQuantizer(dim=d, bits=bits, seed=0)
        g = torch.Generator().manual_seed(0)
        x = torch.randn(16, d, generator=g)
        qt = q.quantize(x)
        x_hat = q.dequantize(qt)
        assert x_hat.shape == x.shape

    def test_bits_too_low_raises(self) -> None:
        with pytest.raises(ValueError, match="bits >= 2"):
            InnerProductQuantizer(dim=64, bits=1)

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_ip_estimation_unbiased(self, bits: int) -> None:
        """Mean estimated IP should be close to true IP over many seeds."""
        d = 128
        n_seeds = 200
        g = torch.Generator().manual_seed(42)
        x = torch.randn(d, generator=g)
        y = torch.randn(d, generator=g)
        true_ip = (x * y).sum().item()

        estimates = []
        for seed in range(n_seeds):
            q = InnerProductQuantizer(dim=d, bits=bits, seed=seed)
            qt = q.quantize(x.unsqueeze(0))
            est = q.estimate_inner_product(qt, y.unsqueeze(0)).item()
            estimates.append(est)

        mean_est = sum(estimates) / len(estimates)
        # With 200 seeds the mean should converge reasonably
        tolerance = abs(true_ip) * 0.25 + 1.0
        assert abs(mean_est - true_ip) < tolerance, (
            f"b={bits}: mean IP est {mean_est:.4f} vs true {true_ip:.4f}"
        )

    def test_dequantize_is_mse_only(self) -> None:
        """Dequantize should return MSE reconstruction (no QJL residual)."""
        d = 64
        from pyturboquant.core.mse_quantizer import MSEQuantizer
        ip_q = InnerProductQuantizer(dim=d, bits=3, seed=0)
        mse_q = MSEQuantizer(dim=d, bits=2, seed=0)

        g = torch.Generator().manual_seed(0)
        x = torch.randn(8, d, generator=g)

        qt_ip = ip_q.quantize(x)
        x_hat_ip = ip_q.dequantize(qt_ip)

        qt_mse = mse_q.quantize(x)
        x_hat_mse = mse_q.dequantize(qt_mse)

        torch.testing.assert_close(x_hat_ip, x_hat_mse, atol=1e-5, rtol=1e-5)

    def test_deterministic(self) -> None:
        d = 64
        q = InnerProductQuantizer(dim=d, bits=3, seed=42)
        g = torch.Generator().manual_seed(0)
        x = torch.randn(8, d, generator=g)
        qt1 = q.quantize(x)
        qt2 = q.quantize(x)
        torch.testing.assert_close(qt1.qjl_bits, qt2.qjl_bits)
        torch.testing.assert_close(qt1.residual_norms, qt2.residual_norms)
