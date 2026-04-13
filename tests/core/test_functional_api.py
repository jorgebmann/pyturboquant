"""Tests for the top-level functional API in pyturboquant.core."""

from __future__ import annotations

import torch

from pyturboquant.core import (
    estimate_inner_product,
    ip_dequantize,
    ip_quantize,
    mse_dequantize,
    mse_quantize,
)


class TestFunctionalAPI:
    """Smoke tests for the functional wrappers."""

    def test_mse_round_trip(self) -> None:
        g = torch.Generator().manual_seed(0)
        x = torch.randn(16, 64, generator=g)
        qt = mse_quantize(x, bits=3, seed=0)
        x_hat = mse_dequantize(qt)
        assert x_hat.shape == x.shape
        # Should not be identical but should be correlated
        cos = torch.nn.functional.cosine_similarity(x, x_hat, dim=-1).mean()
        assert cos > 0.9

    def test_ip_round_trip(self) -> None:
        g = torch.Generator().manual_seed(0)
        x = torch.randn(8, 64, generator=g)
        qt = ip_quantize(x, bits=3, seed=0)
        x_hat = ip_dequantize(qt)
        assert x_hat.shape == x.shape

    def test_estimate_ip(self) -> None:
        g = torch.Generator().manual_seed(0)
        x = torch.randn(8, 64, generator=g)
        y = torch.randn(8, 64, generator=g)
        qt = ip_quantize(x, bits=4, seed=0)
        est = estimate_inner_product(qt, y)
        assert est.shape == (8,)
        true_ip = (x * y).sum(dim=-1)
        # Estimates should be in the right ballpark
        assert ((est - true_ip).abs() < true_ip.abs() * 2.0 + 5.0).all()

    def test_functional_cache_separates_devices(self) -> None:
        """Quantizer cache keys must include device (CPU vs CUDA)."""
        from pyturboquant.core import _IP_CACHE, _MSE_CACHE, ip_quantize, mse_quantize

        _MSE_CACHE.clear()
        _IP_CACHE.clear()
        g = torch.Generator().manual_seed(0)
        x_cpu = torch.randn(4, 16, generator=g)
        mse_quantize(x_cpu, bits=3, seed=0)
        assert len(_MSE_CACHE) == 1
        if torch.cuda.is_available():
            x_cuda = x_cpu.cuda()
            mse_quantize(x_cuda, bits=3, seed=0)
            assert len(_MSE_CACHE) == 2
            _IP_CACHE.clear()
            ip_quantize(x_cpu, bits=3, seed=0)
            ip_quantize(x_cuda, bits=3, seed=0)
            assert len(_IP_CACHE) == 2
