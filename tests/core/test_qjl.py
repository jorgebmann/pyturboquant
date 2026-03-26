"""Tests for the QJL transform (Definition 1)."""

from __future__ import annotations

import torch

from pyturboquant.core.qjl import QJLTransform


class TestQJLTransform:
    """Tests for QJL sketch and inner product estimation."""

    def test_quantize_shape(self) -> None:
        d, m = 64, 64
        qjl = QJLTransform(dim=d, m=m, seed=0)
        g = torch.Generator().manual_seed(0)
        r = torch.randn(16, d, generator=g)
        packed = qjl.quantize(r)
        expected_bytes = (m + 7) // 8
        assert packed.shape == (16, expected_bytes)

    def test_unbiasedness(self) -> None:
        """E[<r, y>_est] should be close to true <r, y> over many trials."""
        d = 128
        n_trials = 50000
        g = torch.Generator().manual_seed(42)
        r = torch.randn(d, generator=g)
        r = r / torch.linalg.norm(r)
        y = torch.randn(d, generator=g)
        y = y / torch.linalg.norm(y)

        true_ip = (r * y).sum().item()

        estimates = []
        for trial in range(n_trials):
            qjl = QJLTransform(dim=d, seed=trial)
            packed = qjl.quantize(r.unsqueeze(0))
            r_norm = torch.linalg.norm(r).unsqueeze(0)
            est = qjl.estimate_inner_product(
                packed, y.unsqueeze(0), r_norm
            ).item()
            estimates.append(est)

        mean_est = sum(estimates) / len(estimates)
        # Unbiased: mean should be close to true IP
        assert abs(mean_est - true_ip) < 0.05, (
            f"Mean estimate {mean_est:.4f} vs true {true_ip:.4f}"
        )

    def test_variance_bound(self) -> None:
        """Variance of QJL estimator should scale roughly as pi/(2m) * ||r||^2 * ||y||^2."""
        d = 64
        m = 64
        n_trials = 10000
        g = torch.Generator().manual_seed(0)
        r = torch.randn(d, generator=g)
        y = torch.randn(d, generator=g)

        estimates = []
        for trial in range(n_trials):
            qjl = QJLTransform(dim=d, m=m, seed=trial)
            packed = qjl.quantize(r.unsqueeze(0))
            r_norm = torch.linalg.norm(r).unsqueeze(0)
            est = qjl.estimate_inner_product(
                packed, y.unsqueeze(0), r_norm
            ).item()
            estimates.append(est)

        t = torch.tensor(estimates)
        empirical_var = t.var().item()
        # Rough theoretical bound: should be O(||r||^2 * ||y||^2 / m)
        expected_order = (torch.linalg.norm(r).item() ** 2 *
                          torch.linalg.norm(y).item() ** 2)
        # Variance shouldn't blow up far beyond expected order
        assert empirical_var < expected_order * 5.0, (
            f"Variance {empirical_var:.4f} too large vs {expected_order:.4f}"
        )

    def test_deterministic(self) -> None:
        d = 32
        qjl = QJLTransform(dim=d, seed=42)
        g = torch.Generator().manual_seed(0)
        r = torch.randn(4, d, generator=g)
        p1 = qjl.quantize(r)
        p2 = qjl.quantize(r)
        torch.testing.assert_close(p1, p2)

    def test_batch_estimate(self) -> None:
        d = 64
        n = 16
        qjl = QJLTransform(dim=d, seed=0)
        g = torch.Generator().manual_seed(0)
        r = torch.randn(n, d, generator=g)
        y = torch.randn(d, generator=g)
        norms = torch.linalg.norm(r, dim=-1)
        safe_norms = norms.clamp(min=1e-12)
        r_normalized = r / safe_norms.unsqueeze(-1)
        packed = qjl.quantize(r_normalized)
        est_loop = qjl.estimate_inner_product(packed, y.unsqueeze(0).expand(n, -1), norms)
        est_batch = qjl.estimate_inner_product_batch(packed, y, norms)
        torch.testing.assert_close(est_loop, est_batch, atol=1e-5, rtol=1e-5)
