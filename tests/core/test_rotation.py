"""Tests for random rotation matrices (Lemma 1 prerequisites)."""

from __future__ import annotations

import pytest
import torch

from pyturboquant.core.rotation import RandomRotation, random_rotate, random_rotate_inverse


class TestRandomRotation:
    """Tests for the RandomRotation class."""

    @pytest.mark.parametrize("d", [16, 64, 128, 256])
    def test_orthogonality(self, d: int) -> None:
        rot = RandomRotation(dim=d, seed=0)
        eye = torch.eye(d)
        product = rot.matrix.T @ rot.matrix
        torch.testing.assert_close(product, eye, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("d", [16, 64, 128, 256])
    def test_norm_preservation(self, d: int) -> None:
        rot = RandomRotation(dim=d, seed=42)
        g = torch.Generator().manual_seed(99)
        x = torch.randn(32, d, generator=g)
        y = rot.forward(x)
        norms_x = torch.linalg.norm(x, dim=-1)
        norms_y = torch.linalg.norm(y, dim=-1)
        torch.testing.assert_close(norms_x, norms_y, atol=1e-5, rtol=1e-5)

    def test_deterministic_seeding(self) -> None:
        r1 = RandomRotation(dim=64, seed=123)
        r2 = RandomRotation(dim=64, seed=123)
        torch.testing.assert_close(r1.matrix, r2.matrix)

    def test_different_seeds_differ(self) -> None:
        r1 = RandomRotation(dim=64, seed=0)
        r2 = RandomRotation(dim=64, seed=1)
        assert not torch.allclose(r1.matrix, r2.matrix)

    def test_inverse_round_trip(self) -> None:
        rot = RandomRotation(dim=128, seed=7)
        g = torch.Generator().manual_seed(0)
        x = torch.randn(16, 128, generator=g)
        y = rot.forward(x)
        x_hat = rot.inverse(y)
        torch.testing.assert_close(x_hat, x, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("d", [32, 64])
    def test_batch_dimensions(self, d: int) -> None:
        rot = RandomRotation(dim=d, seed=0)
        g = torch.Generator().manual_seed(0)
        x = torch.randn(4, 8, d, generator=g)
        y = rot.forward(x)
        assert y.shape == x.shape
        x_hat = rot.inverse(y)
        torch.testing.assert_close(x_hat, x, atol=1e-5, rtol=1e-5)


class TestFunctionalAPI:
    """Tests for the functional wrappers."""

    def test_round_trip(self) -> None:
        g = torch.Generator().manual_seed(0)
        x = torch.randn(10, 64, generator=g)
        y = random_rotate(x, seed=42)
        x_hat = random_rotate_inverse(y, seed=42)
        torch.testing.assert_close(x_hat, x, atol=1e-5, rtol=1e-5)

    def test_wrong_seed_does_not_recover(self) -> None:
        g = torch.Generator().manual_seed(0)
        x = torch.randn(10, 64, generator=g)
        y = random_rotate(x, seed=42)
        x_bad = random_rotate_inverse(y, seed=99)
        assert not torch.allclose(x_bad, x, atol=1e-3)
