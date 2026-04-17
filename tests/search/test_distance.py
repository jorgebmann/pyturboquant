"""Tests for asymmetric distance helpers."""

from __future__ import annotations

import torch

from pyturboquant.core.prod_quantizer import InnerProductQuantizer
from pyturboquant.search.distance import asymmetric_inner_product, asymmetric_l2


def test_asymmetric_l2_matches_ip_identity() -> None:
    """L2 score should equal ||x||^2 - 2 * ip_est + ||q||^2 (same as TurboQuantIndex)."""
    d, bits, seed = 32, 3, 0
    device = torch.device("cpu")
    g = torch.Generator().manual_seed(1)
    db = torch.randn(12, d, generator=g)
    q = torch.randn(d, generator=g)

    qtz = InnerProductQuantizer(dim=d, bits=bits, seed=seed, device=device)
    qt = qtz.quantize(db)
    qt_list = [qt]
    norms_sq = (db * db).sum(dim=-1)

    ip = asymmetric_inner_product(q, qt_list, qtz.mse_quantizer, qtz.qjl_transform)
    l2 = asymmetric_l2(q, qt_list, qtz.mse_quantizer, qtz.qjl_transform, norms_sq)
    q_norm_sq = (q * q).sum()
    expected = norms_sq - 2.0 * ip + q_norm_sq
    torch.testing.assert_close(l2, expected)
