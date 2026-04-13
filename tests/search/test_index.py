"""Tests for TurboQuantIndex."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from pyturboquant.search.index import TurboQuantIndex


class TestTurboQuantIndex:
    """Tests for the NN search index."""

    def _make_data(
        self, n: int = 200, d: int = 64, seed: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator().manual_seed(seed)
        db = torch.randn(n, d, generator=g)
        queries = torch.randn(5, d, generator=g)
        return db, queries

    def test_add_and_ntotal(self) -> None:
        idx = TurboQuantIndex(dim=64, bits=3)
        db, _ = self._make_data()
        idx.add(db)
        assert idx.ntotal == 200

    def test_add_incremental(self) -> None:
        idx = TurboQuantIndex(dim=64, bits=3)
        db, _ = self._make_data(n=100)
        idx.add(db[:50])
        idx.add(db[50:])
        assert idx.ntotal == 100

    def test_consolidate_merges_chunks(self) -> None:
        idx = TurboQuantIndex(dim=64, bits=3, metric="ip", seed=0)
        db, queries = self._make_data(n=30)
        idx.add(db[:10])
        idx.add(db[10:20])
        idx.add(db[20:])
        assert len(idx._qt_data) == 3
        idx.consolidate()
        assert len(idx._qt_data) == 1
        assert idx.ntotal == 30
        d1, i1 = idx.search(queries, k=5)
        idx2 = TurboQuantIndex(dim=64, bits=3, metric="ip", seed=0)
        idx2.add(db)
        d2, i2 = idx2.search(queries, k=5)
        torch.testing.assert_close(i1, i2)
        torch.testing.assert_close(d1, d2, atol=1e-4, rtol=1e-4)

    def test_search_shape(self) -> None:
        idx = TurboQuantIndex(dim=64, bits=3, metric="ip")
        db, queries = self._make_data()
        idx.add(db)
        dists, ids = idx.search(queries, k=10)
        assert dists.shape == (5, 10)
        assert ids.shape == (5, 10)

    def test_search_single_query(self) -> None:
        idx = TurboQuantIndex(dim=64, bits=3, metric="ip")
        db, queries = self._make_data()
        idx.add(db)
        dists, ids = idx.search(queries[0], k=5)
        assert dists.shape == (5,)
        assert ids.shape == (5,)

    def test_search_k_larger_than_db(self) -> None:
        idx = TurboQuantIndex(dim=64, bits=3)
        db, queries = self._make_data(n=5)
        idx.add(db)
        dists, _ids = idx.search(queries[0], k=100)
        assert dists.shape == (5,)

    def test_recall_at_1_ip(self) -> None:
        """Recall@1 should be non-trivial for IP metric."""
        d = 64
        n = 200
        g = torch.Generator().manual_seed(42)
        db = torch.randn(n, d, generator=g)
        queries = torch.randn(20, d, generator=g)

        true_ip = queries @ db.T
        true_top1 = true_ip.argmax(dim=-1)

        idx = TurboQuantIndex(dim=d, bits=4, metric="ip", seed=0)
        idx.add(db)
        _, pred_ids = idx.search(queries, k=1)
        pred_top1 = pred_ids.squeeze(-1)

        recall = (pred_top1 == true_top1).float().mean().item()
        assert recall > 0.3, f"Recall@1 = {recall:.2f}, expected > 0.3"

    def test_recall_at_10_ip(self) -> None:
        """Recall@10 should be decent."""
        d = 64
        n = 200
        g = torch.Generator().manual_seed(42)
        db = torch.randn(n, d, generator=g)
        queries = torch.randn(20, d, generator=g)

        true_ip = queries @ db.T
        true_top10 = torch.topk(true_ip, 10, dim=-1).indices

        idx = TurboQuantIndex(dim=d, bits=4, metric="ip", seed=0)
        idx.add(db)
        _, pred_ids = idx.search(queries, k=10)

        hits = 0
        total = 0
        for i in range(queries.shape[0]):
            true_set = set(true_top10[i].tolist())
            pred_set = set(pred_ids[i].tolist())
            hits += len(true_set & pred_set)
            total += 10
        recall = hits / total
        assert recall > 0.2, f"Recall@10 = {recall:.2f}, expected > 0.2"

    def test_save_load_round_trip(self) -> None:
        d = 64
        idx = TurboQuantIndex(dim=d, bits=3, metric="ip", seed=0)
        db, queries = self._make_data(n=50, d=d)
        idx.add(db)
        dists1, ids1 = idx.search(queries, k=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "index.pt"
            idx.save(path)
            loaded = TurboQuantIndex.load(path)

        assert loaded.ntotal == idx.ntotal
        assert loaded.dim == idx.dim
        assert loaded.bits == idx.bits
        dists2, ids2 = loaded.search(queries, k=5)
        torch.testing.assert_close(ids1, ids2)
        torch.testing.assert_close(dists1, dists2, atol=1e-4, rtol=1e-4)

    def test_l2_metric(self) -> None:
        idx = TurboQuantIndex(dim=64, bits=3, metric="l2")
        db, queries = self._make_data()
        idx.add(db)
        dists, _ids = idx.search(queries, k=5)
        assert dists.shape == (5, 5)
        # L2 distances should be non-negative (approximately)
        assert (dists > -1.0).all()

    def test_invalid_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="metric"):
            TurboQuantIndex(dim=64, metric="cosine")

    def test_wrong_dim_raises(self) -> None:
        idx = TurboQuantIndex(dim=64, bits=3)
        with pytest.raises(ValueError, match="dim"):
            idx.add(torch.randn(10, 32))

    def test_empty_search_raises(self) -> None:
        idx = TurboQuantIndex(dim=64, bits=3)
        with pytest.raises(RuntimeError, match="empty"):
            idx.search(torch.randn(64), k=5)

    def test_memory_usage(self) -> None:
        idx = TurboQuantIndex(dim=64, bits=3)
        db, _ = self._make_data()
        idx.add(db)
        assert idx.memory_usage_mb > 0

    def test_last_add_time(self) -> None:
        idx = TurboQuantIndex(dim=64, bits=3)
        db, _ = self._make_data()
        idx.add(db)
        assert idx.last_add_time_ms > 0
