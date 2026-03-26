"""TurboQuantIndex -- FAISS-inspired nearest neighbor search engine.

Quantizes database vectors using InnerProductQuantizer and performs
asymmetric search against full-precision queries.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from pyturboquant.core.prod_quantizer import InnerProductQuantizer
from pyturboquant.core.types import QuantizedIP


class TurboQuantIndex:
    """GPU-accelerated approximate nearest neighbor index using TurboQuant.

    Quantizes database vectors with InnerProductQuantizer (Algorithm 2)
    and searches via asymmetric inner product / L2 computation.

    Args:
        dim: Vector dimension.
        bits: Total bit budget per coordinate (>= 2).
        metric: Distance metric, one of "ip" (inner product) or "l2".
        seed: Deterministic seed for quantization.
        device: Torch device.
    """

    def __init__(
        self,
        dim: int,
        bits: int = 4,
        metric: str = "ip",
        seed: int = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        if metric not in ("ip", "l2"):
            raise ValueError(f"metric must be 'ip' or 'l2', got {metric!r}")
        self.dim = dim
        self.bits = bits
        self.metric = metric
        self.seed = seed
        self.device = torch.device(device) if isinstance(device, str) else device
        self._quantizer = InnerProductQuantizer(
            dim=dim, bits=bits, seed=seed, device=self.device
        )
        self._qt_data: list[QuantizedIP] = []
        self._norms_sq: list[torch.Tensor] = []
        self._ntotal = 0
        self._last_add_time_ms: float = 0.0

    @property
    def ntotal(self) -> int:
        """Total number of indexed vectors."""
        return self._ntotal

    @property
    def last_add_time_ms(self) -> float:
        """Wall-clock time of the last add() call in milliseconds."""
        return self._last_add_time_ms

    @property
    def memory_usage_mb(self) -> float:
        """Approximate memory usage of the quantized index in MB."""
        total_bytes = 0
        for qt in self._qt_data:
            total_bytes += qt.mse_data.packed_indices.numel()
            total_bytes += qt.mse_data.norms.numel() * 4
            total_bytes += qt.qjl_bits.numel()
            total_bytes += qt.residual_norms.numel() * 4
        return total_bytes / (1024 * 1024)

    def add(self, vectors: torch.Tensor) -> None:
        """Add vectors to the index.

        Args:
            vectors: Tensor of shape (n, d) to index.
        """
        if vectors.shape[-1] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {vectors.shape[-1]}")

        t0 = time.perf_counter()
        vectors = vectors.to(self.device)
        qt = self._quantizer.quantize(vectors)
        self._qt_data.append(qt)
        norms_sq = (vectors * vectors).sum(dim=-1)
        self._norms_sq.append(norms_sq)
        self._ntotal += vectors.shape[0]
        self._last_add_time_ms = (time.perf_counter() - t0) * 1000.0

    def search(
        self, queries: torch.Tensor, k: int = 10
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Search the index for nearest neighbors.

        Args:
            queries: Query vectors of shape (nq, d) or (d,).
            k: Number of neighbors to return.

        Returns:
            Tuple of (distances, indices) each of shape (nq, k).
            For metric="ip", distances are inner products (higher is better).
            For metric="l2", distances are squared L2 (lower is better).
        """
        if self._ntotal == 0:
            raise RuntimeError("Index is empty. Call add() first.")

        single = queries.dim() == 1
        if single:
            queries = queries.unsqueeze(0)
        queries = queries.to(self.device)

        k = min(k, self._ntotal)
        all_distances = []
        all_indices = []

        for qi in range(queries.shape[0]):
            q = queries[qi]
            scores = self._compute_scores(q)
            if self.metric == "ip":
                topk_vals, topk_idx = torch.topk(scores, k, largest=True)
            else:
                topk_vals, topk_idx = torch.topk(scores, k, largest=False)
            all_distances.append(topk_vals)
            all_indices.append(topk_idx)

        distances = torch.stack(all_distances)
        indices = torch.stack(all_indices)

        if single:
            distances = distances.squeeze(0)
            indices = indices.squeeze(0)

        return distances, indices

    def _compute_scores(self, query: torch.Tensor) -> torch.Tensor:
        """Compute scores for a single query against all database vectors."""
        all_scores = []
        for i, qt in enumerate(self._qt_data):
            x_hat = self._quantizer.dequantize(qt)
            flat_x_hat = x_hat.reshape(-1, self.dim)

            mse_ip = (flat_x_hat * query.unsqueeze(0)).sum(dim=-1)

            qjl_ip = self._quantizer.qjl_transform.estimate_inner_product_batch(
                qt.qjl_bits.reshape(-1, qt.qjl_bits.shape[-1]),
                query,
                qt.residual_norms.reshape(-1),
            )

            if self.metric == "ip":
                all_scores.append(mse_ip + qjl_ip)
            else:
                q_norm_sq = (query * query).sum()
                ip = mse_ip + qjl_ip
                l2 = self._norms_sq[i] - 2.0 * ip + q_norm_sq
                all_scores.append(l2)

        return torch.cat(all_scores)

    def save(self, path: str | Path) -> None:
        """Save index state to disk.

        Args:
            path: File path for the saved index.
        """
        path = Path(path)
        state = {
            "dim": self.dim,
            "bits": self.bits,
            "metric": self.metric,
            "seed": self.seed,
            "ntotal": self._ntotal,
            "qt_data": [
                {
                    "mse_packed": qt.mse_data.packed_indices.cpu(),
                    "mse_norms": qt.mse_data.norms.cpu(),
                    "mse_dim": qt.mse_data.dim,
                    "mse_bits": qt.mse_data.bits,
                    "mse_seed": qt.mse_data.seed,
                    "qjl_bits": qt.qjl_bits.cpu(),
                    "residual_norms": qt.residual_norms.cpu(),
                }
                for qt in self._qt_data
            ],
            "norms_sq": [ns.cpu() for ns in self._norms_sq],
        }
        torch.save(state, path)

    @classmethod
    def load(
        cls, path: str | Path, device: str | torch.device = "cpu"
    ) -> TurboQuantIndex:
        """Load a saved index.

        Args:
            path: File path to load from.
            device: Device to load tensors to.

        Returns:
            Loaded TurboQuantIndex.
        """
        from pyturboquant.core.types import QuantizedMSE

        state = torch.load(Path(path), weights_only=True, map_location=device)
        dev = torch.device(device) if isinstance(device, str) else device

        index = cls(
            dim=state["dim"],
            bits=state["bits"],
            metric=state["metric"],
            seed=state["seed"],
            device=dev,
        )

        for qt_dict, ns in zip(state["qt_data"], state["norms_sq"], strict=True):
            mse_data = QuantizedMSE(
                packed_indices=qt_dict["mse_packed"].to(dev),
                norms=qt_dict["mse_norms"].to(dev),
                dim=qt_dict["mse_dim"],
                bits=qt_dict["mse_bits"],
                seed=qt_dict["mse_seed"],
                device=dev,
            )
            qt = QuantizedIP(
                mse_data=mse_data,
                qjl_bits=qt_dict["qjl_bits"].to(dev),
                residual_norms=qt_dict["residual_norms"].to(dev),
                dim=state["dim"],
                bits=state["bits"],
                seed=state["seed"],
                device=dev,
            )
            index._qt_data.append(qt)
            index._norms_sq.append(ns.to(dev))

        index._ntotal = state["ntotal"]
        return index
