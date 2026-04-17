"""TurboQuantIndex -- FAISS-inspired nearest neighbor search engine.

Quantizes database vectors using InnerProductQuantizer and performs
asymmetric search against full-precision queries.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from pyturboquant.core.prod_quantizer import InnerProductQuantizer
from pyturboquant.core.types import QuantizedIP, QuantizedMSE


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
    def nchunks(self) -> int:
        """Number of internal chunks produced by repeated ``add`` calls.

        Useful for deciding when to call :meth:`consolidate`.
        """
        return len(self._qt_data)

    @property
    def last_add_time_ms(self) -> float:
        """Wall-clock time of the last add() call in milliseconds."""
        return self._last_add_time_ms

    @property
    def memory_usage_mb(self) -> float:
        """Approximate memory usage of the quantized index in MB."""
        total_bytes = 0
        for qt in self._qt_data:
            for t in (
                qt.mse_data.packed_indices,
                qt.mse_data.norms,
                qt.qjl_bits,
                qt.residual_norms,
            ):
                total_bytes += t.numel() * t.element_size()
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

    def consolidate(self) -> None:
        """Merge all index chunks from repeated ``add`` into a single chunk.

        Reduces Python-loop overhead during ``search`` when many small batches
        were added. No-op if there is at most one chunk.
        """
        if len(self._qt_data) <= 1:
            return

        first = self._qt_data[0]
        mse_packed = torch.cat([q.mse_data.packed_indices for q in self._qt_data], dim=0)
        mse_norms = torch.cat(
            [q.mse_data.norms.reshape(-1, 1) for q in self._qt_data], dim=0
        )
        qjl_bits = torch.cat(
            [q.qjl_bits.reshape(q.qjl_bits.shape[0], -1) for q in self._qt_data], dim=0
        )
        residual_norms = torch.cat(
            [q.residual_norms.reshape(-1) for q in self._qt_data], dim=0
        )

        mse_merged = QuantizedMSE(
            packed_indices=mse_packed,
            norms=mse_norms,
            dim=first.mse_data.dim,
            bits=first.mse_data.bits,
            seed=first.mse_data.seed,
            device=self.device,
        )
        merged = QuantizedIP(
            mse_data=mse_merged,
            qjl_bits=qjl_bits,
            residual_norms=residual_norms,
            dim=first.dim,
            bits=first.bits,
            seed=first.seed,
            device=self.device,
        )
        self._qt_data = [merged]
        self._norms_sq = [torch.cat(self._norms_sq, dim=0)]

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
        scores = self._compute_scores_batch(queries)

        if self.metric == "ip":
            topk_vals, topk_idx = torch.topk(scores, k, dim=-1, largest=True)
        else:
            topk_vals, topk_idx = torch.topk(scores, k, dim=-1, largest=False)

        if single:
            topk_vals = topk_vals.squeeze(0)
            topk_idx = topk_idx.squeeze(0)

        return topk_vals, topk_idx

    def _compute_scores_batch(self, queries: torch.Tensor) -> torch.Tensor:
        """Scores for each query vs all DB vectors. Shape (nq, n_total)."""
        nq = queries.shape[0]
        q_norm_sq = (queries * queries).sum(dim=-1, keepdim=True)  # (nq, 1)
        cols: list[torch.Tensor] = []
        for i, qt in enumerate(self._qt_data):
            x_hat = self._quantizer.dequantize(qt)
            flat_x_hat = x_hat.reshape(-1, self.dim)

            mse_ip = flat_x_hat @ queries.T  # (chunk, nq)
            qjl_ip = self._quantizer.qjl_transform.estimate_inner_product_batch_queries(
                qt.qjl_bits.reshape(-1, qt.qjl_bits.shape[-1]),
                queries,
                qt.residual_norms.reshape(-1),
            )
            chunk_ip = mse_ip.T + qjl_ip  # (nq, chunk)
            if self.metric == "ip":
                cols.append(chunk_ip)
            else:
                ns = self._norms_sq[i].unsqueeze(0)
                cols.append(ns - 2.0 * chunk_ip + q_norm_sq)
        return torch.cat(cols, dim=-1)

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
