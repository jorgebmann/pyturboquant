"""LangChain VectorStore integration for TurboQuant NN search.

Provides TurboQuantVectorStore, a LangChain-compatible vector store backed
by TurboQuantIndex for low-RAM, high-speed RAG pipelines.

Install with: pip install pyturboquant[langchain]
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore
except ImportError as e:
    raise ImportError(
        "LangChain dependencies not found. "
        "Install with: pip install pyturboquant[langchain]"
    ) from e

import torch

from pyturboquant.search.index import TurboQuantIndex


class TurboQuantVectorStore(VectorStore):
    """LangChain VectorStore backed by a TurboQuantIndex.

    Wraps the TurboQuant NN search engine to provide a standard LangChain
    interface for building RAG pipelines with extreme memory efficiency.

    Args:
        embedding: LangChain Embeddings model for text -> vector.
        bits: Quantization bit-width (>= 2).
        metric: Distance metric ("ip" or "l2").
        seed: Deterministic seed for quantization.
        device: Torch device.
    """

    def __init__(
        self,
        embedding: Embeddings,
        *,
        dim: int | None = None,
        bits: int = 4,
        metric: str = "ip",
        seed: int = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        self._embedding = embedding
        self._bits = bits
        self._metric = metric
        self._seed = seed
        self._device = torch.device(device) if isinstance(device, str) else device
        self._dim = dim
        self._index: TurboQuantIndex | None = None
        self._documents: list[Document] = []
        self._ids: list[str] = []

    @property
    def embeddings(self) -> Embeddings:
        """Access the embedding model."""
        return self._embedding

    def _ensure_index(self, dim: int) -> TurboQuantIndex:
        if self._index is None:
            self._dim = dim
            self._index = TurboQuantIndex(
                dim=dim,
                bits=self._bits,
                metric=self._metric,
                seed=self._seed,
                device=self._device,
            )
        return self._index

    def add_texts(
        self,
        texts: list[str] | Iterator[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Embed texts and add them to the index.

        Args:
            texts: Iterable of text strings to add.
            metadatas: Optional metadata dicts for each text.
            ids: Optional IDs for each text.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            List of IDs for the added texts.
        """
        text_list = list(texts)
        if not text_list:
            return []

        vectors = self._embedding.embed_documents(text_list)
        embeddings_tensor = torch.tensor(vectors, dtype=torch.float32, device=self._device)
        idx = self._ensure_index(embeddings_tensor.shape[-1])
        idx.add(embeddings_tensor)

        result_ids = ids or [str(uuid.uuid4()) for _ in text_list]
        if metadatas is None:
            metadatas = [{} for _ in text_list]

        for text, meta, doc_id in zip(text_list, metadatas, result_ids, strict=True):
            self._documents.append(
                Document(page_content=text, metadata=meta)
            )
            self._ids.append(doc_id)

        return result_ids

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        """Search for documents similar to the query.

        Args:
            query: Query text.
            k: Number of results.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            List of Documents ranked by similarity.
        """
        docs_and_scores = self.similarity_search_with_score(query, k=k)
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """Search with similarity scores.

        Args:
            query: Query text.
            k: Number of results.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            List of (Document, score) tuples. For IP metric, higher is better.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        query_vec = self._embedding.embed_query(query)
        query_tensor = torch.tensor(query_vec, dtype=torch.float32, device=self._device)
        return self.similarity_search_by_vector_with_score(query_tensor, k=k)

    def similarity_search_by_vector(
        self, embedding: torch.Tensor | list[float], k: int = 4, **kwargs: Any
    ) -> list[Document] | list[list[Document]]:
        """Search by pre-computed embedding vector.

        Args:
            embedding: Query embedding as tensor or list (``(d,)``, ``(1, d)``, or ``(nq, d)``).
            k: Number of results.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Documents ranked by similarity; nested list when ``nq > 1``.
        """
        results = self.similarity_search_by_vector_with_score(embedding, k=k)
        if not results:
            return []
        if isinstance(results[0], tuple):
            return [doc for doc, _ in results]
        return [[doc for doc, _ in row] for row in results]

    def similarity_search_by_vector_with_score(
        self,
        embedding: torch.Tensor | list[float],
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]] | list[list[tuple[Document, float]]]:
        """Search by vector with scores.

        Args:
            embedding: Query embedding of shape ``(d,)``, ``(1, d)``, or ``(nq, d)``.
            k: Number of results.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            For ``(d,)`` or ``(1, d)``, a flat list of up to ``k`` ``(Document, score)``
            pairs. For ``(nq, d)`` with ``nq > 1``, a list of length ``nq``, each
            entry a list of up to ``k`` pairs for that query row.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        if isinstance(embedding, list):
            embedding = torch.tensor(embedding, dtype=torch.float32, device=self._device)
        embedding = embedding.to(self._device)

        distances, indices = self._index.search(embedding, k=k)
        if indices.dim() == 1:
            row_results: list[tuple[Document, float]] = []
            for rank in range(indices.shape[0]):
                doc_idx = int(indices[rank].item())
                score = float(distances[rank].item())
                if 0 <= doc_idx < len(self._documents):
                    row_results.append((self._documents[doc_idx], score))
            return row_results

        batch_out: list[list[tuple[Document, float]]] = []
        for qi in range(indices.shape[0]):
            row: list[tuple[Document, float]] = []
            for rank in range(indices.shape[1]):
                doc_idx = int(indices[qi, rank].item())
                score = float(distances[qi, rank].item())
                if 0 <= doc_idx < len(self._documents):
                    row.append((self._documents[doc_idx], score))
            batch_out.append(row)
        if indices.shape[0] == 1:
            return batch_out[0]
        return batch_out

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        bits: int = 4,
        metric: str = "ip",
        seed: int = 0,
        device: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> TurboQuantVectorStore:
        """Create a TurboQuantVectorStore from a list of texts.

        Args:
            texts: List of text strings.
            embedding: LangChain Embeddings model.
            metadatas: Optional metadata for each text.
            bits: Quantization bit-width.
            metric: Distance metric.
            seed: Deterministic seed.
            device: Torch device.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Initialized and populated TurboQuantVectorStore.
        """
        store = cls(
            embedding=embedding,
            bits=bits,
            metric=metric,
            seed=seed,
            device=device,
        )
        store.add_texts(texts, metadatas=metadatas)
        return store

    def save(self, path: str | Path) -> None:
        """Save the vector store to disk.

        Args:
            path: Directory path for saving.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            self._index.save(path / "index.pt")
        torch.save(
            {
                "documents": [
                    {"page_content": d.page_content, "metadata": d.metadata}
                    for d in self._documents
                ],
                "ids": self._ids,
                "bits": self._bits,
                "metric": self._metric,
                "seed": self._seed,
                "dim": self._dim,
            },
            path / "metadata.pt",
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        embedding: Embeddings,
        device: str | torch.device = "cpu",
    ) -> TurboQuantVectorStore:
        """Load a saved vector store.

        Args:
            path: Directory path to load from.
            embedding: LangChain Embeddings model.
            device: Torch device.

        Returns:
            Loaded TurboQuantVectorStore.

        Note:
            Metadata is loaded with ``weights_only=False`` because it contains
            nested Python structures (documents, ids). Only load from trusted paths.
        """
        path = Path(path)
        meta = torch.load(path / "metadata.pt", weights_only=False)

        store = cls(
            embedding=embedding,
            dim=meta["dim"],
            bits=meta["bits"],
            metric=meta["metric"],
            seed=meta["seed"],
            device=device,
        )

        index_path = path / "index.pt"
        if index_path.exists():
            store._index = TurboQuantIndex.load(index_path, device=device)

        store._documents = [
            Document(page_content=d["page_content"], metadata=d["metadata"])
            for d in meta["documents"]
        ]
        store._ids = meta["ids"]
        return store
