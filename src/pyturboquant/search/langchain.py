"""LangChain VectorStore integration for TurboQuant NN search.

Provides TurboQuantVectorStore, a LangChain-compatible vector store backed
by TurboQuantIndex for low-RAM, high-speed RAG pipelines.

Install with: pip install pyturboquant[langchain]
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
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

_AUTO_CONSOLIDATE_THRESHOLD = 16
"""When the underlying index has more than this many pending chunks, a
store-level search triggers :meth:`TurboQuantIndex.consolidate` to amortize
Python-loop overhead from streaming ingestion.
"""


class TurboQuantVectorStore(VectorStore):
    """LangChain VectorStore backed by a TurboQuantIndex.

    Wraps the TurboQuant NN search engine to provide a standard LangChain
    interface for building RAG pipelines with extreme memory efficiency.

    Args:
        embedding: LangChain Embeddings model for text -> vector.
        dim: Optional vector dimension. If provided, the underlying index is
            created eagerly so that ``add_texts`` can run without first inferring
            the dimension from an embedding batch.
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
        if dim is not None:
            self._index = TurboQuantIndex(
                dim=dim,
                bits=bits,
                metric=metric,
                seed=seed,
                device=self._device,
            )
        self._documents: list[Document] = []
        self._ids: list[str] = []
        self._id_to_pos: dict[str, int] = {}
        self._tombstones: set[int] = set()

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

    def _maybe_consolidate(self) -> None:
        if (
            self._index is not None
            and self._index.nchunks > _AUTO_CONSOLIDATE_THRESHOLD
        ):
            self._index.consolidate()

    def consolidate(self) -> None:
        """Force consolidation of any pending index chunks.

        Useful after heavy streaming ingestion; normally the store consolidates
        automatically on search when enough chunks have accumulated.
        """
        if self._index is not None:
            self._index.consolidate()

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Embed texts and add them to the index.

        Args:
            texts: Iterable of text strings to add.
            metadatas: Optional metadata dicts for each text.
            ids: Optional IDs for each text (keyword-only, matching the base
                ``VectorStore`` signature).
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            List of IDs for the added texts.
        """
        text_list = list(texts)
        if not text_list:
            return []

        if ids is not None and len(ids) != len(text_list):
            raise ValueError(
                f"ids length ({len(ids)}) does not match texts length ({len(text_list)})"
            )

        vectors = self._embedding.embed_documents(text_list)
        embeddings_tensor = torch.tensor(vectors, dtype=torch.float32, device=self._device)
        idx = self._ensure_index(embeddings_tensor.shape[-1])
        idx.add(embeddings_tensor)

        result_ids = list(ids) if ids is not None else [str(uuid.uuid4()) for _ in text_list]
        if metadatas is None:
            metadatas = [{} for _ in text_list]

        start = len(self._documents)
        for i, (text, meta, doc_id) in enumerate(
            zip(text_list, metadatas, result_ids, strict=True)
        ):
            pos = start + i
            self._documents.append(Document(page_content=text, metadata=meta))
            self._ids.append(doc_id)
            self._id_to_pos[doc_id] = pos

        return result_ids

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """Soft-delete documents by id.

        The underlying :class:`TurboQuantIndex` is append-only, so deleted
        positions are tombstoned and filtered out at search time. ``add_texts``
        followed by ``delete`` does not reclaim memory; call ``consolidate`` if
        you need to normalize chunk layout afterwards.

        Args:
            ids: IDs to remove. If ``None``, all documents are deleted.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            ``True`` if every requested id was present; ``False`` if any id was
            unknown. When ``ids`` is ``None``, always ``True``.
        """
        if ids is None:
            self._tombstones = set(range(len(self._documents)))
            return True
        all_found = True
        for doc_id in ids:
            pos = self._id_to_pos.get(doc_id)
            if pos is None:
                all_found = False
                continue
            self._tombstones.add(pos)
        return all_found

    def get_by_ids(self, ids: Iterable[str]) -> list[Document]:
        """Return documents matching the given ids, preserving order.

        Unknown or tombstoned ids are silently skipped.
        """
        out: list[Document] = []
        for doc_id in ids:
            pos = self._id_to_pos.get(doc_id)
            if pos is None or pos in self._tombstones:
                continue
            out.append(self._documents[pos])
        return out

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
        return self.similarity_search_by_vector_with_score(query_vec, k=k)

    def similarity_search_by_vector(
        self,
        embedding: list[float] | torch.Tensor,
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        """Search by a single pre-computed embedding vector.

        Matches the base ``VectorStore`` contract: input is a single query and
        output is a flat list of Documents. Use :meth:`similarity_search_by_vectors`
        to search a batch of queries.

        Args:
            embedding: Query embedding of shape ``(d,)`` or ``(1, d)``, either as
                a ``list[float]`` or a ``torch.Tensor``.
            k: Number of results.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Documents ranked by similarity.
        """
        results = self.similarity_search_by_vector_with_score(embedding, k=k)
        return [doc for doc, _ in results]

    def similarity_search_by_vector_with_score(
        self,
        embedding: list[float] | torch.Tensor,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Single-query vector search with scores.

        Args:
            embedding: Query embedding of shape ``(d,)`` or ``(1, d)``.
            k: Number of results.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Flat list of up to ``k`` ``(Document, score)`` pairs.

        Raises:
            ValueError: If a 2-D tensor with more than one row is passed; use
                :meth:`similarity_search_by_vectors_with_score` for batch search.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        tensor = self._coerce_embedding(embedding)
        if tensor.dim() > 1:
            if tensor.shape[0] != 1:
                raise ValueError(
                    "similarity_search_by_vector_with_score expects a single query "
                    f"(got shape {tuple(tensor.shape)}); "
                    "use similarity_search_by_vectors_with_score for batch search"
                )
            tensor = tensor.squeeze(0)

        self._maybe_consolidate()
        return self._search_single(tensor, k)

    def similarity_search_by_vectors(
        self,
        embeddings: list[list[float]] | torch.Tensor,
        k: int = 4,
        **kwargs: Any,
    ) -> list[list[Document]]:
        """Batch variant of :meth:`similarity_search_by_vector`.

        Args:
            embeddings: Query embeddings of shape ``(nq, d)``.
            k: Number of results per query.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            List of length ``nq``; each entry a list of up to ``k`` Documents.
        """
        scored = self.similarity_search_by_vectors_with_score(embeddings, k=k)
        return [[doc for doc, _ in row] for row in scored]

    def similarity_search_by_vectors_with_score(
        self,
        embeddings: list[list[float]] | torch.Tensor,
        k: int = 4,
        **kwargs: Any,
    ) -> list[list[tuple[Document, float]]]:
        """Batch variant of :meth:`similarity_search_by_vector_with_score`.

        Args:
            embeddings: Query embeddings of shape ``(nq, d)``. A 1-D tensor is
                treated as a batch of size 1.
            k: Number of results per query.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            List of length ``nq``; each entry a list of up to ``k``
            ``(Document, score)`` pairs.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        tensor = self._coerce_embedding(embeddings)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)

        self._maybe_consolidate()
        return self._search_batch(tensor, k)

    def _coerce_embedding(
        self, embedding: list[float] | list[list[float]] | torch.Tensor
    ) -> torch.Tensor:
        if isinstance(embedding, torch.Tensor):
            tensor = embedding
        else:
            tensor = torch.tensor(embedding, dtype=torch.float32)
        return tensor.to(device=self._device, dtype=torch.float32)

    def _search_single(
        self, query: torch.Tensor, k: int
    ) -> list[tuple[Document, float]]:
        assert self._index is not None
        effective_k = min(k + len(self._tombstones), self._index.ntotal)
        distances, indices = self._index.search(query, k=effective_k)
        out: list[tuple[Document, float]] = []
        for rank in range(indices.shape[0]):
            if len(out) >= k:
                break
            doc_idx = int(indices[rank].item())
            if doc_idx in self._tombstones:
                continue
            if 0 <= doc_idx < len(self._documents):
                out.append((self._documents[doc_idx], float(distances[rank].item())))
        return out

    def _search_batch(
        self, queries: torch.Tensor, k: int
    ) -> list[list[tuple[Document, float]]]:
        assert self._index is not None
        effective_k = min(k + len(self._tombstones), self._index.ntotal)
        distances, indices = self._index.search(queries, k=effective_k)
        out: list[list[tuple[Document, float]]] = []
        for qi in range(indices.shape[0]):
            row: list[tuple[Document, float]] = []
            for rank in range(indices.shape[1]):
                if len(row) >= k:
                    break
                doc_idx = int(indices[qi, rank].item())
                if doc_idx in self._tombstones:
                    continue
                if 0 <= doc_idx < len(self._documents):
                    row.append(
                        (self._documents[doc_idx], float(distances[qi, rank].item()))
                    )
            out.append(row)
        return out

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
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
            ids: Optional IDs for each text (keyword-only).
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
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store

    def save(self, path: str | Path) -> None:
        """Save the vector store to disk.

        Args:
            path: Directory path for saving.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        index_file = path / "index.pt"
        if self._index is not None:
            self._index.save(index_file)
        elif index_file.exists():
            index_file.unlink()
        torch.save(
            {
                "documents": [
                    {"page_content": d.page_content, "metadata": d.metadata}
                    for d in self._documents
                ],
                "ids": self._ids,
                "tombstones": sorted(self._tombstones),
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
            dim=None,
            bits=meta["bits"],
            metric=meta["metric"],
            seed=meta["seed"],
            device=device,
        )
        store._dim = meta["dim"]

        index_path = path / "index.pt"
        if index_path.exists():
            store._index = TurboQuantIndex.load(index_path, device=device)

        store._documents = [
            Document(page_content=d["page_content"], metadata=d["metadata"])
            for d in meta["documents"]
        ]
        store._ids = list(meta["ids"])
        store._id_to_pos = {doc_id: i for i, doc_id in enumerate(store._ids)}
        store._tombstones = set(meta.get("tombstones", []))
        return store
