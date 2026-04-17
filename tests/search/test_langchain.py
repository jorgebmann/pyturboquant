"""Tests for the LangChain VectorStore integration."""

from __future__ import annotations

import tempfile

import pytest
import torch

langchain_available = True
try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
except ImportError:
    langchain_available = False

pytestmark = pytest.mark.skipif(
    not langchain_available, reason="langchain-core not installed"
)


class MockEmbeddings(Embeddings):
    """Deterministic mock embeddings for testing."""

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        result = []
        for _i, text in enumerate(texts):
            g = torch.Generator().manual_seed(hash(text) % (2**31))
            vec = torch.randn(self.dim, generator=g).tolist()
            result.append(vec)
        return result

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class TestTurboQuantVectorStore:
    """Tests for TurboQuantVectorStore."""

    def _make_store(self) -> TurboQuantVectorStore:  # noqa: F821
        from pyturboquant.search.langchain import TurboQuantVectorStore

        emb = MockEmbeddings(dim=32)
        store = TurboQuantVectorStore(embedding=emb, bits=3, metric="ip")
        return store

    def test_add_texts_and_search(self) -> None:
        store = self._make_store()
        texts = [f"Document number {i}" for i in range(20)]
        ids = store.add_texts(texts)
        assert len(ids) == 20

        results = store.similarity_search("Document number 5", k=3)
        assert len(results) == 3
        assert all(isinstance(r, Document) for r in results)

    def test_from_texts(self) -> None:
        from pyturboquant.search.langchain import TurboQuantVectorStore

        emb = MockEmbeddings(dim=32)
        store = TurboQuantVectorStore.from_texts(
            texts=["Hello world", "Foo bar", "Test document"],
            embedding=emb,
            bits=3,
        )
        results = store.similarity_search("Hello", k=2)
        assert len(results) == 2

    def test_similarity_search_with_score(self) -> None:
        store = self._make_store()
        texts = [f"Doc {i}" for i in range(10)]
        store.add_texts(texts)
        results = store.similarity_search_with_score("Doc 0", k=5)
        assert len(results) == 5
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)

        if store._metric == "ip":
            scores = [s for _, s in results]
            assert scores == sorted(scores, reverse=True), "IP scores should be descending"

    def test_similarity_search_by_vector(self) -> None:
        store = self._make_store()
        texts = [f"Entry {i}" for i in range(10)]
        store.add_texts(texts)
        query_vec = store._embedding.embed_query("Entry 3")
        results = store.similarity_search_by_vector(query_vec, k=3)
        assert len(results) == 3

    def test_similarity_search_by_vectors_batch(self) -> None:
        from pyturboquant.search.langchain import TurboQuantVectorStore

        emb = MockEmbeddings(dim=32)
        store = TurboQuantVectorStore(embedding=emb, bits=3, metric="ip")
        store.add_texts([f"Doc {i}" for i in range(8)])
        q1 = torch.tensor(emb.embed_query("Doc 0"), dtype=torch.float32)
        q2 = torch.tensor(emb.embed_query("Doc 7"), dtype=torch.float32)
        batch = torch.stack([q1, q2], dim=0)
        scored = store.similarity_search_by_vectors_with_score(batch, k=3)
        assert isinstance(scored, list)
        assert len(scored) == 2
        assert all(isinstance(row, list) for row in scored)
        assert all(len(row) <= 3 for row in scored)
        assert all(isinstance(doc, Document) for row in scored for doc, _ in row)

        docs = store.similarity_search_by_vectors(batch, k=2)
        assert len(docs) == 2
        assert all(isinstance(row, list) for row in docs)
        assert all(len(row) <= 2 for row in docs)

    def test_similarity_search_by_vector_single_row_batch(self) -> None:
        """A ``(1, d)`` tensor is treated as a single query and returns a flat list."""
        store = self._make_store()
        store.add_texts([f"Entry {i}" for i in range(8)])
        q = torch.tensor(store._embedding.embed_query("Entry 0"), dtype=torch.float32)
        flat = store.similarity_search_by_vector(q, k=3)
        single_row = store.similarity_search_by_vector(q.unsqueeze(0), k=3)
        assert isinstance(flat[0], Document)
        assert isinstance(single_row[0], Document)
        assert [d.page_content for d in flat] == [d.page_content for d in single_row]

    def test_similarity_search_by_vector_rejects_batch(self) -> None:
        """Passing an actual batch to the single-query API is an error."""
        store = self._make_store()
        store.add_texts([f"Entry {i}" for i in range(4)])
        emb = store._embedding
        q1 = torch.tensor(emb.embed_query("Entry 0"), dtype=torch.float32)
        q2 = torch.tensor(emb.embed_query("Entry 1"), dtype=torch.float32)
        batch = torch.stack([q1, q2], dim=0)
        with pytest.raises(ValueError, match="single query"):
            store.similarity_search_by_vector(batch, k=2)
        with pytest.raises(ValueError, match="single query"):
            store.similarity_search_by_vector_with_score(batch, k=2)

    def test_delete_by_ids(self) -> None:
        store = self._make_store()
        ids = store.add_texts(
            ["alpha", "beta", "gamma", "delta"], ids=["a", "b", "c", "d"]
        )
        assert ids == ["a", "b", "c", "d"]

        assert store.delete(["b", "c"]) is True
        results = store.similarity_search("alpha", k=10)
        contents = {d.page_content for d in results}
        assert "beta" not in contents
        assert "gamma" not in contents
        assert "alpha" in contents
        assert "delta" in contents

        assert store.delete(["nonexistent"]) is False

    def test_delete_all(self) -> None:
        store = self._make_store()
        store.add_texts(["x", "y", "z"])
        assert store.delete() is True
        assert store.similarity_search("x", k=3) == []

    def test_get_by_ids(self) -> None:
        store = self._make_store()
        store.add_texts(["p", "q", "r"], ids=["i1", "i2", "i3"])
        docs = store.get_by_ids(["i3", "i1", "missing"])
        assert [d.page_content for d in docs] == ["r", "p"]
        store.delete(["i1"])
        docs_after = store.get_by_ids(["i1", "i2"])
        assert [d.page_content for d in docs_after] == ["q"]

    def test_add_texts_ids_keyword_only(self) -> None:
        """Mirrors the base ``VectorStore.add_texts`` signature: ``ids`` is keyword-only."""
        store = self._make_store()
        with pytest.raises(TypeError):
            store.add_texts(["a", "b"], None, ["i1", "i2"])  # type: ignore[misc]

    def test_add_texts_ids_length_mismatch(self) -> None:
        store = self._make_store()
        with pytest.raises(ValueError, match="ids length"):
            store.add_texts(["a", "b"], ids=["only-one"])

    def test_add_texts_accepts_iterable(self) -> None:
        store = self._make_store()
        ids = store.add_texts(iter([f"gen-{i}" for i in range(3)]))
        assert len(ids) == 3

    def test_init_with_dim_creates_index_eagerly(self) -> None:
        from pyturboquant.search.langchain import TurboQuantVectorStore

        emb = MockEmbeddings(dim=16)
        store = TurboQuantVectorStore(embedding=emb, dim=16, bits=3, metric="ip")
        assert store._index is not None
        assert store._index.ntotal == 0
        store.add_texts(["hello"])
        assert store._index.ntotal == 1

    def test_auto_consolidate_on_search(self) -> None:
        """Streaming many small batches triggers consolidation on search."""
        from pyturboquant.search.langchain import (
            _AUTO_CONSOLIDATE_THRESHOLD as threshold,
        )
        from pyturboquant.search.langchain import TurboQuantVectorStore

        emb = MockEmbeddings(dim=16)
        store = TurboQuantVectorStore(embedding=emb, bits=3, metric="ip")
        for i in range(threshold + 2):
            store.add_texts([f"doc-{i}"])
        assert store._index is not None
        assert store._index.nchunks > threshold
        store.similarity_search("doc-0", k=1)
        assert store._index.nchunks == 1

    def test_save_clears_stale_index_file(self, tmp_path: object) -> None:
        """Saving a store whose index was deleted must not leave a stale index.pt."""
        import pathlib

        from pyturboquant.search.langchain import TurboQuantVectorStore

        emb = MockEmbeddings(dim=16)
        store = TurboQuantVectorStore.from_texts(["a", "b", "c"], emb, bits=3)
        dir_path = pathlib.Path(str(tmp_path))
        store.save(dir_path)
        assert (dir_path / "index.pt").exists()

        store._index = None
        store._documents = []
        store._ids = []
        store._id_to_pos = {}
        store.save(dir_path)
        assert not (dir_path / "index.pt").exists()

    def test_re_export_from_search_package(self) -> None:
        import pyturboquant.search as search_pkg
        from pyturboquant.search.langchain import (
            TurboQuantVectorStore as CanonicalStore,
        )

        assert search_pkg.TurboQuantVectorStore is CanonicalStore
        assert "TurboQuantVectorStore" in search_pkg.__all__

    def test_add_texts_with_metadata(self) -> None:
        store = self._make_store()
        texts = ["Alpha", "Beta", "Gamma"]
        metadatas = [{"source": "a"}, {"source": "b"}, {"source": "c"}]
        store.add_texts(texts, metadatas=metadatas)
        results = store.similarity_search("Alpha", k=1)
        assert results[0].metadata.get("source") in {"a", "b", "c"}

    def test_empty_search_returns_empty(self) -> None:
        store = self._make_store()
        results = store.similarity_search("anything", k=3)
        assert results == []

    def test_add_texts_empty(self) -> None:
        store = self._make_store()
        ids = store.add_texts([])
        assert ids == []

    def test_custom_ids(self) -> None:
        store = self._make_store()
        ids = store.add_texts(["one", "two"], ids=["id-1", "id-2"])
        assert ids == ["id-1", "id-2"]

    def test_save_load_round_trip(self) -> None:
        from pyturboquant.search.langchain import TurboQuantVectorStore

        emb = MockEmbeddings(dim=32)
        store = TurboQuantVectorStore.from_texts(
            texts=["Doc A", "Doc B", "Doc C"],
            embedding=emb,
            bits=3,
        )
        results_before = store.similarity_search("Doc A", k=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            store.save(tmpdir)
            loaded = TurboQuantVectorStore.load(tmpdir, embedding=emb)

        results_after = loaded.similarity_search("Doc A", k=2)
        assert len(results_after) == len(results_before)
        assert results_after[0].page_content == results_before[0].page_content

    def test_embeddings_property(self) -> None:
        store = self._make_store()
        assert isinstance(store.embeddings, MockEmbeddings)
