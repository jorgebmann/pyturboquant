"""Support utilities for benchmarks/bench_beir.py.

Kept in a separate module so the main script reads top-down without mixing
IO plumbing (BEIR download, sentence-transformers encoding, cache) with the
actual benchmarking flow.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Iterable


BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


@dataclass(frozen=True)
class DatasetSpec:
    """A BEIR dataset entry with the split that has qrels."""

    name: str
    split: str
    display_name: str


DATASETS: dict[str, DatasetSpec] = {
    "scifact": DatasetSpec("scifact", "test", "SciFact"),
    "nfcorpus": DatasetSpec("nfcorpus", "test", "NFCorpus"),
    "fiqa": DatasetSpec("fiqa", "test", "FiQA-2018"),
    "scidocs": DatasetSpec("scidocs", "test", "SciDocs"),
    "arguana": DatasetSpec("arguana", "test", "ArguAna"),
    "msmarco": DatasetSpec("msmarco", "dev", "MSMARCO"),
}

SMALL_SUITE: tuple[str, ...] = ("scifact", "nfcorpus", "fiqa", "scidocs", "arguana")


@dataclass(frozen=True)
class ModelSpec:
    """Embedding model metadata."""

    alias: str
    hf_id: str
    dim: int
    display_name: str


MODELS: dict[str, ModelSpec] = {
    "bge-base": ModelSpec(
        "bge-base", "BAAI/bge-base-en-v1.5", 768, "BGE-base-en-v1.5"
    ),
    "minilm": ModelSpec(
        "minilm", "sentence-transformers/all-MiniLM-L6-v2", 384, "MiniLM-L6-v2"
    ),
}


def parse_dataset_list(raw: str) -> list[str]:
    """Resolve a comma-separated CLI value into dataset keys.

    Supports the special tokens ``small`` (the small suite) and ``all``
    (small suite + msmarco).
    """
    items = [s.strip().lower() for s in raw.split(",") if s.strip()]
    out: list[str] = []
    for item in items:
        if item == "small":
            out.extend(SMALL_SUITE)
        elif item == "all":
            out.extend(SMALL_SUITE)
            out.append("msmarco")
        elif item in DATASETS:
            out.append(item)
        else:
            raise ValueError(
                f"Unknown dataset {item!r}. "
                f"Valid: {sorted(DATASETS)} or 'small'/'all'."
            )
    seen: set[str] = set()
    deduped: list[str] = []
    for d in out:
        if d not in seen:
            seen.add(d)
            deduped.append(d)
    return deduped


def download_and_load(
    dataset_key: str, cache_dir: Path
) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]]]:
    """Download a BEIR dataset (if needed) and return ``(corpus, queries, qrels)``.

    ``corpus`` maps doc_id -> {"title", "text"}. ``queries`` maps qid -> text.
    ``qrels`` maps qid -> {doc_id -> relevance_int}.
    """
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    spec = DATASETS[dataset_key]
    datasets_root = cache_dir / "datasets"
    datasets_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = datasets_root / spec.name
    if not dataset_dir.exists():
        url = BEIR_URL.format(name=spec.name)
        util.download_and_unzip(url, str(datasets_root))

    corpus, queries, qrels = GenericDataLoader(data_folder=str(dataset_dir)).load(
        split=spec.split
    )
    return corpus, queries, qrels


def corpus_to_texts(
    corpus: dict[str, dict[str, str]],
    max_size: int | None = None,
) -> tuple[list[str], list[str]]:
    """Flatten a BEIR corpus dict to parallel ``(doc_ids, texts)`` lists.

    Concatenates ``title`` and ``text`` with ``". "`` (the BEIR convention).
    Iteration order follows ``corpus``'s insertion order (Python dicts preserve
    order; BEIR's loader yields a stable order).
    """
    doc_ids: list[str] = []
    texts: list[str] = []
    for did, entry in corpus.items():
        title = entry.get("title", "") or ""
        body = entry.get("text", "") or ""
        joined = f"{title}. {body}".strip(". ").strip()
        doc_ids.append(did)
        texts.append(joined)
        if max_size is not None and len(doc_ids) >= max_size:
            break
    return doc_ids, texts


def queries_to_texts(queries: dict[str, str]) -> tuple[list[str], list[str]]:
    """Flatten a BEIR queries dict to parallel ``(query_ids, texts)`` lists."""
    qids = list(queries.keys())
    texts = [queries[q] for q in qids]
    return qids, texts


def _cache_paths(
    cache_dir: Path, model_alias: str, dataset_key: str, kind: str
) -> tuple[Path, Path]:
    """Resolve ``(emb_pt_path, ids_json_path)`` for a given cache slot."""
    sub = cache_dir / "embeddings" / model_alias / dataset_key
    sub.mkdir(parents=True, exist_ok=True)
    return sub / f"{kind}.pt", sub / f"{kind}_ids.json"


def _load_cache(
    emb_path: Path, ids_path: Path, expected_ids: list[str]
) -> torch.Tensor | None:
    """Return cached embeddings if both files exist and ids match; else None."""
    if not emb_path.exists() or not ids_path.exists():
        return None
    with ids_path.open() as f:
        cached_ids: list[str] = json.load(f)
    if cached_ids != expected_ids:
        return None
    state = torch.load(emb_path, weights_only=True, map_location="cpu")
    emb = state["emb"] if isinstance(state, dict) else state
    if not isinstance(emb, torch.Tensor):
        return None
    return emb


def _save_cache(
    emb: torch.Tensor, ids: list[str], emb_path: Path, ids_path: Path
) -> None:
    torch.save({"emb": emb.cpu()}, emb_path)
    with ids_path.open("w") as f:
        json.dump(ids, f)


def encode_with_cache(
    *,
    model_alias: str,
    dataset_key: str,
    kind: str,
    texts: list[str],
    ids: list[str],
    cache_dir: Path,
    device: str | torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Encode ``texts`` with the given model, caching the result on disk.

    Cache hit on identical ``ids`` list avoids re-encoding when sweeping bits
    or re-running the benchmark. Returns an fp32 ``(n, d)`` tensor on CPU,
    L2-normalized so inner product equals cosine similarity.
    """
    emb_path, ids_path = _cache_paths(cache_dir, model_alias, dataset_key, kind)
    cached = _load_cache(emb_path, ids_path, ids)
    if cached is not None:
        return cached

    from sentence_transformers import SentenceTransformer

    model_spec = MODELS[model_alias]
    model = SentenceTransformer(model_spec.hf_id, device=str(device))
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    if emb.dtype != torch.float32:
        emb = emb.to(torch.float32)
    emb = emb.cpu().contiguous()
    _save_cache(emb, ids, emb_path, ids_path)
    return emb


def fp32_search(
    corpus_emb: torch.Tensor,
    queries_emb: torch.Tensor,
    k: int,
    chunk_size: int = 65_536,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact top-k by inner product, chunked over the corpus axis.

    Computes ``queries_emb @ corpus_emb.T`` in ``chunk_size``-row windows so
    peak transient memory is bounded by ``chunk_size * n_queries * 4`` bytes.
    Mirrors the memory contract of ``TurboQuantIndex._compute_scores_batch``.
    """
    dev = torch.device(device) if isinstance(device, str) else device
    q = queries_emb.to(dev)
    n_corpus = corpus_emb.shape[0]
    nq = q.shape[0]
    k = min(k, n_corpus)

    best_scores = torch.full((nq, k), float("-inf"), device=dev)
    best_indices = torch.full((nq, k), -1, dtype=torch.long, device=dev)

    for start in range(0, n_corpus, chunk_size):
        end = min(start + chunk_size, n_corpus)
        c = corpus_emb[start:end].to(dev)
        scores = q @ c.T
        combined_scores = torch.cat([best_scores, scores], dim=1)
        local_idx = torch.arange(start, end, device=dev).unsqueeze(0).expand(nq, -1)
        combined_idx = torch.cat([best_indices, local_idx], dim=1)
        top_vals, top_cols = torch.topk(combined_scores, k, dim=1, largest=True)
        best_scores = top_vals
        best_indices = combined_idx.gather(1, top_cols)

    return best_scores.cpu(), best_indices.cpu()


def to_trec_results(
    scores: torch.Tensor,
    indices: torch.Tensor,
    doc_ids: list[str],
    query_ids: list[str],
) -> dict[str, dict[str, float]]:
    """Convert ``(scores, indices)`` top-k tensors to BEIR's TREC-style dict.

    Output: ``{qid: {doc_id: float_score, ...}}``. Invalid indices (``-1``,
    which can only appear if ``k`` exceeds the corpus size) are skipped.
    """
    results: dict[str, dict[str, float]] = {}
    scores_list = scores.tolist()
    indices_list = indices.tolist()
    for qi, qid in enumerate(query_ids):
        row: dict[str, float] = {}
        for s, di in zip(scores_list[qi], indices_list[qi], strict=True):
            if di < 0:
                continue
            row[doc_ids[di]] = float(s)
        results[qid] = row
    return results


def self_recall_at_k(
    pred_indices: torch.Tensor, baseline_indices: torch.Tensor, k: int
) -> float:
    """Mean fraction of baseline top-k neighbors that appear in pred top-k.

    Both tensors are ``(nq, K)`` with ``K >= k``. Returns a float in [0, 1].
    """
    if pred_indices.numel() == 0:
        return 0.0
    pred = pred_indices[:, :k].tolist()
    base = baseline_indices[:, :k].tolist()
    nq = len(pred)
    total = 0.0
    for p, b in zip(pred, base, strict=True):
        total += len(set(p) & set(b))
    return total / (nq * k)


def format_number(x: float | int | None, spec: str) -> str:
    """Format a metric cell, rendering ``None`` as ``"-"``."""
    if x is None:
        return "-"
    return format(x, spec)


def ensure_writable(path: Path) -> None:
    """Create parent directories for ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)


def pick_device(requested: str) -> str:
    """Resolve ``auto`` / explicit device strings to a concrete torch device."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def env_hf_offline_hint() -> str:
    """Short diagnostic about Hugging Face offline mode for the header banner."""
    flags = [k for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE") if os.getenv(k) == "1"]
    return f" (offline: {','.join(flags)})" if flags else ""


class Timer:
    """Context manager returning elapsed wall-clock milliseconds."""

    def __init__(self) -> None:
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> Timer:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0


def iter_rel_counts(qrels: dict[str, dict[str, int]]) -> Iterable[int]:
    """Yield per-query counts of relevant documents (diagnostic only)."""
    for qid, docs in qrels.items():  # noqa: B007
        yield sum(1 for rel in docs.values() if rel > 0)
