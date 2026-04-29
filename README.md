# pyturboquant

A GPU-accelerated Python implementation of Google's **TurboQuant: Online Vector Quantization with Near-Optimal Distortion Rate** ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874), [Google Research blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)).

pyturboquant provides data-oblivious vector quantization that achieves near-Shannon-optimal distortion with **zero indexing time** -- no codebook training, no k-means, no data passes. Vectors are quantized independently using random rotations and precomputed Lloyd-Max codebooks, making it ideal for online settings and massive-scale nearest neighbor search.

## Context

TurboQuant was introduced by Google Research in 2025 and has since been widely adopted by the community to compress the Key-Value (KV) cache of long-context models at inference time -- with Google's **Gemma 4** (released April 2026) a particularly popular target through runtimes like MLX, [llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/20969), and [inferrs](https://github.com/ericcurtin/inferrs). TurboQuant is **not baked into Gemma 4's pretrained weights**; it is applied online during the forward pass as a post-training optimization.

pyturboquant brings the same algorithm to the **retrieval side** of the modern LLM stack: compressing sentence-embedding vectors for memory-efficient RAG pipelines and high-throughput approximate nearest neighbor search. The hot path (KV cache during generation) and the cold path (embedding stores for retrieval) share the same core primitive -- a data-oblivious online quantizer -- and this library focuses on making that primitive usable for the latter.

## Use Case: On-Premise RAG

Run an open-source embedding model (BGE, Gemma embeddings, `nomic-embed`, etc.) locally and store the resulting vectors with pyturboquant -- no data leaves your machine or VPC. At 4 bits per coordinate with the inner-product quantizer, a 10 M chunk BGE-base corpus fits in **~4 GB of RAM instead of ~31 GB**, small enough to colocate with the embedding model on a single workstation GPU or an air-gapped enterprise server. There is no codebook training and no periodic reindexing as the corpus grows.

Per-vector storage cost (bytes) for the inner-product quantizer at `bits=b` is `d*b/8 + 8`, versus `4*d` for fp32 and `2*d` for fp16:

| Embedding model | d | fp32 | fp16 | **b=4** | **b=3** | **b=2** |
|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 384 | 1 536 | 768 | **200** | **152** | **104** |
| BGE-base / Gemma embeddings | 768 | 3 072 | 1 536 | **392** | **296** | **200** |
| BGE-large | 1 024 | 4 096 | 2 048 | **520** | **392** | **264** |
| OpenAI text-embedding-3-large | 3 072 | 12 288 | 6 144 | **1 544** | **1 160** | **776** |

Scope notes:

- pyturboquant compresses the **embedding vector store**, not the embedding model itself. VRAM needed to run the embedding model is unchanged; use model-level quantization (AWQ, bitsandbytes, GGUF) for that.
- **Storage and search both fit the same budget.** Peak fp32 memory during `search()` is bounded by `search_batch_size * dim * 4` bytes (default window: 65,536 vectors) -- not by the total number of indexed vectors -- so the ~4 GB storage number for a 10 M chunk BGE-base corpus is a realistic working-set number for query time, not just at-rest storage. Tune `search_batch_size` lower for tight memory budgets or higher for slightly better throughput; `benchmarks/bench_search_memory.py` reproduces the numbers.
- Search compute is currently O(n) per query (dense asymmetric matmul). Sub-linear search via IVF partitioning is on the roadmap (v0.5.0).
- For small corpora (< 100 k chunks) the document payload typically dominates; pyturboquant's savings become material at 1 M chunks and transformative at 10 M+.

## Features

- **MSE-Optimal Quantizer** (Algorithm 1) -- per-coordinate scalar quantization after random rotation, approaching the Shannon lower bound at 1-8 bits
- **Inner Product Quantizer** (Algorithm 2) -- two-stage MSE + 1-bit QJL residual for unbiased inner product estimation
- **Zero-indexing-time ANN search** -- `TurboQuantIndex` with FAISS-like `.add()` / `.search()` API, asymmetric distance computation, save/load persistence; no training step, no data passes
- **Truly online** -- stream new documents in at any time without rebuilding or reindexing
- **Bounded search-time memory** -- configurable `search_batch_size` caps peak fp32 reconstruction at `search_batch_size * dim * 4` bytes regardless of how many vectors are indexed; a 10 M vector index does not require tens of GB of scratch RAM during queries
- **LangChain VectorStore** -- drop-in `TurboQuantVectorStore` for low-RAM RAG pipelines
- **Pure PyTorch** -- no custom C++ extensions; runs on CPU and CUDA out of the box
- **Deterministic** -- seed-based rotation matrices and QJL projections for full reproducibility

### How it compares to FAISS, ScaNN, and HNSW

| | Training required | Online ingestion | Distortion guarantee |
|---|---|---|---|
| FAISS IVF/PQ | yes (k-means on a sample) | rebuild on drift | empirical |
| ScaNN | yes (learned quantizer) | rebuild on drift | empirical |
| HNSW | no | yes (graph mutation) | none |
| **pyturboquant** | **no** | **yes (independent per vector)** | **within ~2.7x of Shannon bound** |

## Installation

```bash
# Core library (torch only)
pip install pyturboquant

# With LangChain RAG support
pip install pyturboquant[langchain]

# Development (tests, linting, codebook generation)
pip install pyturboquant[dev]

# Everything
pip install pyturboquant[all]
```

Requires Python >= 3.12 and PyTorch >= 2.4.

## Quick Start

### Pure PyTorch Building Blocks

```python
import torch
from pyturboquant.core import (
    mse_quantize, mse_dequantize,
    ip_quantize, estimate_inner_product,
    random_rotate, random_rotate_inverse,
)

x = torch.randn(1000, 256)

# MSE-optimal quantization (Algorithm 1)
qt = mse_quantize(x, bits=3, seed=42)
x_hat = mse_dequantize(qt)
mse = ((x - x_hat) ** 2).sum(dim=-1).mean()
print(f"Normalized MSE: {mse / (x ** 2).sum(dim=-1).mean():.4f}")  # ~0.034

# Inner-product-preserving quantization (Algorithm 2)
y = torch.randn(1000, 256)
qt_ip = ip_quantize(x, bits=4, seed=42)
ip_est = estimate_inner_product(qt_ip, y)
ip_true = (x * y).sum(dim=-1)
print(f"IP error: {(ip_est - ip_true).abs().mean():.4f}")
```

### Nearest Neighbor Search

```python
import torch
from pyturboquant.search import TurboQuantIndex

# Build index -- near-zero indexing time (no training step)
index = TurboQuantIndex(dim=128, bits=4, metric="ip")
database = torch.randn(100_000, 128)
index.add(database)

print(f"Vectors: {index.ntotal}")
print(f"Memory:  {index.memory_usage_mb:.1f} MB")
print(f"Index time: {index.last_add_time_ms:.0f} ms")

# Search
queries = torch.randn(10, 128)
distances, indices = index.search(queries, k=10)

# Save / load
index.save("my_index.pt")
loaded = TurboQuantIndex.load("my_index.pt")
```

### LangChain RAG Pipeline

```python
from langchain_huggingface import HuggingFaceEmbeddings
from pyturboquant.search.langchain import TurboQuantVectorStore

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

store = TurboQuantVectorStore.from_texts(
    texts=["Document 1...", "Document 2...", "Document 3..."],
    embedding=embeddings,
    bits=4,
)

docs = store.similarity_search("my query", k=3)
for doc in docs:
    print(doc.page_content)

# Use as a standard LangChain retriever
retriever = store.as_retriever(search_kwargs={"k": 5})
```

### Class API (Stateful Quantizers)

```python
from pyturboquant.core import MSEQuantizer, InnerProductQuantizer

# Reusable quantizer -- rotation matrix and codebook created once
mse_q = MSEQuantizer(dim=256, bits=3, seed=42)
ip_q = InnerProductQuantizer(dim=256, bits=4, seed=42)

for batch in dataloader:
    qt = mse_q.quantize(batch)
    reconstructed = mse_q.dequantize(qt)
```

## Architecture

```
pyturboquant
├── core/              Pure PyTorch math (depends only on torch)
│   ├── rotation       Random orthogonal matrices via QR decomposition
│   ├── codebook       Lloyd-Max codebooks for Gaussian coordinates
│   ├── mse_quantizer  Algorithm 1: MSE-optimal vector quantization
│   ├── qjl            1-bit Quantized Johnson-Lindenstrauss transform
│   ├── prod_quantizer Algorithm 2: MSE + QJL for unbiased inner products
│   ├── packed         Bit-packing utilities for compact storage
│   └── types          QuantizedMSE, QuantizedIP, Codebook dataclasses
├── search/            Nearest neighbor search engine
│   ├── index          TurboQuantIndex with FAISS-like API
│   ├── distance       Asymmetric IP and L2 computation
│   └── langchain      LangChain VectorStore wrapper (optional)
└── utils/
    ├── beta_distribution  Sphere coordinate PDF (Lemma 1)
    └── metrics            MSE distortion, IP error, Shannon bounds
```

The `core` package has no dependencies beyond PyTorch. The `search` package builds on `core`. LangChain integration is guarded behind the `[langchain]` extra.

## How It Works

TurboQuant exploits a key insight: after applying a random orthogonal rotation to a unit vector in high dimensions, each coordinate becomes approximately independent and Gaussian with variance 1/d. This allows optimal *per-coordinate* scalar quantization using precomputed Lloyd-Max codebooks.

**MSE Quantizer (Algorithm 1):**
1. Extract and store the vector norm
2. Normalize to the unit sphere
3. Apply a seeded random rotation
4. Quantize each coordinate independently via `searchsorted` against Lloyd-Max boundaries
5. Pack indices into compact bit representation

**Inner Product Quantizer (Algorithm 2):**
1. Apply the MSE quantizer at (bits - 1) bits
2. Compute the quantization residual
3. Apply a 1-bit QJL transform (random projection + sign) to the residual
4. At query time, estimate `<x, y> ≈ <x_hat_mse, y> + QJL_estimate(<residual, y>)`

The result is an unbiased inner product estimator that enables high-recall approximate nearest neighbor search with zero indexing time.

## Empirical Distortion

Measured on 5,000 random unit vectors at d=256:

| Bits | Empirical MSE | Shannon Lower Bound | Ratio |
|------|--------------|-------------------|-------|
| 1    | 0.362        | 0.250             | 1.45x |
| 2    | 0.117        | 0.0625            | 1.87x |
| 3    | 0.034        | 0.0156            | 2.19x |
| 4    | 0.009        | 0.00391           | 2.41x |

These match the paper's Theorem 1 values within 1%.

## Retrieval Quality on BEIR

Measured with `BAAI/bge-base-en-v1.5` (d=768) on five [BEIR](https://github.com/beir-cellar/beir) datasets using `benchmarks/bench_beir.py`. The fp32 baseline is an exact chunked matmul over L2-normalised embeddings; pyturboquant uses `metric="ip"` (cosine similarity on unit vectors). Self-Recall@k is the overlap between the compressed top-k and the exact fp32 top-k -- it isolates quantiser error from embedding-model error.

| Dataset | n corpus | fp32 nDCG@10 | b=4 nDCG@10 | Δ b=4 | b=3 nDCG@10 | Δ b=3 | Compression (b=4) |
|---|---|---|---|---|---|---|---|
| SciFact | 5 183 | 0.7392 | 0.7281 | −1.5% | 0.7291 | −1.4% | 7.8× |
| NFCorpus | 3 633 | 0.3694 | 0.3677 | −0.5% | 0.3556 | −3.7% | 7.8× |
| FiQA-2018 | 57 638 | 0.3904 | 0.3816 | −2.3% | 0.3629 | −7.0% | 7.8× |
| SciDocs | 25 657 | 0.2207 | 0.2152 | −2.5% | 0.2057 | −6.8% | 7.8× |
| ArguAna | 8 674 | 0.6021 | 0.5801 | −3.7% | 0.5715 | −5.1% | 7.8× |
| **Average** | | **0.4644** | **0.4546** | **−2.1%** | **0.4449** | **−4.8%** | **7.8×** |

Memory at b=4 per dataset ranges from 1.36 MB (NFCorpus) to 21.55 MB (FiQA); the fp32 baseline ranges from 10.64 MB to 168.86 MB, a consistent **~7.8× reduction**.

Self-recall at b=4 averages 0.83 at k=10 and 0.86 at k=100 -- meaning about 1 in 6 top-10 results is swapped -- yet the downstream nDCG@10 loss is only 2.1%. The gap between self-recall and quality retention demonstrates that swapped results are nearly always equally relevant, which is the expected behaviour for a near-optimal unbiased inner-product estimator.

b=2 is available for extreme memory budgets but degrades more sharply (avg −11.4% nDCG@10, range 5.7–15.5% across datasets); prefer b≥3 for production deployments.

## Development

```bash
git clone https://github.com/jorgbahlmann/pyturboquant.git
cd pyturboquant
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run tests (162 tests)
pytest

# Lint
ruff check src/ tests/

# Precompute codebooks (requires scipy)
python scripts/precompute_codebooks.py

# Run benchmarks
python benchmarks/bench_distortion.py
python benchmarks/bench_nn_search.py
python benchmarks/bench_search_memory.py   # validates search-time memory bound
```

### Evaluate retrieval quality on BEIR

`benchmarks/bench_beir.py` quantifies how much retrieval quality is lost to compression on real IR benchmarks. For each bit setting it reports nDCG@10, Recall@100 and MAP@10 against BEIR's relevance judgments, plus self-recall (the overlap between the TurboQuant top-k and the fp32 top-k for the same embedding model), index memory, indexing time, and search latency.

```bash
# Install the bench extra (adds beir, sentence-transformers, pytrec-eval).
pip install -e ".[bench]"

# Small suite (SciFact, NFCorpus, FiQA, SciDocs, ArguAna) with BGE-base
# at bits 2/3/4 -- about 10-20 minutes end-to-end on a single GPU.
python benchmarks/bench_beir.py

# Fast single-dataset iteration with the lighter MiniLM encoder.
python benchmarks/bench_beir.py --datasets scifact --model minilm

# MSMARCO (8.84M passages, split=dev) at full scale. First run is encoding
# bound; subsequent runs hit the on-disk embedding cache under
# data/beir_cache/. Subsample via --max-corpus-size for quick checks.
python benchmarks/bench_beir.py --datasets msmarco --output-json msmarco.json
```

Available dataset keys: `scifact`, `nfcorpus`, `fiqa`, `scidocs`, `arguana`, `msmarco`, plus `small` (small suite) and `all` (small + msmarco). Embedding caches are keyed by `(model, dataset)` so sweeping `--bits` re-uses them automatically.

## Roadmap

- **v0.1.0** (current) -- Core quantizers, NN search index, LangChain VectorStore
- **v0.2.0** -- LlamaIndex `VectorStore` integration
- **v0.3.0** -- Triton fused kernels for GPU hot paths
- **v0.4.0** -- Haystack `DocumentStore` + Chroma / Weaviate adapters
- **v0.5.0** -- IVF partitioning for sublinear search on billion-scale corpora
- **Future** -- KV cache compression backend for inference runtimes (vLLM / SGLang plugin)

## Citation

```bibtex
@article{zandieh2025turboquant,
  title={TurboQuant: Online Vector Quantization with Near-Optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2504.19874},
  year={2025}
}
```

## License

MIT
