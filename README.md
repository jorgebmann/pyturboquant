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
