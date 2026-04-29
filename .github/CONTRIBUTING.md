# Contributing to pyturboquant

Thanks for your interest in contributing to **pyturboquant**! This project is a
GPU-accelerated Python implementation of Google's
[TurboQuant](https://arxiv.org/abs/2504.19874) algorithm, and we welcome
contributions of all sizes -- bug reports, documentation fixes, benchmarks,
integrations (LlamaIndex, Haystack, Chroma, ...), and core algorithmic work.

This document describes how to get set up, what we expect from contributions,
and how to get your change merged quickly.

## Code of conduct

By participating in this project you agree to keep interactions respectful and
constructive. Please be kind, assume good intent, and focus feedback on code
and ideas rather than people.

## Ways to contribute

- **Report bugs** -- open a [GitHub issue](https://github.com/jorgbahlmann/pyturboquant/issues)
  with a minimal reproducer.
- **Propose features or API changes** -- open an issue first to discuss scope
  and design before writing code. Larger changes (new quantizers, new
  integrations, public API additions) should be agreed on in an issue before a
  PR is opened.
- **Improve documentation** -- README fixes, docstring clarifications, and new
  examples are always welcome and do not require a prior issue.
- **Add benchmarks** -- new datasets for `benchmarks/bench_beir.py`, new
  workloads, or comparisons against FAISS / ScaNN / HNSW are highly valued.
- **Pick up a roadmap item** -- see the "Roadmap" section of the
  [README](../README.md) for in-flight work (LlamaIndex, Triton kernels, IVF,
  ...). Comment on or open a tracking issue before you start so we can avoid
  duplicate work.

## Reporting bugs

A good bug report includes:

1. **Environment**: OS, Python version, PyTorch version, CUDA version (if
   applicable), and pyturboquant version (`python -c "import pyturboquant;
   print(pyturboquant.__version__)"`).
2. **Minimal reproducer**: the smallest possible script that shows the issue.
   Use random tensors with a fixed seed when possible -- we cannot debug
   against proprietary embeddings.
3. **Expected vs. actual behavior**, including full tracebacks.
4. **Seed / determinism note**: if the bug is numerical, include the seed you
   used so we can reproduce it exactly. pyturboquant's rotations and QJL
   projections are seed-driven and fully deterministic.

## Suggesting enhancements

Before opening a PR for a non-trivial change, please open an issue describing:

- The problem the change solves.
- The proposed API (for public surface changes).
- Any performance or memory implications.
- Whether it touches the `core/`, `search/`, or integration layer.

This keeps us aligned on scope and saves you from rewriting a PR later.

## Development setup

```bash
git clone https://github.com/jorgbahlmann/pyturboquant.git
cd pyturboquant
python -m venv .venv && source .venv/bin/activate

# Dev install with tests, lint, and scipy (for codebook generation)
pip install -e ".[dev]"

# Optional extras, depending on what you are touching
pip install -e ".[langchain]"   # LangChain VectorStore
pip install -e ".[bench]"       # BEIR benchmark suite
pip install -e ".[all]"         # everything

# Enable pre-commit hooks (ruff + mypy on commit)
pre-commit install
```

Requires Python **>= 3.12** and PyTorch **>= 2.4**. A CUDA GPU is not required
for development, but GPU-marked tests (`-m gpu`) are skipped without one.

## Repository layout

See the "Architecture" section of the [README](../README.md) for the full
picture. Quick summary of where changes typically go:

- `src/pyturboquant/core/` -- pure PyTorch math (rotations, codebooks,
  quantizers, QJL, bit-packing). Only depends on `torch`.
- `src/pyturboquant/search/` -- `TurboQuantIndex`, distance computation, and
  optional integrations (LangChain today; LlamaIndex / Haystack / Chroma
  planned).
- `src/pyturboquant/utils/` -- shared helpers (beta distribution, metrics).
- `tests/` -- pytest test suite.
- `benchmarks/` -- reproducible performance and retrieval-quality benchmarks.
- `scripts/` -- developer utilities (e.g. `precompute_codebooks.py`).
- `data/codebooks/` -- precomputed Lloyd-Max codebooks shipped with the wheel.

New third-party integrations should live under `search/` behind an optional
extra in `pyproject.toml`, following the `langchain` extra as a template.

## Making a change

1. **Fork** the repository and create a topic branch from `main`:
   `git checkout -b feat/short-description` or `fix/short-description`.
2. Keep the change focused. One logical change per PR makes review much
   faster.
3. Update or add tests for any behavior change.
4. Update the README if you add a public API, a new extra, or a new CLI /
   benchmark entry point.
5. If you touch the public API, add a note to the changelog section of your PR
   description so we can roll it into the next release notes.

### Coding standards

- **Style** is enforced by `ruff` (config in `pyproject.toml`). Pre-commit will
  auto-fix most issues.
- **Types**: new code should be fully typed. `mypy` runs in strict-ish mode
  (`disallow_untyped_defs = true`) against the `pyturboquant` package.
- **Line length**: 100 characters.
- **Python version**: target 3.12+. Use modern syntax (`match`, PEP 604 union
  types, etc.).
- **Dependencies**: `core/` must not grow dependencies beyond `torch`.
  Everything else (scipy, sentence-transformers, beir, langchain-*) goes
  behind an optional extra.
- **Determinism**: any quantizer or index code path that uses randomness must
  accept a `seed` argument and be reproducible given the same seed.

### Running the checks locally

```bash
# Lint + format
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy

# Full test suite (162+ tests)
pytest

# Skip slow / GPU tests during iteration
pytest -m "not slow and not gpu"
```

Please make sure `ruff`, `mypy`, and `pytest` all pass before opening a PR.
CI will run the same checks.

### Benchmarks

If your change could affect performance (quantization speed, search latency,
memory, or retrieval quality), include before/after numbers from the relevant
benchmark in the PR description:

```bash
python benchmarks/bench_distortion.py
python benchmarks/bench_nn_search.py
python benchmarks/bench_search_memory.py
python benchmarks/bench_beir.py --datasets scifact --model minilm
```

For retrieval-quality changes, the BEIR small suite
(`python benchmarks/bench_beir.py`) is the canonical reference.

## Commit messages

We prefer short, imperative commit subjects (< 72 chars) following a lightweight
Conventional Commits style. Examples:

- `feat(core): add 5-bit Lloyd-Max codebook`
- `fix(search): guard against empty index in search()`
- `perf(search): vectorize asymmetric IP distance`
- `docs(readme): clarify search_batch_size semantics`
- `test(core): property tests for mse_quantize roundtrip`

Squash fix-up commits before requesting review where possible.

## Pull request process

1. Push your branch and open a PR against `main`.
2. Fill in the PR description:
   - **What** changed and **why**.
   - Linked issue (`Closes #123`) if applicable.
   - Benchmark numbers or test output for performance/quality changes.
   - Any API-visible changes a user should know about.
3. Make sure CI is green (lint, type check, tests).
4. A maintainer will review. Please respond to review comments by pushing
   additional commits; we will squash on merge.
5. Once approved and green, a maintainer will merge. You do not need to
   rebase unless asked -- GitHub's "Squash and merge" is the default.

### What we look for in review

- Correctness: tests cover the new behavior, edge cases are considered.
- API design: public additions are minimal, typed, and consistent with the
  FAISS-like naming already in `TurboQuantIndex`.
- Memory and determinism: changes respect the `search_batch_size` bound and
  stay seed-reproducible.
- Docs: README and docstrings updated where relevant.

## Releasing (maintainers)

Releases follow semantic versioning. The version lives in
`src/pyturboquant/_version.py`. Maintainers cut a release by:

1. Updating `_version.py` and the "Roadmap" section of the README.
2. Tagging `vX.Y.Z` on `main`.
3. Building and publishing via the usual `python -m build` + `twine upload`
   flow.

## License

pyturboquant is MIT licensed (see [`LICENSE`](../LICENSE)). By submitting a
contribution you agree that your work will be distributed under the same
license.

## Questions?

If something here is unclear, open a
[GitHub issue](https://github.com/jorgbahlmann/pyturboquant/issues) and tag it
`question` -- we would rather improve this doc than have you guess. Thanks for
helping make pyturboquant better!
