"""Shared test fixtures and markers for pyturboquant."""

from __future__ import annotations

import pytest
import torch

DEFAULT_SEED = 42
TEST_DIMS = [64, 128, 256]
TEST_BITWIDTHS = [1, 2, 3, 4]


@pytest.fixture
def rng() -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(DEFAULT_SEED)
    return g


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(params=TEST_DIMS, ids=[f"d={d}" for d in TEST_DIMS])
def dim(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture(params=TEST_BITWIDTHS, ids=[f"b={b}" for b in TEST_BITWIDTHS])
def bits(request: pytest.FixtureRequest) -> int:
    return request.param
