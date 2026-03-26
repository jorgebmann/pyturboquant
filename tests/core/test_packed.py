"""Tests for bit-packing utilities."""

from __future__ import annotations

import pytest
import torch

from pyturboquant.core.packed import pack_bits, pack_indices, unpack_bits, unpack_indices


class TestPackIndices:
    """Round-trip tests for index packing."""

    @pytest.mark.parametrize("bits", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_round_trip(self, bits: int) -> None:
        n = 128
        max_val = (1 << bits) - 1
        g = torch.Generator().manual_seed(42)
        indices = torch.randint(0, max_val + 1, (n,), generator=g, dtype=torch.int32)
        packed = pack_indices(indices, bits)
        unpacked = unpack_indices(packed, bits, n)
        torch.testing.assert_close(unpacked, indices)

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_small_counts(self, bits: int) -> None:
        for count in [1, 2, 7, 15]:
            indices = torch.arange(count, dtype=torch.int32) % (1 << bits)
            packed = pack_indices(indices, bits)
            unpacked = unpack_indices(packed, bits, count)
            torch.testing.assert_close(unpacked, indices)

    def test_invalid_bits(self) -> None:
        with pytest.raises(ValueError):
            pack_indices(torch.zeros(8, dtype=torch.int32), bits=0)
        with pytest.raises(ValueError):
            pack_indices(torch.zeros(8, dtype=torch.int32), bits=9)


class TestPackBits:
    """Round-trip tests for sign-bit packing."""

    def test_round_trip_01(self) -> None:
        n = 100
        g = torch.Generator().manual_seed(0)
        signs = torch.randint(0, 2, (n,), generator=g, dtype=torch.uint8)
        packed = pack_bits(signs)
        unpacked = unpack_bits(packed, n)
        expected = signs.float() * 2.0 - 1.0  # 0/1 -> -1/+1
        torch.testing.assert_close(unpacked, expected)

    def test_round_trip_pm1(self) -> None:
        n = 77
        g = torch.Generator().manual_seed(1)
        signs = torch.randint(0, 2, (n,), generator=g).float() * 2.0 - 1.0
        packed = pack_bits(signs)
        unpacked = unpack_bits(packed, n)
        torch.testing.assert_close(unpacked, signs)

    def test_all_positive(self) -> None:
        signs = torch.ones(16)
        packed = pack_bits(signs)
        unpacked = unpack_bits(packed, 16)
        assert (unpacked == 1.0).all()

    def test_all_negative(self) -> None:
        signs = -torch.ones(16)
        packed = pack_bits(signs)
        unpacked = unpack_bits(packed, 16)
        assert (unpacked == -1.0).all()
