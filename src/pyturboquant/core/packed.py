"""Bit-packing utilities for compact storage of b-bit quantization indices.

Packs b-bit integers into uint8 tensors for memory-efficient storage.
"""

from __future__ import annotations

import torch


def pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack b-bit index values into a compact uint8 tensor.

    Args:
        indices: Integer tensor with values in [0, 2^bits). Shape (*, d).
        bits: Number of bits per index (1-8).

    Returns:
        Packed uint8 tensor. Shape (*, ceil(d * bits / 8)).
    """
    if bits < 1 or bits > 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")
    if bits == 8:
        return indices.to(torch.uint8)

    flat = indices.reshape(-1).to(torch.int32)
    n = flat.numel()
    total_bits = n * bits
    n_bytes = (total_bits + 7) // 8

    packed = torch.zeros(n_bytes, dtype=torch.uint8, device=indices.device)

    bit_offset = 0
    for i in range(n):
        byte_idx = bit_offset // 8
        bit_pos = bit_offset % 8
        val = flat[i].item()

        # May span two bytes
        packed[byte_idx] |= (val << bit_pos) & 0xFF
        overflow = bit_pos + bits - 8
        if overflow > 0 and byte_idx + 1 < n_bytes:
            packed[byte_idx + 1] |= (val >> (bits - overflow)) & 0xFF

        bit_offset += bits

    return packed


def unpack_indices(packed: torch.Tensor, bits: int, count: int) -> torch.Tensor:
    """Unpack b-bit index values from a compact uint8 tensor.

    Args:
        packed: Packed uint8 tensor from pack_indices.
        bits: Number of bits per index (1-8).
        count: Number of indices to unpack.

    Returns:
        Integer tensor of shape (count,) with values in [0, 2^bits).
    """
    if bits < 1 or bits > 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")
    if bits == 8:
        return packed[:count].to(torch.int32)

    mask = (1 << bits) - 1
    result = torch.zeros(count, dtype=torch.int32, device=packed.device)

    bit_offset = 0
    for i in range(count):
        byte_idx = bit_offset // 8
        bit_pos = bit_offset % 8
        val = (packed[byte_idx].item() >> bit_pos) & mask

        overflow = bit_pos + bits - 8
        if overflow > 0 and byte_idx + 1 < len(packed):
            val |= (packed[byte_idx + 1].item() << (8 - bit_pos)) & mask

        result[i] = val
        bit_offset += bits

    return result


def pack_bits(signs: torch.Tensor) -> torch.Tensor:
    """Pack a boolean/int8 sign tensor into uint8 bitfield.

    Args:
        signs: Tensor of 0/1 or -1/+1 values. Shape (*, d).

    Returns:
        Packed uint8 tensor. Shape (*, ceil(d / 8)).
    """
    flat = (signs.reshape(-1) > 0).to(torch.uint8)
    n = flat.numel()
    n_bytes = (n + 7) // 8
    padded = torch.zeros(n_bytes * 8, dtype=torch.uint8, device=signs.device)
    padded[:n] = flat

    reshaped = padded.reshape(n_bytes, 8)
    shifts = torch.arange(8, device=signs.device, dtype=torch.uint8)
    packed = (reshaped << shifts).sum(dim=1).to(torch.uint8)
    return packed


def unpack_bits(packed: torch.Tensor, count: int) -> torch.Tensor:
    """Unpack uint8 bitfield into a sign tensor of +1/-1.

    Args:
        packed: Packed uint8 tensor from pack_bits.
        count: Number of sign values to unpack.

    Returns:
        Tensor of shape (count,) with values in {-1, +1}.
    """
    shifts = torch.arange(8, device=packed.device, dtype=torch.uint8)
    unpacked = ((packed.unsqueeze(-1) >> shifts) & 1).reshape(-1)[:count]
    return unpacked.to(torch.float32) * 2.0 - 1.0
