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
        Packed uint8 tensor. Shape (ceil(n * bits / 8),) for n = indices.numel().
    """
    if bits < 1 or bits > 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")
    if bits == 8:
        return indices.reshape(-1).to(torch.uint8)

    flat = indices.reshape(-1).to(torch.int64)
    device = flat.device
    n = flat.numel()
    if n == 0:
        return torch.zeros(0, dtype=torch.uint8, device=device)

    k_arange = torch.arange(bits, device=device, dtype=torch.int64)
    bit_vals = ((flat.unsqueeze(-1) >> k_arange) & 1).to(torch.uint8)
    bits_stream = bit_vals.reshape(-1)
    total_bits = bits_stream.numel()
    n_bytes = (total_bits + 7) // 8
    pad_len = n_bytes * 8 - total_bits
    if pad_len:
        bits_stream = torch.cat(
            [bits_stream, torch.zeros(pad_len, dtype=torch.uint8, device=device)]
        )
    reshaped = bits_stream.view(n_bytes, 8)
    shifts = torch.arange(8, device=device, dtype=torch.int32)
    return (reshaped.to(torch.int32) << shifts).sum(dim=1).to(torch.uint8)


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
        return packed[:count].reshape(-1).to(torch.int32)

    device = packed.device
    if count == 0:
        return torch.zeros(0, dtype=torch.int32, device=device)

    total_bits = count * bits
    shifts_u8 = torch.arange(8, device=device, dtype=torch.uint8)
    bits_from_bytes = ((packed.unsqueeze(-1) >> shifts_u8) & 1).reshape(-1)
    stream = bits_from_bytes[:total_bits].to(torch.int64)
    mat = stream.view(count, bits)
    powers = 1 << torch.arange(bits, device=device, dtype=torch.int64)
    return (mat * powers).sum(dim=-1).to(torch.int32)


def pack_bits(signs: torch.Tensor) -> torch.Tensor:
    """Pack a boolean/int8 sign tensor into uint8 bitfield.

    Args:
        signs: Tensor of 0/1 or -1/+1 values. Any shape; values are packed in
        flattened (row-major) order into a single 1D uint8 tensor.

    Returns:
        Packed uint8 tensor. Shape (ceil(signs.numel() / 8),).
    """
    flat = (signs.reshape(-1) > 0).to(torch.uint8)
    n = flat.numel()
    n_bytes = (n + 7) // 8
    padded = torch.zeros(n_bytes * 8, dtype=torch.uint8, device=signs.device)
    padded[:n] = flat

    reshaped = padded.reshape(n_bytes, 8)
    shifts = torch.arange(8, device=signs.device, dtype=torch.int32)
    return (reshaped.to(torch.int32) << shifts).sum(dim=1).to(torch.uint8)


def pack_bits_batch(signs: torch.Tensor) -> torch.Tensor:
    """Vectorized sign packing for a batch of shape (n, d)."""
    if signs.ndim < 2:
        raise ValueError("pack_bits_batch expects signs of shape (n, d)")
    n, m = signs.shape
    device = signs.device
    flat = (signs > 0).to(torch.uint8)
    n_bytes = (m + 7) // 8
    padded_cols = n_bytes * 8
    padded = torch.zeros(n, padded_cols, dtype=torch.uint8, device=device)
    padded[:, :m] = flat
    reshaped = padded.view(n, n_bytes, 8)
    shifts = torch.arange(8, device=device, dtype=torch.int32)
    return (reshaped.to(torch.int32) << shifts).sum(dim=-1).to(torch.uint8)


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


def unpack_bits_batch(packed: torch.Tensor, count: int) -> torch.Tensor:
    """Unpack batch packed rows (n, n_bytes) to signs (n, count), values in {-1, +1}."""
    n = packed.shape[0]
    device = packed.device
    shifts = torch.arange(8, device=device, dtype=torch.uint8)
    bits = ((packed.unsqueeze(-1) >> shifts) & 1).reshape(n, -1)[:, :count]
    return bits.to(torch.float32) * 2.0 - 1.0
