# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU KV cache compaction: rearrange kept KV entries into contiguous blocks.

After a compression policy selects which tokens to keep, this module
physically moves the KV data so the kept entries occupy the first M blocks
contiguously, allowing the remaining blocks to be freed.
"""

import torch


def compact_kv_cache(
    kv_cache: torch.Tensor,
    block_ids: list[int],
    kept_token_indices: torch.Tensor,
    block_size: int,
) -> None:
    """Compact kept KV entries into contiguous positions in the first M blocks.

    This function moves KV cache entries in-place. It uses a staging buffer
    to handle overlapping source/destination safely.

    Args:
        kv_cache: KV cache tensor with shape
            [2, num_blocks, block_size, num_kv_heads, head_size].
            Dim 0: key (0) / value (1).
        block_ids: Physical block IDs for this request (logical order).
        kept_token_indices: Sorted 1D tensor of token positions (in the
            original sequence) to keep. Shape [compressed_len].
        block_size: Number of tokens per block.
    """
    compressed_len = kept_token_indices.shape[0]
    if compressed_len == 0:
        return

    device = kv_cache.device

    # Compute source (block_id, offset) from the original token positions
    src_block_indices = kept_token_indices // block_size
    src_offsets = kept_token_indices % block_size

    # Compute destination (block_id, offset) for contiguous packing
    dst_positions = torch.arange(compressed_len, device=device)
    dst_block_indices = dst_positions // block_size
    dst_offsets = dst_positions % block_size

    # Map logical block indices to physical block IDs
    block_ids_tensor = torch.tensor(
        block_ids, device=device, dtype=torch.int64
    )
    src_physical = block_ids_tensor[src_block_indices]
    dst_physical = block_ids_tensor[dst_block_indices]

    # Filter out no-op copies (src == dst)
    needs_copy = (src_physical != dst_physical) | (src_offsets != dst_offsets)
    if not needs_copy.any():
        return

    src_p = src_physical[needs_copy]
    src_o = src_offsets[needs_copy]
    dst_p = dst_physical[needs_copy]
    dst_o = dst_offsets[needs_copy]

    # Batch gather into staging buffer, then scatter to destinations.
    # This avoids aliasing issues when source and destination overlap.
    # kv_cache shape: [2, num_blocks, block_size, num_kv_heads, head_size]
    buf = kv_cache[:, src_p, src_o].clone()  # [2, N, num_kv_heads, head_size]
    kv_cache[:, dst_p, dst_o] = buf
