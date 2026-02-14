# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache compression policies for physical KV cache compression.

These policies run on GPU (worker side) and decide which tokens' KV entries
to keep after prefill, enabling physical compression of the KV cache.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class CompressionPolicyConfig:
    """Configuration for KV cache compression."""
    policy_type: str  # "knorm", "streaming_llm", "expected_attention", etc.
    keep_ratio: float = 0.5  # Fraction of tokens to keep
    keep_first: int = 4  # Always keep first N tokens (attention sinks)
    keep_last: int = 64  # Always keep last N tokens (recent context)


class KVCacheCompressionPolicy(ABC):
    """Base class for KV cache compression policies.

    Policies run on the GPU worker and determine which token KV entries
    to retain based on the KV cache tensor values.
    """

    @abstractmethod
    def select_tokens_to_keep(
        self,
        kv_cache: torch.Tensor,
        block_ids: list[int],
        seq_len: int,
        target_len: int,
        block_size: int,
        keep_first: int,
        keep_last: int,
    ) -> torch.Tensor:
        """Select which tokens to keep from the KV cache.

        Args:
            kv_cache: KV cache tensor [2, num_blocks, block_size,
                      num_kv_heads, head_size]. Dim 0: key(0) / value(1).
            block_ids: Physical block IDs for this request.
            seq_len: Current total sequence length.
            target_len: Number of tokens to keep after compression.
            block_size: Tokens per block.
            keep_first: Always keep first N tokens (attention sinks).
            keep_last: Always keep last N tokens (recent context).

        Returns:
            Sorted 1D tensor of token indices to keep, shape [target_len].
        """
        ...

    def _gather_keys(
        self,
        kv_cache: torch.Tensor,
        block_ids: list[int],
        seq_len: int,
        block_size: int,
    ) -> torch.Tensor:
        """Gather key vectors for all tokens in the sequence.

        Returns:
            Keys tensor [seq_len, num_kv_heads, head_size]
        """
        device = kv_cache.device
        positions = torch.arange(seq_len, device=device)
        block_indices = positions // block_size
        offsets = positions % block_size

        block_ids_t = torch.tensor(
            block_ids, device=device, dtype=torch.int64
        )
        physical_block_ids = block_ids_t[block_indices]

        # kv_cache[0] = key cache: [num_blocks, block_size, num_kv_heads, head_size]
        keys = kv_cache[0, physical_block_ids, offsets]  # [seq_len, num_kv_heads, head_size]
        return keys

    def _split_indices(
        self,
        seq_len: int,
        target_len: int,
        keep_first: int,
        keep_last: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Split the sequence into mandatory-keep and candidate regions.

        Returns:
            (mandatory_indices, candidate_indices, num_to_select_from_candidates)
        """
        keep_first = min(keep_first, seq_len)
        keep_last = min(keep_last, seq_len - keep_first)

        first_indices = torch.arange(keep_first, device=device)
        last_indices = torch.arange(
            seq_len - keep_last, seq_len, device=device
        )
        mandatory = torch.cat([first_indices, last_indices])

        # Candidate region: between keep_first and seq_len - keep_last
        candidate_start = keep_first
        candidate_end = seq_len - keep_last
        if candidate_end <= candidate_start:
            # Nothing to select from candidates; mandatory covers everything
            return mandatory, torch.tensor([], device=device, dtype=torch.long), 0

        candidates = torch.arange(candidate_start, candidate_end, device=device)
        num_to_select = target_len - len(mandatory)
        num_to_select = max(0, min(num_to_select, len(candidates)))

        return mandatory, candidates, num_to_select


class StreamingLLMCompressionPolicy(KVCacheCompressionPolicy):
    """StreamingLLM: keep first (attention sink) + last (recent) tokens.

    Reference: Efficient Streaming Language Models with Attention Sinks
    """

    def select_tokens_to_keep(
        self,
        kv_cache: torch.Tensor,
        block_ids: list[int],
        seq_len: int,
        target_len: int,
        block_size: int,
        keep_first: int,
        keep_last: int,
    ) -> torch.Tensor:
        device = kv_cache.device
        mandatory, _, _ = self._split_indices(
            seq_len, target_len, keep_first, keep_last, device
        )
        # StreamingLLM only keeps the mandatory tokens
        # Adjust keep_last if target_len requires more
        actual_keep_first = min(keep_first, seq_len)
        actual_keep_last = min(target_len - actual_keep_first, seq_len - actual_keep_first)
        actual_keep_last = max(0, actual_keep_last)

        first_indices = torch.arange(actual_keep_first, device=device)
        last_indices = torch.arange(
            seq_len - actual_keep_last, seq_len, device=device
        )
        result = torch.cat([first_indices, last_indices])
        result = result[:target_len]
        result, _ = torch.sort(result)
        return result


class KnormCompressionPolicy(KVCacheCompressionPolicy):
    """KnormPress: keep tokens with highest key vector L2 norms.

    Reference: KVPress - Key-Value Cache Compression
    Tokens with larger key norms tend to contribute more to attention.
    """

    def select_tokens_to_keep(
        self,
        kv_cache: torch.Tensor,
        block_ids: list[int],
        seq_len: int,
        target_len: int,
        block_size: int,
        keep_first: int,
        keep_last: int,
    ) -> torch.Tensor:
        device = kv_cache.device
        mandatory, candidates, num_to_select = self._split_indices(
            seq_len, target_len, keep_first, keep_last, device
        )

        if num_to_select == 0 or len(candidates) == 0:
            result, _ = torch.sort(mandatory)
            return result[:target_len]

        # Gather keys for candidate tokens
        keys = self._gather_keys(kv_cache, block_ids, seq_len, block_size)
        candidate_keys = keys[candidates]  # [num_candidates, num_kv_heads, head_size]

        # Compute L2 norm across heads and head_size
        # Average norms across heads for a single importance score per token
        norms = torch.norm(
            candidate_keys.float(), dim=-1
        ).mean(dim=-1)  # [num_candidates]

        # Select top-k by norm
        _, topk_indices = torch.topk(norms, num_to_select)
        selected = candidates[topk_indices]

        result = torch.cat([mandatory, selected])
        result, _ = torch.sort(result)
        return result[:target_len]


class ExpectedAttentionCompressionPolicy(KVCacheCompressionPolicy):
    """ExpectedAttentionPress: estimate expected attention weight per token.

    Approximates the attention a token would receive based on key magnitude
    relative to the average key, without computing full attention.
    """

    def select_tokens_to_keep(
        self,
        kv_cache: torch.Tensor,
        block_ids: list[int],
        seq_len: int,
        target_len: int,
        block_size: int,
        keep_first: int,
        keep_last: int,
    ) -> torch.Tensor:
        device = kv_cache.device
        mandatory, candidates, num_to_select = self._split_indices(
            seq_len, target_len, keep_first, keep_last, device
        )

        if num_to_select == 0 or len(candidates) == 0:
            result, _ = torch.sort(mandatory)
            return result[:target_len]

        keys = self._gather_keys(kv_cache, block_ids, seq_len, block_size)
        candidate_keys = keys[candidates].float()  # [N, num_kv_heads, head_size]

        # Expected attention approximation:
        # Score per token = ||k_i||^2 (proportional to expected attention weight)
        # This is a simplification; full ExpectedAttention uses softmax normalization
        scores = (candidate_keys ** 2).sum(dim=-1).mean(dim=-1)  # [N]

        _, topk_indices = torch.topk(scores, num_to_select)
        selected = candidates[topk_indices]

        result = torch.cat([mandatory, selected])
        result, _ = torch.sort(result)
        return result[:target_len]


class ThinKCompressionPolicy(KVCacheCompressionPolicy):
    """ThinKPress: select tokens based on key channel importance.

    Keeps tokens that have the most distinctive key patterns by
    looking at the variance of key channels.
    """

    def select_tokens_to_keep(
        self,
        kv_cache: torch.Tensor,
        block_ids: list[int],
        seq_len: int,
        target_len: int,
        block_size: int,
        keep_first: int,
        keep_last: int,
    ) -> torch.Tensor:
        device = kv_cache.device
        mandatory, candidates, num_to_select = self._split_indices(
            seq_len, target_len, keep_first, keep_last, device
        )

        if num_to_select == 0 or len(candidates) == 0:
            result, _ = torch.sort(mandatory)
            return result[:target_len]

        keys = self._gather_keys(kv_cache, block_ids, seq_len, block_size)
        candidate_keys = keys[candidates].float()  # [N, num_kv_heads, head_size]

        # ThinK: channel-wise importance
        # Compute L1 norm per channel, pick tokens with largest total deviation
        mean_key = candidate_keys.mean(dim=0, keepdim=True)  # [1, H, D]
        deviation = (candidate_keys - mean_key).abs().sum(dim=-1).mean(dim=-1)  # [N]

        _, topk_indices = torch.topk(deviation, num_to_select)
        selected = candidates[topk_indices]

        result = torch.cat([mandatory, selected])
        result, _ = torch.sort(result)
        return result[:target_len]


class KeyDiffCompressionPolicy(KVCacheCompressionPolicy):
    """KeyDiffPress: keep tokens with most unique keys (high difference
    from neighbors).

    Tokens whose keys are very different from adjacent tokens carry more
    unique information.
    """

    def select_tokens_to_keep(
        self,
        kv_cache: torch.Tensor,
        block_ids: list[int],
        seq_len: int,
        target_len: int,
        block_size: int,
        keep_first: int,
        keep_last: int,
    ) -> torch.Tensor:
        device = kv_cache.device
        mandatory, candidates, num_to_select = self._split_indices(
            seq_len, target_len, keep_first, keep_last, device
        )

        if num_to_select == 0 or len(candidates) == 0:
            result, _ = torch.sort(mandatory)
            return result[:target_len]

        keys = self._gather_keys(
            kv_cache, block_ids, seq_len, block_size
        ).float()  # [seq_len, num_kv_heads, head_size]

        # Compute difference from neighbors for candidate positions
        candidate_keys = keys[candidates]  # [N, H, D]

        # Left neighbor diff
        left_indices = candidates - 1
        left_indices = left_indices.clamp(min=0)
        left_keys = keys[left_indices]
        left_diff = torch.norm(candidate_keys - left_keys, dim=-1).mean(dim=-1)

        # Right neighbor diff
        right_indices = candidates + 1
        right_indices = right_indices.clamp(max=seq_len - 1)
        right_keys = keys[right_indices]
        right_diff = torch.norm(candidate_keys - right_keys, dim=-1).mean(dim=-1)

        # Score = average of left and right differences
        scores = (left_diff + right_diff) / 2.0

        _, topk_indices = torch.topk(scores, num_to_select)
        selected = candidates[topk_indices]

        result = torch.cat([mandatory, selected])
        result, _ = torch.sort(result)
        return result[:target_len]


class LagKVCompressionPolicy(KVCacheCompressionPolicy):
    """LagKVPress: keep tokens based on lagged key correlation.

    Identifies tokens that introduce new information by comparing
    each key with a lagged version of itself.
    """

    def __init__(self, lag: int = 1):
        self.lag = lag

    def select_tokens_to_keep(
        self,
        kv_cache: torch.Tensor,
        block_ids: list[int],
        seq_len: int,
        target_len: int,
        block_size: int,
        keep_first: int,
        keep_last: int,
    ) -> torch.Tensor:
        device = kv_cache.device
        mandatory, candidates, num_to_select = self._split_indices(
            seq_len, target_len, keep_first, keep_last, device
        )

        if num_to_select == 0 or len(candidates) == 0:
            result, _ = torch.sort(mandatory)
            return result[:target_len]

        keys = self._gather_keys(
            kv_cache, block_ids, seq_len, block_size
        ).float()  # [seq_len, H, D]

        # Compare each candidate with its lagged counterpart
        candidate_keys = keys[candidates]  # [N, H, D]
        lagged_indices = (candidates - self.lag).clamp(min=0)
        lagged_keys = keys[lagged_indices]  # [N, H, D]

        # Score = L2 distance from lagged key (higher = more novel information)
        scores = torch.norm(
            candidate_keys - lagged_keys, dim=-1
        ).mean(dim=-1)  # [N]

        _, topk_indices = torch.topk(scores, num_to_select)
        selected = candidates[topk_indices]

        result = torch.cat([mandatory, selected])
        result, _ = torch.sort(result)
        return result[:target_len]


# Registry of available compression policies
COMPRESSION_POLICIES: dict[str, type[KVCacheCompressionPolicy]] = {
    "streaming_llm": StreamingLLMCompressionPolicy,
    "knorm": KnormCompressionPolicy,
    "expected_attention": ExpectedAttentionCompressionPolicy,
    "think": ThinKCompressionPolicy,
    "keydiff": KeyDiffCompressionPolicy,
    "lagkv": LagKVCompressionPolicy,
}


def get_compression_policy(policy_type: str) -> KVCacheCompressionPolicy:
    """Get a compression policy instance by name."""
    cls = COMPRESSION_POLICIES.get(policy_type)
    if cls is None:
        raise ValueError(
            f"Unknown compression policy: {policy_type}. "
            f"Available: {list(COMPRESSION_POLICIES.keys())}"
        )
    return cls()
