# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode timing utility for measuring per-module CUDA execution time
during online serving.

Enable via environment variables:
    VLLM_DECODE_TIMING=1              # Enable timing
    VLLM_DECODE_TIMING_INTERVAL=100   # Report every N iterations
    VLLM_DECODE_TIMING_CSV=timing.csv # Optional CSV output path
    VLLM_DECODE_METADATA_CSV=meta.csv # Optional per-request metadata CSV

Must use --enforce-eager since CUDA graphs prevent per-op timing.
"""

import csv
import os
import time
from collections import defaultdict

import torch
import torch.nn as nn

from vllm.logger import init_logger

logger = init_logger(__name__)

# Module types we want to time (imported lazily to avoid circular imports)
_TRACKED_MODULE_TYPES: tuple[type, ...] | None = None


def _get_tracked_module_types() -> tuple[type, ...]:
    global _TRACKED_MODULE_TYPES
    if _TRACKED_MODULE_TYPES is None:
        from vllm.model_executor.layers.attention.attention import Attention
        from vllm.model_executor.layers.linear import (
            ColumnParallelLinear,
            MergedColumnParallelLinear,
            QKVParallelLinear,
            RowParallelLinear,
        )
        from vllm.model_executor.layers.layernorm import RMSNorm

        _TRACKED_MODULE_TYPES = (
            Attention,
            ColumnParallelLinear,
            MergedColumnParallelLinear,
            QKVParallelLinear,
            RowParallelLinear,
            RMSNorm,
        )
    return _TRACKED_MODULE_TYPES


class DecodeTimer:
    """Collects per-module CUDA timing via forward hooks during serving.

    Usage:
        timer = DecodeTimer(enabled=True, log_interval=100)
        timer.attach_hooks(model)
        # ... in execute_model loop ...
        timer.on_iteration_end()
    """

    def __init__(
        self,
        enabled: bool = False,
        log_interval: int = 10,
        csv_path: str | None = None,
        metadata_csv_path: str | None = None,
    ):
        self.enabled = enabled
        self.log_interval = log_interval
        self.csv_path = csv_path
        self.metadata_csv_path = metadata_csv_path
        self._metadata_csv_written_header = False

        if not self.enabled:
            return

        # Per-iteration event pairs: {name: [(start_event, end_event), ...]}
        self._current_events: dict[str, list[tuple[torch.cuda.Event,
                                                    torch.cuda.Event]]] = (
            defaultdict(list))

        # Cumulative stats: {name: {"count": int, "total_ms": float}}
        self._cumulative: dict[str, dict] = defaultdict(
            lambda: {
                "count": 0,
                "total_ms": 0.0,
                "min_ms": float("inf"),
                "max_ms": 0.0,
            })

        self._iteration_count = 0
        self._csv_written_header = False
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._start_time = time.monotonic()

    def attach_hooks(self, model: nn.Module) -> None:
        """Register forward pre/post hooks on tracked module types."""
        if not self.enabled:
            return

        tracked_types = _get_tracked_module_types()

        for name, module in model.named_modules():
            if isinstance(module, tracked_types):
                pre_hook = module.register_forward_pre_hook(
                    self._make_pre_hook(name))
                post_hook = module.register_forward_hook(
                    self._make_post_hook(name))
                self._hooks.extend([pre_hook, post_hook])

        logger.info("DecodeTimer: attached hooks to %d modules, "
                     "reporting every %d iterations",
                     len(self._hooks) // 2, self.log_interval)

    def _make_pre_hook(self, name: str):
        def hook(module, input):
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            # Store start event on module for retrieval in post hook
            module._decode_timer_start = start

        return hook

    def _make_post_hook(self, name: str):
        def hook(module, input, output):
            start = getattr(module, "_decode_timer_start", None)
            if start is None:
                return
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self._current_events[name].append((start, end))
            del module._decode_timer_start

        return hook

    def on_iteration_end(self) -> None:
        """Call after each execute_model. Syncs GPU and accumulates stats."""
        if not self.enabled:
            return

        self._iteration_count += 1

        # Synchronize to get accurate timings
        torch.cuda.synchronize()

        # Accumulate current iteration events into cumulative stats
        for name, event_pairs in self._current_events.items():
            for start, end in event_pairs:
                elapsed = start.elapsed_time(end)  # milliseconds
                stats = self._cumulative[name]
                stats["count"] += 1
                stats["total_ms"] += elapsed
                stats["min_ms"] = min(stats["min_ms"], elapsed)
                stats["max_ms"] = max(stats["max_ms"], elapsed)

        self._current_events.clear()

        # Report at interval
        if self._iteration_count % self.log_interval == 0:
            self._report()

    def _report(self) -> None:
        """Write timing data to CSV."""
        if not self._cumulative:
            return

        total_ms = sum(s["total_ms"] for s in self._cumulative.values())
        sorted_stats = sorted(self._cumulative.items(),
                              key=lambda x: x[1]["total_ms"],
                              reverse=True)

        if self.csv_path:
            self._write_csv(sorted_stats, total_ms)

    def _write_csv(self, sorted_stats, total_ms: float) -> None:
        """Append timing data to CSV file."""
        write_header = not self._csv_written_header

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "iteration", "module", "count", "total_ms", "avg_ms",
                    "min_ms", "max_ms", "pct"
                ])
                self._csv_written_header = True

            for name, stats in sorted_stats:
                if stats["count"] == 0:
                    continue
                avg = stats["total_ms"] / stats["count"]
                pct = (stats["total_ms"] / total_ms *
                       100) if total_ms > 0 else 0.0
                writer.writerow([
                    self._iteration_count, name, stats["count"],
                    f"{stats['total_ms']:.3f}", f"{avg:.4f}",
                    f"{stats['min_ms']:.4f}", f"{stats['max_ms']:.4f}",
                    f"{pct:.1f}"
                ])

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def record_batch_metadata(
        self,
        batch_metadata: list[dict[str, object]],
    ) -> None:
        """Record per-request metadata for the current iteration.

        Each entry in batch_metadata should contain:
            req_id, is_decode, num_scheduled_tokens, seq_len, num_prompt_tokens
        """
        if not self.enabled or not self.metadata_csv_path:
            return

        num_decode_reqs = sum(1 for m in batch_metadata if m["is_decode"])
        num_prefill_reqs = len(batch_metadata) - num_decode_reqs
        total_decode_tokens = sum(
            m["num_scheduled_tokens"]
            for m in batch_metadata
            if m["is_decode"]
        )
        total_prefill_tokens = sum(
            m["num_scheduled_tokens"]
            for m in batch_metadata
            if not m["is_decode"]
        )

        write_header = not self._metadata_csv_written_header
        with open(self.metadata_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "iteration", "req_id", "is_decode",
                    "num_scheduled_tokens", "seq_len", "logical_seq_len",
                    "num_prompt_tokens",
                    "num_decode_reqs", "num_prefill_reqs",
                    "total_decode_tokens", "total_prefill_tokens",
                ])
                self._metadata_csv_written_header = True

            iteration = self._iteration_count + 1  # will be incremented in on_iteration_end
            for m in batch_metadata:
                writer.writerow([
                    iteration,
                    m["req_id"],
                    m["is_decode"],
                    m["num_scheduled_tokens"],
                    m["seq_len"],
                    m.get("logical_seq_len", m["seq_len"]),
                    m["num_prompt_tokens"],
                    num_decode_reqs,
                    num_prefill_reqs,
                    total_decode_tokens,
                    total_prefill_tokens,
                ])

    def reset(self) -> None:
        """Reset all accumulated statistics."""
        self._current_events.clear()
        self._cumulative.clear()
        self._iteration_count = 0
        self._start_time = time.monotonic()
        self._csv_written_header = False
        self._metadata_csv_written_header = False
        if self.csv_path and os.path.exists(self.csv_path):
            os.remove(self.csv_path)
        if self.metadata_csv_path and os.path.exists(self.metadata_csv_path):
            os.remove(self.metadata_csv_path)
