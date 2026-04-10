# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Compression-Aware Adaptive Scheduler (CAAS)

Addresses the TPOT degradation caused by KV cache compression: when compression
frees KV blocks the scheduler greedily admits too many requests, causing decode
batch size to balloon and TPOT to spike.

CAAS uses an online step-cost model to dynamically constrain admission so that
the predicted step duration stays within the user-specified TPOT SLO, while
still maximising goodput (fraction of requests meeting both TTFT and TPOT SLOs).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Per-request metrics
# ---------------------------------------------------------------------------

@dataclass
class PerRequestMetrics:
    """Latency tracking for a single request."""
    arrival_time: float
    input_len: int
    first_token_time: float = 0.0   # wall-clock time when prefill completes
    ttft: float = 0.0               # = first_token_time - arrival_time
    tokens_generated: int = 0
    tpot_sum: float = 0.0           # sum of inter-token intervals
    last_decode_time: float = 0.0   # timestamp of last decode step

    @property
    def avg_tpot(self) -> float:
        if self.tokens_generated <= 0:
            return 0.0
        return self.tpot_sum / self.tokens_generated

    @property
    def in_decode(self) -> bool:
        return self.first_token_time > 0.0


# ---------------------------------------------------------------------------
# Request lifecycle tracker
# ---------------------------------------------------------------------------

class RequestTracker:
    """Tracks per-request latency metrics and computes aggregate SLO stats."""

    def __init__(self, history_window: int = 1000) -> None:
        self._active: dict[str, PerRequestMetrics] = {}
        # Rolling window of (ttft, avg_tpot) for completed requests
        self._completed: deque[tuple[float, float]] = deque(
            maxlen=history_window)

    # -- lifecycle hooks --

    def register(self, req_id: str, arrival_time: float,
                 input_len: int) -> None:
        self._active[req_id] = PerRequestMetrics(
            arrival_time=arrival_time, input_len=input_len)

    def on_prefill_complete(self, req_id: str, now: float) -> None:
        m = self._active.get(req_id)
        if m is None:
            return
        m.first_token_time = now
        m.ttft = now - m.arrival_time
        m.last_decode_time = now

    def on_decode_step(self, req_id: str, now: float) -> None:
        m = self._active.get(req_id)
        if m is None or not m.in_decode:
            return
        interval = now - m.last_decode_time
        m.tpot_sum += interval
        m.tokens_generated += 1
        m.last_decode_time = now

    def on_finish(self, req_id: str) -> None:
        m = self._active.pop(req_id, None)
        if m is not None and m.in_decode:
            self._completed.append((m.ttft, m.avg_tpot))

    # -- aggregate queries --

    def get_worst_active_tpot(self) -> float:
        """Max avg TPOT among active decode requests."""
        worst = 0.0
        for m in self._active.values():
            if m.in_decode:
                worst = max(worst, m.avg_tpot)
        return worst

    def get_worst_waiting_ttft(self, now: float) -> float:
        """Max elapsed wait time for requests not yet in decode."""
        worst = 0.0
        for m in self._active.values():
            if not m.in_decode:
                worst = max(worst, now - m.arrival_time)
        return worst

    def get_goodput(self, ttft_slo: float, tpot_slo: float) -> float:
        """Fraction of recently completed requests meeting both SLOs."""
        if not self._completed:
            return 1.0
        good = sum(
            1 for ttft, tpot in self._completed
            if ttft <= ttft_slo and tpot <= tpot_slo
        )
        return good / len(self._completed)


# ---------------------------------------------------------------------------
# Online step-cost model
# ---------------------------------------------------------------------------

class StepCostModel:
    """
    Online linear model: step_time ≈ w·x where
        x = [N_decode, prefill_tokens, total_kv_length, 1]

    Updated with Exponentially-Weighted Recursive Least Squares (EW-RLS)
    using forgetting factor λ so the model adapts to workload changes.
    """

    def __init__(self, forgetting_factor: float = 0.95,
                 warmup_steps: int = 50) -> None:
        self.lam = forgetting_factor
        self.warmup_steps = warmup_steps
        self._n_features = 4
        self._w = np.zeros(self._n_features, dtype=np.float64)
        # Large initial P → high uncertainty, fast early adaptation
        self._P = np.eye(self._n_features, dtype=np.float64) * 1e4
        self._step_count = 0

    @property
    def is_warmed_up(self) -> bool:
        return self._step_count >= self.warmup_steps

    def _features(self, n_decode: int, prefill_toks: int,
                  total_kv: int) -> np.ndarray:
        return np.array(
            [float(n_decode), float(prefill_toks), float(total_kv), 1.0],
            dtype=np.float64,
        )

    def predict(self, n_decode: int, prefill_toks: int,
                total_kv: int) -> float:
        """Predict step duration in seconds."""
        x = self._features(n_decode, prefill_toks, total_kv)
        return float(max(0.0, x @ self._w))

    def update(self, n_decode: int, prefill_toks: int, total_kv: int,
               actual_time: float) -> None:
        """Update model with observed step duration."""
        x = self._features(n_decode, prefill_toks, total_kv)
        Px = self._P @ x
        denom = self.lam + float(x @ Px)
        if abs(denom) < 1e-12:
            return
        K = Px / denom
        error = actual_time - float(x @ self._w)
        self._w += K * error
        self._P = (self._P - np.outer(K, Px)) / self.lam
        self._step_count += 1


# ---------------------------------------------------------------------------
# Admission constraints returned to the scheduler
# ---------------------------------------------------------------------------

@dataclass
class AdmissionConstraints:
    """Constraints produced by CAAS for one scheduling step."""
    max_decode_batch: int    # upper bound on total running requests
    max_prefill_tokens: int  # upper bound on prefill tokens this step
    admit_new: bool          # whether to admit any new requests at all


# ---------------------------------------------------------------------------
# Admission controller
# ---------------------------------------------------------------------------

class AdmissionController:
    """
    Core CAAS component.  Called once per scheduling step to produce
    AdmissionConstraints that the scheduler enforces in its waiting loop.
    """

    def __init__(
        self,
        ttft_slo: float,
        tpot_slo: float,
        base_max_running: int,
        base_token_budget: int,
        cooldown_steps: int = 5,
        warmup_steps: int = 50,
        forgetting_factor: float = 0.95,
    ) -> None:
        self.ttft_slo = ttft_slo
        self.tpot_slo = tpot_slo
        self._base_max_running = base_max_running
        self._base_token_budget = base_token_budget
        self._cooldown_steps = cooldown_steps

        self.cost_model = StepCostModel(
            forgetting_factor=forgetting_factor,
            warmup_steps=warmup_steps,
        )
        self.tracker = RequestTracker()

        self._compression_cooldown: int = 0

    def on_compression_event(self, num_compressed: int) -> None:
        """Signal that KV compression just freed blocks for `num_compressed`
        requests.  Triggers an admission cooldown."""
        if num_compressed > 0:
            self._compression_cooldown = self._cooldown_steps
            logger.debug(
                "CAAS: compression event (%d reqs) → cooldown %d steps",
                num_compressed, self._cooldown_steps,
            )

    def get_constraints(
        self,
        running: list["Request"],
        waiting: list["Request"],
    ) -> AdmissionConstraints:
        """Compute admission constraints for the upcoming scheduling step."""
        now = time.monotonic()

        n_decode = sum(1 for r in running if not r.is_prefill_chunk)
        total_kv = sum(r.num_computed_tokens for r in running)
        avg_kv = total_kv / n_decode if n_decode > 0 else 0.0

        # During warmup: permissive defaults so the cost model can gather data
        if not self.cost_model.is_warmed_up:
            return AdmissionConstraints(
                max_decode_batch=self._base_max_running,
                max_prefill_tokens=self._base_token_budget,
                admit_new=True,
            )

        # 1) Decode batch upper bound
        #    Binary search: largest N s.t. predict(N, 0, N*avg_kv) <= tpot_slo
        #    Rationale: in a pure-decode step each request gets 1 token,
        #    so TPOT ≈ step_time.
        max_decode_batch = self._binary_search_max(
            lo=max(n_decode, 1),
            hi=self._base_max_running,
            predict_fn=lambda N: self.cost_model.predict(
                N, 0, int(N * avg_kv)),
            threshold=self.tpot_slo,
        )

        # 2) Prefill token budget
        #    Binary search: largest P s.t. predict(n_decode, P, total_kv) <= tpot_slo
        #    Rationale: prefill tokens in mixed steps inflate step_time for
        #    all concurrent decode requests.
        max_prefill_tokens = self._binary_search_max(
            lo=0,
            hi=self._base_token_budget,
            predict_fn=lambda P: self.cost_model.predict(
                n_decode, P, total_kv),
            threshold=self.tpot_slo,
        )
        max_prefill_tokens = max(max_prefill_tokens, 128)  # ensure progress

        # 3) Compression cooldown: suppress admission for N steps after
        #    compression events to avoid instant batch bloat
        admit_new = True
        if self._compression_cooldown > 0:
            self._compression_cooldown -= 1
            admit_new = False
            max_decode_batch = min(max_decode_batch, len(running))
            logger.debug(
                "CAAS: cooldown active (%d steps left), max_batch=%d",
                self._compression_cooldown + 1, max_decode_batch,
            )

        # 4) TPOT emergency brake: observed TPOT already exceeds SLO
        worst_tpot = self.tracker.get_worst_active_tpot()
        if worst_tpot > self.tpot_slo * 1.2:
            admit_new = False
            max_decode_batch = min(max_decode_batch, n_decode)
            max_prefill_tokens = 128
            logger.debug(
                "CAAS: TPOT emergency brake (worst=%.3fs > SLO*1.2=%.3fs)",
                worst_tpot, self.tpot_slo * 1.2,
            )

        # 5) TTFT anti-starvation: force at least 1 admit if a waiting
        #    request has already waited 80% of its TTFT SLO budget
        worst_wait = self.tracker.get_worst_waiting_ttft(now)
        if worst_wait > self.ttft_slo * 0.8:
            admit_new = True
            max_prefill_tokens = max(max_prefill_tokens, 256)
            logger.debug(
                "CAAS: anti-starvation (worst_wait=%.1fs > SLO*0.8=%.1fs)",
                worst_wait, self.ttft_slo * 0.8,
            )

        return AdmissionConstraints(
            max_decode_batch=max_decode_batch,
            max_prefill_tokens=max_prefill_tokens,
            admit_new=admit_new,
        )

    @staticmethod
    def _binary_search_max(
        lo: int,
        hi: int,
        predict_fn,
        threshold: float,
        n_iters: int = 20,
    ) -> int:
        """Return the largest integer in [lo, hi] where predict_fn(x) <=
        threshold, or lo if even lo violates the threshold."""
        if predict_fn(lo) > threshold:
            return lo
        best = lo
        for _ in range(n_iters):
            if lo > hi:
                break
            mid = (lo + hi) // 2
            if predict_fn(mid) <= threshold:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return best
