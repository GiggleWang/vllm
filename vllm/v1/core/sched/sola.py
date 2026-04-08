# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SOLA: State-Aware Scheduling for LLM Serving (MLSys 2025).

Implements the SOLA scheduling strategy from:
  "SOLA: Optimizing SLO Attainment for Large Language Model Serving
   with State-Aware Scheduling" (Hong et al., MLSys 2025)

Core idea: at each iteration, dynamically decide whether to prioritize
prefill or decode based on real-time TTFT/TPOT state, and control
workload size (ki for prefill token budget, ni for decode batch limit)
via constrained optimization.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Per-request timing metrics
# ---------------------------------------------------------------------------
@dataclass
class RequestMetrics:
    """Tracks real-time latency state for a single request."""

    arrival_time: float
    input_len: int
    predicted_output_len: int

    # set when prefill completes (first token generated)
    first_token_time: float | None = None
    ttft: float = 0.0

    # decode phase tracking
    tokens_generated: int = 0
    tpot_cumulative: float = 0.0
    _last_decode_time: float | None = None

    @property
    def tpot(self) -> float:
        """Average time-per-output-token so far."""
        if self.tokens_generated <= 0:
            return 0.0
        return self.tpot_cumulative / self.tokens_generated


# ---------------------------------------------------------------------------
# Polynomial cost model with EMA scaling calibration
# ---------------------------------------------------------------------------
class CostModel:
    """Predicts prefill and decode iteration latency.

    Prefill: C_p = a0 * sum(l_cached * l_extend) + b0 * sum(l_extend^2)
                  + c0 * sum(l_extend) + d0
    Decode:  C_d = a1 * N + b1 * sum(l_cached) + c1

    A scaling ratio gamma is maintained via EMA to calibrate predictions
    to actual hardware performance (Eq. 5 in paper).
    """

    def __init__(self, alpha: float = 0.1):
        # Prefill coefficients (quadratic in sequence length)
        self.a0 = 1e-7
        self.b0 = 1e-7
        self.c0 = 1e-5
        self.d0 = 1e-3
        # Decode coefficients (linear in batch size and seq len)
        self.a1 = 1e-4
        self.b1 = 1e-7
        self.c1 = 1e-3
        # EMA scaling
        self.alpha = alpha
        self.gamma = 1.0

    def estimate_prefill_cost_from_lens(
        self, cached_lens: list[int], extend_lens: list[int]
    ) -> float:
        """Estimate prefill iteration cost given lists of cached and
        extend lengths."""
        sum_cross = sum(c * e for c, e in zip(cached_lens, extend_lens))
        sum_sq = sum(e * e for e in extend_lens)
        sum_lin = sum(extend_lens)
        raw = (self.a0 * sum_cross + self.b0 * sum_sq
               + self.c0 * sum_lin + self.d0)
        return self.gamma * raw

    def estimate_prefill_cost_single(self, input_len: int) -> float:
        """Estimate prefill cost for a single new request
        (no cached prefix)."""
        raw = self.b0 * input_len * input_len + self.c0 * input_len + self.d0
        return self.gamma * raw

    def estimate_decode_cost(
        self, n_reqs: int, total_seq_len: int
    ) -> float:
        """Estimate decode iteration cost."""
        raw = self.a1 * n_reqs + self.b1 * total_seq_len + self.c1
        return self.gamma * raw

    def update_scaling(
        self, actual_time: float, predicted_time: float
    ) -> None:
        """Update the EMA scaling ratio after observing actual iteration
        latency (Eq. 5)."""
        if predicted_time <= 0:
            return
        ratio = actual_time / predicted_time
        self.gamma = self.alpha * ratio + (1 - self.alpha) * self.gamma


# ---------------------------------------------------------------------------
# State monitor — tracks request-level and system-level state
# ---------------------------------------------------------------------------
class StateMonitor:
    """Maintains real-time request metrics and system-level latency
    statistics."""

    def __init__(self):
        self.metrics: dict[str, RequestMetrics] = {}
        self.p_ttft: float = 0.0  # max_r(t_TTFT_r) / T_TTFT
        self.p_tpot: float = 0.0  # max_r(t_TPOT_r) / T_TPOT
        self.output_length_history: list[int] = []
        self._median_output_len: int = 128  # fallback

    def register_request(
        self,
        request_id: str,
        arrival_time: float,
        input_len: int,
        predicted_output_len: int,
    ) -> None:
        self.metrics[request_id] = RequestMetrics(
            arrival_time=arrival_time,
            input_len=input_len,
            predicted_output_len=predicted_output_len,
        )

    def on_prefill_complete(self, request_id: str, now: float) -> None:
        m = self.metrics.get(request_id)
        if m is None:
            return
        m.first_token_time = now
        m.ttft = now - m.arrival_time
        m._last_decode_time = now

    def on_decode_step(self, request_id: str, now: float) -> None:
        m = self.metrics.get(request_id)
        if m is None or m._last_decode_time is None:
            return
        m.tokens_generated += 1
        m.tpot_cumulative += now - m._last_decode_time
        m._last_decode_time = now

    def on_request_finish(self, request_id: str) -> None:
        m = self.metrics.pop(request_id, None)
        if m is not None and m.tokens_generated > 0:
            self.output_length_history.append(m.tokens_generated)
            # keep history bounded
            if len(self.output_length_history) > 1000:
                self.output_length_history = self.output_length_history[-500:]
            self._update_median()

    def update_system_state(
        self, slo_ttft: float, slo_tpot: float
    ) -> None:
        """Recompute system-level latency ratios p_TTFT and p_TPOT.

        For p_TPOT we use a stale-aware projection: if a request has
        finished prefill but has not received a decode step recently,
        the idle time is included so that decode starvation is visible
        to the phase decision.
        """
        now = time.monotonic()
        max_ttft = 0.0
        max_tpot = 0.0
        for m in self.metrics.values():
            # For requests still in prefill, use waiting time
            if m.first_token_time is None:
                ttft = now - m.arrival_time
            else:
                ttft = m.ttft
            max_ttft = max(max_ttft, ttft)
            # Stale-aware TPOT projection
            if m._last_decode_time is not None:
                stale = max(0.0, now - m._last_decode_time)
                effective_tokens = max(m.tokens_generated, 1)
                projected_tpot = (
                    (m.tpot_cumulative + stale) / effective_tokens
                )
                max_tpot = max(max_tpot, projected_tpot)
        self.p_ttft = max_ttft / slo_ttft if slo_ttft > 0 else 0.0
        self.p_tpot = max_tpot / slo_tpot if slo_tpot > 0 else 0.0

    def get_ttft(self, request_id: str) -> float:
        """Get current TTFT for a request (waiting time if still in
        prefill)."""
        m = self.metrics.get(request_id)
        if m is None:
            return 0.0
        if m.first_token_time is not None:
            return m.ttft
        return time.monotonic() - m.arrival_time

    def get_tpot(self, request_id: str) -> float:
        """Return actual measured average TPOT for a request."""
        m = self.metrics.get(request_id)
        if m is None:
            return 0.0
        return m.tpot

    def get_tokens_generated(self, request_id: str) -> int:
        m = self.metrics.get(request_id)
        return m.tokens_generated if m is not None else 0

    def predict_remaining_length(self, request_id: str) -> int:
        m = self.metrics.get(request_id)
        if m is None:
            return self._median_output_len
        remaining = m.predicted_output_len - m.tokens_generated
        if remaining <= 0:
            return max(1, self._median_output_len - m.tokens_generated)
        return remaining

    def _update_median(self) -> None:
        if self.output_length_history:
            s = sorted(self.output_length_history)
            self._median_output_len = s[len(s) // 2]


# ---------------------------------------------------------------------------
# Strategy generator — decides phase priority and workload limits
# ---------------------------------------------------------------------------
@dataclass
class SOLADecision:
    """Output of the strategy generator for one iteration."""

    phase_priority: Literal["prefill_first", "decode_first"]
    max_new_tokens: int | None = None  # ki: prefill token budget
    max_prefill_reqs: int | None = None  # ni: max prefill reqs
    sorted_waiting: list[Request] | None = None
    last_predicted_cost: float = 0.0


class StrategyGenerator:
    """Core SOLA strategy generator.

    At each iteration, decides:
    1. Phase priority (prefill-first or decode-first)
    2. Workload limits (ki, ni) via constrained optimization
    3. Request ordering within each phase (hierarchical prioritization)
    """

    def __init__(
        self,
        slo_ttft: float,
        slo_tpot: float,
        cost_model: CostModel,
        state_monitor: StateMonitor,
        percentile: float = 0.95,
        window_size: int = 10,
    ):
        self.slo_ttft = slo_ttft
        self.slo_tpot = slo_tpot
        self.cost_model = cost_model
        self.state_monitor = state_monitor
        self.percentile = percentile
        self._last_predicted_cost: float = 0.0
        # Sliding window for phase decision (paper Sec 4.4.2)
        self._phase_window: deque[tuple[float, float]] = deque(
            maxlen=window_size
        )

    @property
    def last_predicted_cost(self) -> float:
        return self._last_predicted_cost

    def decide(
        self,
        waiting: list[Request],
        running: list[Request],
    ) -> SOLADecision:
        """Generate scheduling strategy for the current iteration."""
        sm = self.state_monitor
        cm = self.cost_model

        sm.update_system_state(self.slo_ttft, self.slo_tpot)

        # Sliding window smoothing (paper Sec 4.4.2)
        self._phase_window.append((sm.p_ttft, sm.p_tpot))
        avg_p_ttft = sum(p[0] for p in self._phase_window) / len(
            self._phase_window
        )
        avg_p_tpot = sum(p[1] for p in self._phase_window) / len(
            self._phase_window
        )

        n_decode = len(running)
        total_decode_seq_len = (
            sum(r.num_computed_tokens for r in running) if running else 0
        )

        # Default: no limits
        ki: int | None = None
        ni: int | None = None

        sorted_waiting = self._sort_prefill_by_ttft(waiting, cm)

        if avg_p_tpot >= avg_p_ttft:
            # TPOT is less fulfilled → optimize TPOT, constrain TTFT
            phase: Literal["prefill_first", "decode_first"] = "decode_first"
            ni = self._compute_ni(
                sorted_waiting, running, cm, n_decode, total_decode_seq_len
            )
        else:
            # TTFT is less fulfilled → optimize TTFT, constrain TPOT
            phase = "prefill_first"
            ki = self._compute_ki(running, cm)

        return SOLADecision(
            phase_priority=phase,
            max_new_tokens=ki,
            max_prefill_reqs=ni,
            sorted_waiting=sorted_waiting,
            last_predicted_cost=self._last_predicted_cost,
        )

    # -- Hierarchical prioritization (Sec 4.4.2) --

    def _sort_prefill_by_ttft(
        self, waiting: list[Request], cm: CostModel
    ) -> list[Request]:
        """Sort prefill requests descending by predicted total TTFT.

        Chunked requests (already partially prefilled) stay at front.
        Others sorted by (current_ttft + predicted_prefill_cost) desc.
        """
        chunked = [
            r for r in waiting
            if r.num_computed_tokens > 0 and r.is_prefill_chunk
        ]
        non_chunked = [
            r for r in waiting
            if not (r.num_computed_tokens > 0 and r.is_prefill_chunk)
        ]

        sm = self.state_monitor

        def prefill_priority(r: Request) -> float:
            current_ttft = sm.get_ttft(r.request_id)
            predicted_prefill = cm.estimate_prefill_cost_single(
                r.num_prompt_tokens
            )
            return current_ttft + predicted_prefill

        non_chunked.sort(key=prefill_priority, reverse=True)
        return chunked + non_chunked

    def _sort_decode_by_tpot(
        self,
        running: list[Request],
        cm: CostModel,
        n_decode: int,
        total_seq_len: int,
    ) -> list[Request]:
        """Sort decode requests descending by predicted TPOT.

        predicted_TPOT = (t_TPOT * l_out + C_d * l_left) / (l_out + l_left)
        """
        sm = self.state_monitor
        decode_cost = cm.estimate_decode_cost(n_decode, total_seq_len)

        def decode_priority(r: Request) -> float:
            t_tpot = sm.get_tpot(r.request_id)
            l_out = max(sm.get_tokens_generated(r.request_id), 1)
            l_left = max(sm.predict_remaining_length(r.request_id), 1)
            return (t_tpot * l_out + decode_cost * l_left) / (l_out + l_left)

        return sorted(running, key=decode_priority, reverse=True)

    # -- Constrained workload computation (Sec 4.4.3) --

    def _compute_ki(
        self, running: list[Request], cm: CostModel
    ) -> int | None:
        """Compute max prefill tokens ki satisfying TPOT constraint (Eq 1).

        Constraint: for each running decode req r,
          t_TPOT_r + C_p(Q_run) / l_out_r <= T_TPOT

        Rearranging: C_p(Q_run) <= min_r((T_TPOT - t_TPOT_r) * l_out_r)

        Binary search finds largest ki (token budget, multiples of 128)
        satisfying this constraint.
        """
        sm = self.state_monitor
        if not running:
            return None  # no constraint

        # Compute the tightest bound across all decode requests
        min_budget = float("inf")
        for r in running:
            t_tpot = sm.get_tpot(r.request_id)
            l_out = max(sm.get_tokens_generated(r.request_id), 1)
            slack = (self.slo_tpot - t_tpot) * l_out
            min_budget = min(min_budget, slack)

        if min_budget <= 0:
            # TPOT already violated — use loosened constraint (percentile)
            slacks = []
            for r in running:
                t_tpot = sm.get_tpot(r.request_id)
                l_out = max(sm.get_tokens_generated(r.request_id), 1)
                slacks.append((self.slo_tpot - t_tpot) * l_out)
            slacks.sort()
            idx = max(0, int(len(slacks) * self.percentile) - 1)
            min_budget = slacks[idx]
            if min_budget <= 0:
                return 128  # minimum budget

        # Binary search: find largest ki (multiple of 128) where
        # C_p([0],[ki]) <= min_budget
        tile = 128
        lo, hi = tile, 65536
        best = tile
        while lo <= hi:
            mid = ((lo + hi) // 2 // tile) * tile
            if mid == 0:
                mid = tile
            cost = cm.estimate_prefill_cost_from_lens([0], [mid])
            if cost <= min_budget:
                best = mid
                lo = mid + tile
            else:
                hi = mid - tile
        return max(best, tile)

    def _compute_ni(
        self,
        waiting: list[Request],
        running: list[Request],
        cm: CostModel,
        n_decode: int,
        total_seq_len: int,
    ) -> int | None:
        """Compute max prefill request count ni satisfying TTFT constraint
        (Eq 2).

        Constraint: for each waiting prefill req r,
          t_TTFT_r + C_d(ni, total_seq_len) + C_p(r) <= T_TTFT

        We find the largest ni such that the decode cost C_d(ni, ...)
        keeps TTFT within SLO for the most urgent waiting request.
        """
        sm = self.state_monitor
        if not waiting or not running:
            return None  # no constraint needed

        # Find the tightest TTFT constraint among waiting prefill requests
        min_slack = float("inf")
        for r in waiting:
            # Skip chunked requests (already partially prefilled)
            if r.num_computed_tokens > 0 and r.is_prefill_chunk:
                continue
            t_ttft = sm.get_ttft(r.request_id)
            prefill_cost_r = cm.estimate_prefill_cost_single(
                r.num_prompt_tokens
            )
            slack = self.slo_ttft - t_ttft - prefill_cost_r
            min_slack = min(min_slack, slack)

        if min_slack == float("inf"):
            return None  # no non-chunked prefill requests

        if min_slack <= 0:
            # TTFT already violated — use loosened constraint
            slacks = []
            for r in waiting:
                if r.num_computed_tokens > 0 and r.is_prefill_chunk:
                    continue
                t_ttft = sm.get_ttft(r.request_id)
                prefill_cost_r = cm.estimate_prefill_cost_single(
                    r.num_prompt_tokens
                )
                slacks.append(self.slo_ttft - t_ttft - prefill_cost_r)
            if not slacks:
                return None
            slacks.sort()
            idx = max(0, int(len(slacks) * self.percentile) - 1)
            min_slack = slacks[idx]
            if min_slack <= 0:
                return 1  # minimum: allow 1 prefill request

        # Find largest ni where C_d(ni, total_seq_len) <= min_slack
        # C_d = a1 * ni + b1 * total_seq_len + c1  (linear in ni)
        if cm.a1 > 0:
            max_ni = int(
                (min_slack - cm.b1 * total_seq_len * cm.gamma
                 - cm.c1 * cm.gamma)
                / (cm.a1 * cm.gamma)
            )
        else:
            max_ni = n_decode
        return max(1, min(max_ni, n_decode))
