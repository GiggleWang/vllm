# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SOLA state-aware scheduling components.

Tests the core SOLA classes (RequestMetrics, CostModel, StateMonitor,
StrategyGenerator) in isolation — no GPU required.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from vllm.v1.core.sched.sola import (
    CostModel,
    RequestMetrics,
    SOLADecision,
    StateMonitor,
    StrategyGenerator,
)


# ---------------------------------------------------------------------------
# Helpers — lightweight fakes for vllm Request
# ---------------------------------------------------------------------------
class FakeRequest:
    """Minimal mock of vllm.v1.request.Request for SOLA tests."""

    def __init__(
        self,
        request_id: str,
        num_prompt_tokens: int = 512,
        num_computed_tokens: int = 0,
        num_output_tokens: int = 0,
        max_tokens: int = 128,
        is_prefill_chunk: bool = True,
        arrival_time: float = 0.0,
    ):
        self.request_id = request_id
        self.num_prompt_tokens = num_prompt_tokens
        self.num_computed_tokens = num_computed_tokens
        self.num_output_tokens = num_output_tokens
        self.max_tokens = max_tokens
        self.is_prefill_chunk = is_prefill_chunk
        self.arrival_time = arrival_time


# ---------------------------------------------------------------------------
# RequestMetrics
# ---------------------------------------------------------------------------
class TestRequestMetrics:
    def test_tpot_zero_tokens(self):
        m = RequestMetrics(
            arrival_time=0.0, input_len=100, predicted_output_len=50
        )
        assert m.tpot == 0.0

    def test_tpot_positive(self):
        m = RequestMetrics(
            arrival_time=0.0, input_len=100, predicted_output_len=50
        )
        m.tokens_generated = 10
        m.tpot_cumulative = 1.0
        assert m.tpot == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# CostModel
# ---------------------------------------------------------------------------
class TestCostModel:
    def test_prefill_cost_increases_with_length(self):
        cm = CostModel()
        c1 = cm.estimate_prefill_cost_from_lens([0], [100])
        c2 = cm.estimate_prefill_cost_from_lens([0], [1000])
        assert c2 > c1

    def test_decode_cost_increases_with_batch(self):
        cm = CostModel()
        c1 = cm.estimate_decode_cost(1, 100)
        c2 = cm.estimate_decode_cost(10, 100)
        assert c2 > c1

    def test_scaling_ratio_update(self):
        cm = CostModel(alpha=0.5)
        cm.gamma = 1.0
        # actual is 2x predicted -> gamma should move toward 2.0
        cm.update_scaling(actual_time=2.0, predicted_time=1.0)
        assert cm.gamma == pytest.approx(1.5)  # 0.5 * 2.0 + 0.5 * 1.0

    def test_scaling_ratio_skip_zero_predicted(self):
        cm = CostModel()
        old_gamma = cm.gamma
        cm.update_scaling(actual_time=1.0, predicted_time=0.0)
        assert cm.gamma == old_gamma

    def test_single_prefill_cost(self):
        cm = CostModel()
        c = cm.estimate_prefill_cost_single(256)
        assert c > 0


# ---------------------------------------------------------------------------
# StateMonitor
# ---------------------------------------------------------------------------
class TestStateMonitor:
    def test_register_and_get(self):
        sm = StateMonitor()
        sm.register_request(
            request_id="r1",
            arrival_time=100.0,
            input_len=512,
            predicted_output_len=128,
        )
        assert "r1" in sm.metrics
        assert sm.metrics["r1"].input_len == 512

    def test_prefill_complete(self):
        sm = StateMonitor()
        sm.register_request(
            request_id="r1",
            arrival_time=100.0,
            input_len=512,
            predicted_output_len=128,
        )
        sm.on_prefill_complete(request_id="r1", now=100.5)
        assert sm.metrics["r1"].ttft == pytest.approx(0.5)
        assert sm.metrics["r1"].first_token_time == 100.5

    def test_decode_step(self):
        sm = StateMonitor()
        sm.register_request(
            request_id="r1",
            arrival_time=100.0,
            input_len=512,
            predicted_output_len=128,
        )
        sm.on_prefill_complete(request_id="r1", now=100.5)
        sm.on_decode_step(request_id="r1", now=100.6)
        sm.on_decode_step(request_id="r1", now=100.8)
        assert sm.metrics["r1"].tokens_generated == 2
        assert sm.metrics["r1"].tpot == pytest.approx(0.15)  # 0.3 / 2

    def test_request_finish(self):
        sm = StateMonitor()
        sm.register_request(
            request_id="r1",
            arrival_time=100.0,
            input_len=512,
            predicted_output_len=128,
        )
        sm.on_prefill_complete(request_id="r1", now=100.5)
        sm.on_decode_step(request_id="r1", now=100.6)
        sm.on_request_finish(request_id="r1")
        assert "r1" not in sm.metrics
        assert len(sm.output_length_history) == 1

    def test_update_system_state(self):
        sm = StateMonitor()
        now = time.monotonic()
        sm.register_request(
            request_id="r1",
            arrival_time=now - 2.0,
            input_len=100,
            predicted_output_len=50,
        )
        sm.on_prefill_complete(request_id="r1", now=now - 1.5)
        sm.on_decode_step(request_id="r1", now=now - 1.0)
        sm.on_decode_step(request_id="r1", now=now)
        sm.update_system_state(slo_ttft=1.0, slo_tpot=0.5)
        # p_ttft = ttft / slo_ttft = 0.5 / 1.0 = 0.5
        assert sm.p_ttft == pytest.approx(0.5, abs=0.05)
        # tpot_cumulative = 0.5 + 1.0 = 1.5, tokens = 2, stale ~ 0
        # projected_tpot ~ 1.5 / 2 = 0.75, p_tpot = 0.75 / 0.5 = 1.5
        assert sm.p_tpot == pytest.approx(1.5, abs=0.15)

    def test_system_tpot_detects_decode_starvation(self):
        """System-level p_TPOT should detect decode starvation via stale
        projection."""
        sm = StateMonitor()
        now = time.monotonic()
        sm.register_request(
            request_id="r1",
            arrival_time=now - 5.0,
            input_len=100,
            predicted_output_len=50,
        )
        sm.on_prefill_complete(request_id="r1", now=now - 4.5)
        sm.on_decode_step(
            request_id="r1", now=now - 4.0
        )  # one decode step, then starved
        # ~4 seconds of starvation
        sm.update_system_state(slo_ttft=10.0, slo_tpot=1.0)
        assert sm.p_tpot > 3.0

    def test_system_tpot_detects_wait_before_first_decode(self):
        """System-level p_TPOT should grow after prefill, even before
        first decode."""
        sm = StateMonitor()
        now = time.monotonic()
        sm.register_request(
            request_id="r1",
            arrival_time=now - 2.0,
            input_len=100,
            predicted_output_len=50,
        )
        sm.on_prefill_complete(request_id="r1", now=now - 0.6)
        sm.update_system_state(slo_ttft=10.0, slo_tpot=0.1)
        assert sm.p_tpot > 5.0

    def test_get_tpot_returns_actual_measurement(self):
        """get_tpot() should return actual, not projected."""
        sm = StateMonitor()
        now = time.monotonic()
        sm.register_request(
            request_id="r1",
            arrival_time=now - 5.0,
            input_len=100,
            predicted_output_len=50,
        )
        sm.on_prefill_complete(request_id="r1", now=now - 4.5)
        sm.on_decode_step(
            request_id="r1", now=now - 4.0
        )  # tpot = 0.5 / 1 = 0.5
        assert sm.get_tpot("r1") == pytest.approx(0.5)
        # Before any decode step, get_tpot is 0
        sm.register_request(
            request_id="r2",
            arrival_time=now - 2.0,
            input_len=100,
            predicted_output_len=50,
        )
        sm.on_prefill_complete(request_id="r2", now=now - 1.0)
        assert sm.get_tpot("r2") == pytest.approx(0.0)

    def test_predict_remaining_length(self):
        sm = StateMonitor()
        sm.register_request(
            request_id="r1",
            arrival_time=0.0,
            input_len=100,
            predicted_output_len=50,
        )
        assert sm.predict_remaining_length("r1") == 50
        sm.metrics["r1"].tokens_generated = 10
        assert sm.predict_remaining_length("r1") == 40

    def test_unknown_request_id(self):
        sm = StateMonitor()
        assert sm.get_ttft("unknown") == 0.0
        assert sm.get_tpot("unknown") == 0.0
        assert sm.get_tokens_generated("unknown") == 0


# ---------------------------------------------------------------------------
# StrategyGenerator
# ---------------------------------------------------------------------------
class TestStrategyGenerator:
    def _make_generator(
        self, slo_ttft=5.0, slo_tpot=0.1
    ) -> StrategyGenerator:
        return StrategyGenerator(
            slo_ttft=slo_ttft,
            slo_tpot=slo_tpot,
            cost_model=CostModel(),
            state_monitor=StateMonitor(),
        )

    def test_empty_queues(self):
        sg = self._make_generator()
        decision = sg.decide([], [])
        assert decision.phase_priority in ("prefill_first", "decode_first")
        assert decision.sorted_waiting == []

    def test_prefill_only(self):
        sg = self._make_generator()
        now = time.monotonic()
        waiting = [
            FakeRequest("r1", num_prompt_tokens=512, arrival_time=now - 0.1),
            FakeRequest("r2", num_prompt_tokens=256, arrival_time=now - 0.2),
        ]
        sg.state_monitor.register_request("r1", now - 0.1, 512, 128)
        sg.state_monitor.register_request("r2", now - 0.2, 256, 128)
        decision = sg.decide(waiting, [])
        assert len(decision.sorted_waiting) == 2

    def test_decode_only(self):
        sg = self._make_generator()
        now = time.monotonic()
        running = [
            FakeRequest(
                "r1",
                num_computed_tokens=100,
                is_prefill_chunk=False,
                arrival_time=now - 1.0,
            ),
            FakeRequest(
                "r2",
                num_computed_tokens=200,
                is_prefill_chunk=False,
                arrival_time=now - 1.0,
            ),
        ]
        for r in running:
            sg.state_monitor.register_request(
                r.request_id, now - 1.0, 100, 50
            )
            sg.state_monitor.on_prefill_complete(r.request_id, now - 0.5)
            sg.state_monitor.on_decode_step(r.request_id, now - 0.1)
        decision = sg.decide([], running)
        # With no waiting requests, sorted_waiting should be empty
        assert decision.sorted_waiting == []

    def test_tpot_pressure_triggers_decode_first(self):
        """When TPOT is more violated, should prioritize decode."""
        sg = self._make_generator(slo_ttft=10.0, slo_tpot=0.001)
        now = time.monotonic()
        running = [
            FakeRequest(
                "r1",
                num_computed_tokens=100,
                is_prefill_chunk=False,
                arrival_time=now - 2.0,
            ),
        ]
        waiting = [
            FakeRequest("r2", num_prompt_tokens=256, arrival_time=now - 0.01),
        ]
        sg.state_monitor.register_request("r1", now - 2.0, 100, 50)
        sg.state_monitor.on_prefill_complete("r1", now - 1.5)
        sg.state_monitor.on_decode_step("r1", now - 0.5)
        sg.state_monitor.register_request("r2", now - 0.01, 256, 128)
        decision = sg.decide(waiting, running)
        assert decision.phase_priority == "decode_first"

    def test_ttft_pressure_triggers_prefill_first(self):
        """When TTFT is more violated, should prioritize prefill."""
        sg = self._make_generator(slo_ttft=0.001, slo_tpot=10.0)
        now = time.monotonic()
        running = [
            FakeRequest(
                "r1",
                num_computed_tokens=100,
                is_prefill_chunk=False,
                arrival_time=now - 0.5,
            ),
        ]
        waiting = [
            FakeRequest("r2", num_prompt_tokens=256, arrival_time=now - 1.0),
        ]
        sg.state_monitor.register_request("r1", now - 0.5, 100, 50)
        sg.state_monitor.on_prefill_complete("r1", now - 0.4)
        sg.state_monitor.on_decode_step("r1", now - 0.01)
        sg.state_monitor.register_request("r2", now - 1.0, 256, 128)
        decision = sg.decide(waiting, running)
        assert decision.phase_priority == "prefill_first"

    def test_tpot_pressure_before_first_decode_triggers_decode_first(self):
        """Requests waiting for first decode should contribute TPOT
        pressure."""
        sg = self._make_generator(slo_ttft=10.0, slo_tpot=0.1)
        now = time.monotonic()
        running = [
            FakeRequest(
                "r1",
                num_computed_tokens=100,
                is_prefill_chunk=False,
                arrival_time=now - 2.0,
            ),
        ]
        waiting = [
            FakeRequest("r2", num_prompt_tokens=256, arrival_time=now - 0.01),
        ]
        sg.state_monitor.register_request("r1", now - 2.0, 100, 50)
        sg.state_monitor.on_prefill_complete("r1", now - 0.6)
        sg.state_monitor.register_request("r2", now - 0.01, 256, 128)
        decision = sg.decide(waiting, running)
        assert decision.phase_priority == "decode_first"

    def test_chunked_req_stays_at_front(self):
        sg = self._make_generator()
        now = time.monotonic()
        # Chunked request: has computed tokens and still in prefill
        chunked = FakeRequest(
            "r1",
            num_prompt_tokens=1024,
            num_computed_tokens=512,
            is_prefill_chunk=True,
            arrival_time=now - 5.0,
        )
        normal = FakeRequest(
            "r2", num_prompt_tokens=256, arrival_time=now - 0.1
        )
        sg.state_monitor.register_request("r1", now - 5.0, 1024, 128)
        sg.state_monitor.register_request("r2", now - 0.1, 256, 128)
        decision = sg.decide([normal, chunked], [])
        # chunked should be first regardless of sort
        assert decision.sorted_waiting[0].request_id == "r1"

    def test_ki_computation(self):
        sg = self._make_generator(slo_tpot=1.0)
        now = time.monotonic()
        running = [
            FakeRequest(
                "r1",
                num_computed_tokens=100,
                is_prefill_chunk=False,
                arrival_time=now - 1.0,
            ),
        ]
        sg.state_monitor.register_request("r1", now - 1.0, 100, 50)
        sg.state_monitor.on_prefill_complete("r1", now - 0.5)
        sg.state_monitor.on_decode_step("r1", now - 0.1)
        ki = sg._compute_ki(running, sg.cost_model)
        assert ki is not None
        assert ki >= 128
        assert ki % 128 == 0

    def test_ki_no_decode_reqs(self):
        sg = self._make_generator()
        ki = sg._compute_ki([], sg.cost_model)
        assert ki is None  # no constraint

    def test_ni_no_pending(self):
        sg = self._make_generator()
        running = [
            FakeRequest(
                "r1",
                num_computed_tokens=100,
                is_prefill_chunk=False,
            ),
        ]
        ni = sg._compute_ni([], running, sg.cost_model, 1, 100)
        assert ni is None

    def test_predicted_cost_deferred_to_scheduler(self):
        """predicted cost is no longer set by decide() — it is computed
        by the scheduler after the actual batch is formed."""
        sg = self._make_generator()
        now = time.monotonic()
        waiting = [
            FakeRequest("r1", num_prompt_tokens=256, arrival_time=now),
        ]
        sg.state_monitor.register_request("r1", now, 256, 128)
        decision = sg.decide(waiting, [])
        assert decision.last_predicted_cost == 0.0
