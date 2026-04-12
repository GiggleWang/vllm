# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Compression-Aware Adaptive Scheduler (CAAS)."""

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

# Import directly from the module — no torch/vllm runtime needed
from vllm.v1.core.sched.caas import (
    AdmissionConstraints,
    AdmissionController,
    PerRequestMetrics,
    RequestTracker,
    StepCostModel,
)


# ---------------------------------------------------------------------------
# PerRequestMetrics
# ---------------------------------------------------------------------------

class TestPerRequestMetrics:
    def test_avg_tpot_no_tokens(self):
        m = PerRequestMetrics(arrival_time=0.0, input_len=100)
        assert m.avg_tpot == 0.0

    def test_avg_tpot_with_tokens(self):
        m = PerRequestMetrics(arrival_time=0.0, input_len=100)
        m.tpot_sum = 0.3
        m.tokens_generated = 3
        assert abs(m.avg_tpot - 0.1) < 1e-9

    def test_in_decode(self):
        m = PerRequestMetrics(arrival_time=0.0, input_len=100)
        assert not m.in_decode
        m.first_token_time = 1.0
        assert m.in_decode


# ---------------------------------------------------------------------------
# RequestTracker
# ---------------------------------------------------------------------------

class TestRequestTracker:
    def _make_tracker(self):
        return RequestTracker(history_window=100)

    def test_register_and_finish(self):
        t = self._make_tracker()
        t.register("r1", arrival_time=0.0, input_len=50)
        t.on_prefill_complete("r1", now=1.0)
        t.on_decode_step("r1", now=1.1)
        t.on_decode_step("r1", now=1.2)
        t.on_finish("r1")
        # After finish, not in active
        assert "r1" not in t._active

    def test_goodput_both_slo_met(self):
        t = self._make_tracker()
        t.register("r1", arrival_time=0.0, input_len=50)
        t.on_prefill_complete("r1", now=1.0)  # ttft=1.0
        t.on_decode_step("r1", now=1.1)
        t.on_decode_step("r1", now=1.2)
        t.on_finish("r1")  # avg_tpot = 0.1
        # SLO: ttft<=2.0, tpot<=0.2 → both met
        assert t.get_goodput(ttft_slo=2.0, tpot_slo=0.2) == 1.0

    def test_goodput_ttft_violated(self):
        t = self._make_tracker()
        t.register("r1", arrival_time=0.0, input_len=50)
        t.on_prefill_complete("r1", now=5.0)  # ttft=5.0
        t.on_decode_step("r1", now=5.1)
        t.on_finish("r1")
        # SLO: ttft<=2.0 → violated
        assert t.get_goodput(ttft_slo=2.0, tpot_slo=1.0) == 0.0

    def test_goodput_empty_history(self):
        t = self._make_tracker()
        assert t.get_goodput(ttft_slo=1.0, tpot_slo=0.1) == 1.0

    def test_worst_active_tpot_no_decode(self):
        t = self._make_tracker()
        t.register("r1", arrival_time=0.0, input_len=50)
        assert t.get_worst_active_tpot() == 0.0

    def test_worst_active_tpot_with_decode(self):
        t = self._make_tracker()
        t.register("r1", arrival_time=0.0, input_len=50)
        t.on_prefill_complete("r1", now=1.0)
        t.on_decode_step("r1", now=1.5)  # interval=0.5
        t.on_decode_step("r1", now=2.0)  # interval=0.5; avg=0.5
        assert abs(t.get_worst_active_tpot() - 0.5) < 1e-6

    def test_worst_waiting_ttft(self):
        t = self._make_tracker()
        t.register("r1", arrival_time=0.0, input_len=50)
        t.register("r2", arrival_time=1.0, input_len=50)
        # Neither has received first token
        worst = t.get_worst_waiting_ttft(now=3.0)
        assert abs(worst - 3.0) < 1e-6  # r1 waited 3s

    def test_unknown_req_is_noop(self):
        """Operations on unknown req_id should not crash."""
        t = self._make_tracker()
        t.on_prefill_complete("nonexistent", now=1.0)
        t.on_decode_step("nonexistent", now=2.0)
        t.on_finish("nonexistent")


# ---------------------------------------------------------------------------
# StepCostModel
# ---------------------------------------------------------------------------

class TestStepCostModel:
    def test_predict_zeros_initially(self):
        m = StepCostModel(warmup_steps=5)
        # Weights are zero → prediction is 0
        assert m.predict(10, 100, 5000) == 0.0

    def test_not_warmed_up_initially(self):
        m = StepCostModel(warmup_steps=5)
        assert not m.is_warmed_up

    def test_warmed_up_after_enough_updates(self):
        m = StepCostModel(warmup_steps=3)
        for _ in range(3):
            m.update(10, 0, 1000, 0.05)
        assert m.is_warmed_up

    def test_update_improves_prediction(self):
        """After many updates the model should converge toward actual values."""
        m = StepCostModel(warmup_steps=5, forgetting_factor=0.99)
        # Ground truth: step_time = 0.001 * n_decode + 0.00001 * total_kv + 0.01
        rng = np.random.default_rng(42)
        for _ in range(200):
            n = int(rng.integers(10, 100))
            kv = int(rng.integers(1000, 50000))
            true_time = 0.001 * n + 0.00001 * kv + 0.01
            noisy_time = true_time + rng.normal(0, 0.001)
            m.update(n, 0, kv, noisy_time)

        # After convergence, prediction should be close
        pred = m.predict(50, 0, 20000)
        expected = 0.001 * 50 + 0.00001 * 20000 + 0.01
        assert abs(pred - expected) < 0.005, f"pred={pred:.4f} expected={expected:.4f}"

    def test_prefill_feature_learned(self):
        """Model should learn prefill token cost independently."""
        m = StepCostModel(warmup_steps=5, forgetting_factor=0.99)
        rng = np.random.default_rng(7)
        for _ in range(200):
            n = 50
            p = int(rng.integers(0, 2000))
            kv = 20000
            true_time = 0.001 * n + 0.0001 * p + 0.00001 * kv + 0.01
            m.update(n, p, kv, true_time + rng.normal(0, 0.002))

        # With 500 prefill tokens vs 0, prediction should differ noticeably
        pred_with = m.predict(50, 500, 20000)
        pred_without = m.predict(50, 0, 20000)
        assert pred_with > pred_without


# ---------------------------------------------------------------------------
# AdmissionController
# ---------------------------------------------------------------------------

def _make_mock_request(req_id: str, is_prefill: bool, computed: int = 1000):
    r = MagicMock()
    r.request_id = req_id
    r.is_prefill_chunk = is_prefill
    r.num_computed_tokens = computed
    return r


class TestAdmissionController:
    def _make_ctrl(self, ttft_slo=10.0, tpot_slo=0.2,
                   base_max=128, base_budget=16384,
                   warmup=0):
        """warmup=0 so constraints are active immediately in tests."""
        return AdmissionController(
            ttft_slo=ttft_slo,
            tpot_slo=tpot_slo,
            base_max_running=base_max,
            base_token_budget=base_budget,
            warmup_steps=warmup,
            forgetting_factor=0.9,
        )

    def test_warmup_returns_permissive(self):
        ctrl = AdmissionController(
            ttft_slo=10.0, tpot_slo=0.2,
            base_max_running=128, base_token_budget=16384,
            warmup_steps=10,
        )
        running = [_make_mock_request(f"r{i}", False) for i in range(5)]
        c = ctrl.get_constraints(running, [])
        assert c.max_decode_batch == 128
        assert c.max_prefill_tokens == 16384
        assert c.admit_new is True

    def test_compression_cooldown_immediate_admit_when_safe(self):
        """Cost model predicts safe right after compression → admit immediately."""
        ctrl = self._make_ctrl(tpot_slo=0.2, warmup=0)
        # Teach cost model: step_time = 0.001 * N_decode (light load)
        # With 10 decode requests, predicted for 11 = 0.011 << 0.2 SLO
        for _ in range(10):
            ctrl.cost_model.update(10, 0, 10000, 0.01)

        running = [_make_mock_request(f"r{i}", False) for i in range(10)]
        ctrl.on_compression_event(3)

        # Cost model says safe → admit_new=True immediately
        c = ctrl.get_constraints(running, [])
        assert c.admit_new is True

    def test_compression_cooldown_held_when_unsafe(self):
        """Cost model predicts unsafe → cooldown blocks admission."""
        ctrl = self._make_ctrl(tpot_slo=0.1, warmup=0)
        # Teach cost model: step_time ≈ 0.002 * N_decode
        # With 60 requests, predicted for 61 = 0.122 > 0.1 SLO
        for _ in range(10):
            ctrl.cost_model.update(60, 0, 60000, 0.12)
            ctrl.cost_model.update(50, 0, 50000, 0.10)

        running = [_make_mock_request(f"r{i}", False) for i in range(60)]
        ctrl.on_compression_event(5)

        # Cost model says unsafe → blocked
        c = ctrl.get_constraints(running, [])
        assert c.admit_new is False

        # Still unsafe on next step → still blocked
        c = ctrl.get_constraints(running, [])
        assert c.admit_new is False

    def test_compression_cooldown_clears_when_running_empty(self):
        """Cooldown must not block admission when running queue is empty."""
        ctrl = self._make_ctrl(tpot_slo=0.1, warmup=0)
        # Teach cost model with heavy load so predict(1,0,0) might exceed SLO
        for _ in range(20):
            ctrl.cost_model.update(60, 0, 60000, 0.12)

        ctrl.on_compression_event(5)

        # All requests finished → running is empty
        c = ctrl.get_constraints([], [])
        # Must allow admission — nothing to cool down from
        assert c.admit_new is True

    def test_tpot_emergency_brake(self):
        ctrl = self._make_ctrl(tpot_slo=0.2, warmup=0)
        for _ in range(5):
            ctrl.cost_model.update(50, 0, 50000, 0.05)

        # Simulate a request with very bad TPOT (> 0.2 * 1.2 = 0.24)
        ctrl.tracker.register("r1", arrival_time=0.0, input_len=100)
        ctrl.tracker.on_prefill_complete("r1", now=1.0)
        # 5 decode steps with 0.1s interval → avg_tpot=0.1 (ok)
        for i in range(5):
            ctrl.tracker.on_decode_step("r1", now=1.0 + (i + 1) * 0.3)
        # Now worst tpot is 0.3 > 0.24

        running = [_make_mock_request(f"r{i}", False) for i in range(50)]
        c = ctrl.get_constraints(running, [])
        assert c.admit_new is False
        assert c.max_prefill_tokens == 128

    def test_antistarvation_overrides_cooldown(self):
        ctrl = self._make_ctrl(ttft_slo=5.0, tpot_slo=0.2, warmup=0)
        for _ in range(5):
            ctrl.cost_model.update(50, 0, 50000, 0.05)

        # Trigger cooldown
        ctrl.on_compression_event(3)

        # Register a request that has been waiting 4.5s (> 5.0 * 0.8 = 4.0)
        ctrl.tracker.register("starving", arrival_time=0.0, input_len=100)
        # _get_worst_waiting_ttft uses time.monotonic(), but the internal
        # tracker uses arrival_time; set arrival very early
        ctrl.tracker._active["starving"].arrival_time = (
            time.monotonic() - 4.5)

        running = [_make_mock_request(f"r{i}", False) for i in range(50)]
        c = ctrl.get_constraints(running, [])
        # Anti-starvation should override cooldown and set admit_new=True
        assert c.admit_new is True

    def test_binary_search_max(self):
        # predict_fn returns x * 0.01, threshold=0.5 → max should be 50
        result = AdmissionController._binary_search_max(
            lo=1, hi=200,
            predict_fn=lambda x: x * 0.01,
            threshold=0.5,
        )
        assert result == 50

    def test_binary_search_lo_already_violates(self):
        # Even lo=1 violates → return lo
        result = AdmissionController._binary_search_max(
            lo=1, hi=100,
            predict_fn=lambda x: 1.0,  # always > threshold
            threshold=0.5,
        )
        assert result == 1

    def test_get_constraints_limits_batch(self):
        """After learning a cost model, batch limit should be < base_max."""
        ctrl = self._make_ctrl(tpot_slo=0.1, base_max=500, warmup=0)
        # teach: step_time = 0.001 * N (so for TPOT=0.1 → N <= 100)
        for _ in range(20):
            ctrl.cost_model.update(100, 0, 0, 0.1)
            ctrl.cost_model.update(200, 0, 0, 0.2)

        running = [_make_mock_request(f"r{i}", False, computed=500)
                   for i in range(80)]
        c = ctrl.get_constraints(running, [])
        # Should be well below 500
        assert c.max_decode_batch < 500

    def test_prefill_budget_limited(self):
        """Prefill budget should be less than base when decode is active."""
        ctrl = self._make_ctrl(tpot_slo=0.1, base_budget=16384, warmup=0)
        # teach: step_time increases with prefill tokens
        for _ in range(20):
            ctrl.cost_model.update(50, 0, 50000, 0.05)
            ctrl.cost_model.update(50, 8000, 50000, 0.15)
            ctrl.cost_model.update(50, 16384, 50000, 0.25)

        running = [_make_mock_request(f"r{i}", False, computed=1000)
                   for i in range(50)]
        c = ctrl.get_constraints(running, [])
        # Prefill budget should be < 16384
        assert c.max_prefill_tokens < 16384
