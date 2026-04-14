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

import csv
import io
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
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

class _PureDecodeQuantile:
    """Rolling p90 of recent pure-decode step durations, bucketed by
    (log2 n_decode, log2 total_kv).

    Pure-decode step times are dominated by non-linear noise (CUDA sync,
    block-table updates, compression kernels) and cannot be fit by a linear
    regression — offline R² ≈ 0.05 on 9k+ samples. For these steps we
    therefore use an empirical p90 as a pessimistic upper bound, which is
    exactly the quantity the admission controller's binary search wants.
    """

    def __init__(self, window: int = 200, min_samples: int = 10) -> None:
        self._window = window
        self._min_samples = min_samples
        self._buckets: dict[tuple[int, int], deque[float]] = {}

    @staticmethod
    def _bucket(n_decode: int, total_kv: int) -> tuple[int, int]:
        return (max(n_decode, 1).bit_length(),
                max(total_kv, 1).bit_length())

    def observe(self, n_decode: int, total_kv: int,
                actual: float) -> None:
        key = self._bucket(n_decode, total_kv)
        d = self._buckets.get(key)
        if d is None:
            d = deque(maxlen=self._window)
            self._buckets[key] = d
        d.append(actual)

    def predict(self, n_decode: int, total_kv: int) -> float | None:
        key = self._bucket(n_decode, total_kv)
        d = self._buckets.get(key)
        if d is None or len(d) < self._min_samples:
            return None
        return float(np.percentile(d, 90))


class StepCostModel:
    """
    Online step-time cost model.

    Two-path design:

    * Mixed steps (prefill_toks > 0): a linear regression
        step_time ≈ w·φ(x)  where
        φ(x) = [1, n_decode, total_kv, prefill_toks, prefill_toks*avg_kv]
      Feature 5 (prefill · average KV length) captures prefill
      cross-attention to the existing prefix and lifts offline R² on real
      workloads from ~0.66 to ~0.70. The weights are fit with
      Exponentially-Weighted Recursive Least Squares using forgetting
      factor λ.

    * Pure-decode steps (prefill_toks == 0): an empirical rolling p90
      from _PureDecodeQuantile, bucketed by (log2 n_decode, log2 total_kv).
      Linear regression is the wrong tool here — the signal is dominated
      by CUDA / Python / block-table noise with p99/p50 ≈ 30×.

    Numerical stability notes:

    * Features are internally normalized by _FEATURE_SCALE before every
      RLS step; otherwise per-coefficient effective learning rates would
      differ by ~10^12 (features span 6 orders of magnitude).
    * Weights are constrained to be non-negative after every update
      (doing more work cannot reduce step time).
    * The covariance matrix diagonal is capped at _P_MAX_DIAG to prevent
      "covariance wind-up" — exponential growth of uncertainty along
      directions that are not excited by recent samples, which was the
      main failure mode in real traces.
    * Samples whose residual is > 3× the prediction (after warmup) are
      dropped — a single GC / compression-kernel spike can otherwise
      destabilize the fit for hundreds of subsequent steps.

    The training signal is the duration of one engine iteration measured
    from the start of schedule() through the end of update_from_output()
    for the SAME step — NOT the wall-clock delta between successive
    schedule() calls. See Scheduler._caas_pending_steps.
    """

    _CSV_HEADER = [
        "step", "timestamp", "n_decode", "prefill_tokens", "total_kv",
        "predicted_sec", "actual_sec", "error_sec", "error_pct",
        "w0", "w1", "w2", "w3",
        "source", "is_warmed_up", "skipped",
    ]

    # Any single step that claims to have taken longer than this is almost
    # certainly contaminated by idle time (scheduler/worker stalls, GIL pauses,
    # a blocking queue wait). Drop it rather than poison the RLS fit.
    _MAX_PLAUSIBLE_STEP_SEC: float = 5.0

    # Maximum plausible feature magnitudes. Used to whiten the regression
    # input so (a) normalized features are bounded by ~1, (b) the isotropic
    # prior P₀ = σ²·I makes sense across features whose raw magnitudes
    # span 6 orders of magnitude, and (c) the normalized weights have a
    # direct physical reading of "maximum seconds this feature can add
    # to a step". Empirically calibrated from the caas-log traces; the
    # exact values are not sensitive (2x off is fine — the RLS will
    # compensate inside the cap).
    _FEATURE_SCALE: "np.ndarray" = np.array(
        [
            1.0,       # φ₀ = bias
            256.0,     # φ₁ = n_decode     (observed max 178)
            1.0e6,     # φ₂ = total_kv     (observed max 787k)
            2.0e4,     # φ₃ = prefill_toks (observed max 16384)
        ],
        dtype=np.float64,
    )

    # Per-feature cap on the (diagonal of the) normalized covariance.
    # With normalized features, "unit uncertainty" is O(1); 1e3 is ~30σ,
    # comfortably permissive while still preventing exponential wind-up.
    _P_MAX_DIAG: float = 1.0e3

    # Per-feature upper bound on normalized weights. With max-based
    # scales, w_norm[i] reads directly as "seconds of contribution at
    # max feature value" — so 1.5 means a single feature can claim up
    # to 1.5 seconds of step time when that feature is at its maximum,
    # a tight but workload-achievable envelope based on offline NNLS
    # calibration.
    _W_MAX_NORM: float = 1.5

    # Multiplicative shrinkage applied to the normalized weight vector
    # after every update. Acts as a per-step Tikhonov regularizer,
    # pulling unidentified (null-space) directions back toward 0 without
    # disturbing well-supported coefficients. With 2e-3, unsupported
    # weights decay ~86% over 1000 steps while a well-supported weight
    # with gradient ~0.1 only loses ~2%/step — comfortably replenished
    # by the RLS update.
    _SHRINKAGE: float = 2.0e-3

    # After warmup, drop samples whose residual is more than this times
    # the prediction — catches GC/compression spikes without rejecting
    # legitimate outliers during cold start (when prediction is ~0 and
    # the ratio is meaningless).
    _MAX_REL_RESIDUAL: float = 3.0

    def __init__(self, forgetting_factor: float = 0.999,
                 warmup_steps: int = 50,
                 log_dir: str | None = None) -> None:
        self.lam = forgetting_factor
        self.warmup_steps = warmup_steps
        self._n_features = 4
        # Weights live in the *normalized* feature basis throughout. The
        # "raw" weights are recovered as _w / _FEATURE_SCALE when we need
        # to log or interpret them.
        self._w = np.zeros(self._n_features, dtype=np.float64)
        # Isotropic prior in normalized space — all features are O(1).
        self._P = np.eye(self._n_features, dtype=np.float64) * 10.0
        self._step_count = 0
        self._skipped_count = 0

        # Empirical estimator for pure-decode steps (where linear
        # regression has no signal).
        self._pure_decode = _PureDecodeQuantile()

        # CSV logging
        self._csv_file: io.TextIOWrapper | None = None
        self._csv_writer: csv.writer | None = None
        if log_dir is not None:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            csv_path = log_path / f"caas_cost_model_{ts}.csv"
            self._csv_file = open(csv_path, "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(self._CSV_HEADER)
            self._csv_file.flush()
            logger.info("CAAS cost model logging to %s", csv_path)

    @property
    def is_warmed_up(self) -> bool:
        return self._step_count >= self.warmup_steps

    def _features(self, n_decode: int, prefill_toks: int,
                  total_kv: int) -> np.ndarray:
        """Raw (unscaled) 4-dim feature vector.

        φ₀ = 1                fixed per-step overhead
        φ₁ = n_decode         decode FFN / linear work
        φ₂ = total_kv         decode attention (sum of per-req KV lookups)
        φ₃ = prefill_toks     prefill FFN / linear + cross-attention

        Note: a 5th feature prefill_toks·avg_kv (prefill cross-attention
        to prefix) was considered and offline-evaluated. It raises pooled
        R² from 0.64 to 0.70 but its fitted coefficient is severely
        ill-conditioned across different workloads (per-file NNLS weights
        vary by 100×), making it unusable in an online setting where the
        training distribution shifts. We keep the simpler 4-feature form.
        """
        return np.array(
            [
                1.0,
                float(n_decode),
                float(total_kv),
                float(prefill_toks),
            ],
            dtype=np.float64,
        )

    def _normalized(self, n_decode: int, prefill_toks: int,
                    total_kv: int) -> np.ndarray:
        return (self._features(n_decode, prefill_toks, total_kv)
                / self._FEATURE_SCALE)

    def _linear_predict(self, n_decode: int, prefill_toks: int,
                        total_kv: int) -> float:
        x = self._normalized(n_decode, prefill_toks, total_kv)
        # Clamp into [0, _MAX_PLAUSIBLE_STEP_SEC] — a prediction beyond
        # _MAX_PLAUSIBLE_STEP_SEC means "worse than anything we'd ever
        # want to schedule", which is what the admission controller
        # actually cares about. Clamping here prevents pathological
        # weight vectors (from underdetermined fits on correlated data)
        # from propagating nonsense into the scheduler's binary search.
        raw = float(x @ self._w)
        if raw < 0.0:
            return 0.0
        if raw > self._MAX_PLAUSIBLE_STEP_SEC:
            return self._MAX_PLAUSIBLE_STEP_SEC
        return raw

    def _predict_with_source(
        self, n_decode: int, prefill_toks: int, total_kv: int,
    ) -> tuple[float, str]:
        """Return (prediction, source) where source is 'linear' or 'quantile'."""
        if prefill_toks == 0:
            q = self._pure_decode.predict(n_decode, total_kv)
            if q is not None:
                return q, "quantile"
        return self._linear_predict(n_decode, prefill_toks, total_kv), "linear"

    def predict(self, n_decode: int, prefill_toks: int,
                total_kv: int) -> float:
        """Predict step duration in seconds."""
        return self._predict_with_source(n_decode, prefill_toks, total_kv)[0]

    def _cap_covariance(self) -> None:
        """Anti-windup: scale down rows/columns of P whose diagonal entries
        exceed _P_MAX_DIAG. Symmetric scaling preserves positive
        semidefiniteness."""
        diag = np.diag(self._P).copy()
        overflow = diag > self._P_MAX_DIAG
        if not overflow.any():
            return
        # Scale factor per feature: sqrt(max / current) ≤ 1 for overflowing
        # entries, 1 otherwise. Applying D·P·D shrinks both the row and
        # the column, and squares to the desired diagonal cap.
        scale = np.ones_like(diag)
        scale[overflow] = np.sqrt(self._P_MAX_DIAG / diag[overflow])
        D = np.diag(scale)
        self._P = D @ self._P @ D

    def update(self, n_decode: int, prefill_toks: int, total_kv: int,
               actual_time: float) -> None:
        """Update model with an observed step duration.

        Samples that are obviously contaminated (empty step, non-positive
        duration, or longer than _MAX_PLAUSIBLE_STEP_SEC) are logged but
        not fed to the fit. After warmup, samples whose residual exceeds
        _MAX_REL_RESIDUAL × prediction are also dropped.
        """
        # First-line outlier filter — obvious contamination.
        skipped = (
            actual_time <= 0.0
            or actual_time > self._MAX_PLAUSIBLE_STEP_SEC
            or (n_decode == 0 and prefill_toks == 0)
        )

        predicted, source = self._predict_with_source(
            n_decode, prefill_toks, total_kv)

        # Second-line outlier filter — only trip after warmup, and only
        # when we have a meaningful prediction to compare against.
        if (not skipped and self.is_warmed_up and predicted > 0.02
                and abs(predicted - actual_time)
                > self._MAX_REL_RESIDUAL * predicted):
            logger.debug(
                "CAAS: dropped spike sample "
                "(predicted=%.3fs actual=%.3fs ratio=%.1f, N=%d P=%d KV=%d)",
                predicted, actual_time, actual_time / max(predicted, 1e-9),
                n_decode, prefill_toks, total_kv,
            )
            skipped = True

        # Log prediction-vs-actual BEFORE updating weights.
        if self._csv_writer is not None:
            error_sec = predicted - actual_time
            error_pct = (
                error_sec / actual_time * 100.0
                if actual_time > 0 else float("nan")
            )
            # Log weights in raw (de-normalized) basis so their magnitudes
            # are directly interpretable as seconds-per-feature-unit.
            w_raw = self._w / self._FEATURE_SCALE
            self._csv_writer.writerow([
                self._step_count,
                f"{time.time():.6f}",
                n_decode,
                prefill_toks,
                total_kv,
                f"{predicted:.9f}",
                f"{actual_time:.9f}",
                f"{error_sec:.9f}",
                f"{error_pct:.4f}",
                f"{w_raw[0]:.9e}",
                f"{w_raw[1]:.9e}",
                f"{w_raw[2]:.9e}",
                f"{w_raw[3]:.9e}",
                source,
                self.is_warmed_up,
                skipped,
            ])
            self._csv_file.flush()

        if skipped:
            self._skipped_count += 1
            if actual_time > self._MAX_PLAUSIBLE_STEP_SEC:
                logger.debug(
                    "CAAS: dropped contaminated step sample "
                    "(actual=%.2fs > %.1fs threshold, N=%d P=%d KV=%d)",
                    actual_time, self._MAX_PLAUSIBLE_STEP_SEC,
                    n_decode, prefill_toks, total_kv,
                )
            return

        # Feed pure-decode samples into the empirical quantile estimator
        # (in addition to — not instead of — the linear fit, so the
        # linear model stays useful as a cold-start fallback).
        if prefill_toks == 0:
            self._pure_decode.observe(n_decode, total_kv, actual_time)

        # --- Stable RLS update in normalized feature space ----------------
        x = self._normalized(n_decode, prefill_toks, total_kv)
        Px = self._P @ x
        denom = self.lam + float(x @ Px)
        if abs(denom) < 1e-12:
            self._step_count += 1
            return
        K = Px / denom
        error = actual_time - float(x @ self._w)
        self._w = self._w + K * error
        self._P = (self._P - np.outer(K, Px)) / self.lam
        # Symmetrize to kill floating-point drift that would otherwise
        # eventually break the PSD invariant.
        self._P = 0.5 * (self._P + self._P.T)
        # Shrink toward 0 — pulls ridge-drift / null-space components
        # back to the prior without affecting well-supported weights
        # (which are replenished by the RLS gradient each step).
        self._w *= (1.0 - self._SHRINKAGE)
        # Physical constraints:
        #   lower bound: every coefficient is non-negative (more work
        #     cannot reduce step time).
        #   upper bound: no single normalized feature should contribute
        #     more than _W_MAX_NORM seconds — anything beyond that is the
        #     signature of an overfit on a collinear training subset.
        np.clip(self._w, 0.0, self._W_MAX_NORM, out=self._w)
        # Anti-windup: cap covariance diagonal.
        self._cap_covariance()
        self._step_count += 1

    def close_log(self) -> None:
        """Close the CSV log file if open."""
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None


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
        warmup_steps: int = 50,
        forgetting_factor: float = 0.95,
        log_dir: str | None = None,
    ) -> None:
        self.ttft_slo = ttft_slo
        self.tpot_slo = tpot_slo
        self._base_max_running = base_max_running
        self._base_token_budget = base_token_budget

        self.cost_model = StepCostModel(
            forgetting_factor=forgetting_factor,
            warmup_steps=warmup_steps,
            log_dir=log_dir,
        )
        self.tracker = RequestTracker()

        self._in_post_compression: bool = False

    def on_compression_event(self, num_compressed: int) -> None:
        """Signal that KV compression just freed blocks for `num_compressed`
        requests.  Enters cost-model-gated observation period."""
        if num_compressed > 0:
            self._in_post_compression = True
            logger.debug(
                "CAAS: compression event (%d reqs) → cost-model cooldown",
                num_compressed,
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

        # 3) Post-compression cost-model-gated cooldown
        admit_new = True
        if self._in_post_compression and len(running) > 0:
            # Ask cost model whether admitting one more request would
            # still stay within the TPOT SLO.
            predicted = self.cost_model.predict(
                n_decode + 1, 0, int((n_decode + 1) * avg_kv))
            if predicted <= self.tpot_slo:
                self._in_post_compression = False
                logger.debug("CAAS: cooldown lifted (predicted "
                             "%.3fs <= SLO %.3fs)", predicted, self.tpot_slo)
            else:
                admit_new = False
                max_decode_batch = min(max_decode_batch, len(running))
                logger.debug("CAAS: cooldown held (predicted "
                             "%.3fs > SLO %.3fs)", predicted, self.tpot_slo)

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
