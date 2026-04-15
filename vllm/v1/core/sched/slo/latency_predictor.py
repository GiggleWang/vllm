# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Latency predictors for SLO-aware spatial scheduling.

Three implementations are provided:

1. ``HeuristicPredictor``
   Affine model with hard-coded default coefficients.  No profiling required.
   ``latency_ms = BASE + A*num_reqs + B*num_new_tokens + C*num_prefill_tokens``

2. ``OfflineProfilePredictor``
   Loads a JSON profile produced by ``vllm/tools/profile_step_latency.py`` and
   fits (or loads a pre-fit) affine model.  Falls back to ``HeuristicPredictor``
   when the profile file is missing.

3. ``OnlinePredictor``
   Maintains a running EMA-updated linear regression over observed
   (batch_features, step_latency) samples.  Uses ``HeuristicPredictor`` for
   cold-start until ``WARMUP_SAMPLES`` observations have been collected.
"""

from __future__ import annotations

import json
import logging
import math
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)

# Default heuristic coefficients (milliseconds).
# Tuned for a mid-size GPU with typical transformer workloads; override via
# the offline profile for accuracy.
_DEFAULT_BASE_MS: float = 10.0
_DEFAULT_COEF_REQS: float = 0.5    # per additional running request
_DEFAULT_COEF_TOKENS: float = 0.05  # per additional new token (decode-heavy)
_DEFAULT_COEF_PREFILL: float = 0.12  # per additional prefill token


class LatencyPredictor(ABC):
    """Abstract base class for per-step latency predictors."""

    @abstractmethod
    def predict(
        self,
        *,
        num_reqs: int,
        num_new_tokens: int,
        num_prefill_tokens_in_batch: int,
    ) -> float:
        """Return predicted step latency in **milliseconds**."""

    def observe(
        self,
        *,
        num_reqs: int,
        num_new_tokens: int,
        num_prefill_tokens_in_batch: int,
        measured_ms: float,
    ) -> None:
        """Feed a measured (batch_features, latency) sample.

        Default implementation is a no-op; override in online learners.
        """


# ---------------------------------------------------------------------------
# 1. Heuristic predictor
# ---------------------------------------------------------------------------

class HeuristicPredictor(LatencyPredictor):
    """Simple affine predictor with configurable coefficients.

    ``latency_ms = base_ms
                 + coef_reqs * num_reqs
                 + coef_tokens * num_new_tokens
                 + coef_prefill * num_prefill_tokens_in_batch``
    """

    def __init__(
        self,
        base_ms: float = _DEFAULT_BASE_MS,
        coef_reqs: float = _DEFAULT_COEF_REQS,
        coef_tokens: float = _DEFAULT_COEF_TOKENS,
        coef_prefill: float = _DEFAULT_COEF_PREFILL,
    ) -> None:
        self.base_ms = base_ms
        self.coef_reqs = coef_reqs
        self.coef_tokens = coef_tokens
        self.coef_prefill = coef_prefill

    def predict(
        self,
        *,
        num_reqs: int,
        num_new_tokens: int,
        num_prefill_tokens_in_batch: int,
    ) -> float:
        return (
            self.base_ms
            + self.coef_reqs * num_reqs
            + self.coef_tokens * num_new_tokens
            + self.coef_prefill * num_prefill_tokens_in_batch
        )


# ---------------------------------------------------------------------------
# 2. Offline profile predictor
# ---------------------------------------------------------------------------

class OfflineProfilePredictor(LatencyPredictor):
    """Affine predictor whose coefficients are loaded from a JSON profile.

    Expected JSON schema (produced by ``vllm/tools/profile_step_latency.py``)::

        {
          "base_ms": 8.5,
          "coef_reqs": 0.4,
          "coef_tokens": 0.06,
          "coef_prefill": 0.15
        }

    Falls back to ``HeuristicPredictor`` with default coefficients when the
    profile file is missing or malformed.
    """

    def __init__(self, profile_path: str | None) -> None:
        self._fallback = HeuristicPredictor()
        self._delegate: LatencyPredictor = self._fallback

        if profile_path is None:
            logger.warning(
                "OfflineProfilePredictor: no profile_path provided, "
                "falling back to heuristic coefficients."
            )
            return

        path = Path(profile_path)
        if not path.exists():
            logger.warning(
                "OfflineProfilePredictor: profile file %s not found, "
                "falling back to heuristic coefficients.",
                profile_path,
            )
            return

        try:
            with path.open() as f:
                data = json.load(f)
            self._delegate = HeuristicPredictor(
                base_ms=float(data["base_ms"]),
                coef_reqs=float(data["coef_reqs"]),
                coef_tokens=float(data["coef_tokens"]),
                coef_prefill=float(data["coef_prefill"]),
            )
            logger.info(
                "OfflineProfilePredictor: loaded coefficients from %s: %s",
                profile_path,
                data,
            )
        except Exception as exc:
            logger.warning(
                "OfflineProfilePredictor: failed to load %s (%s), "
                "falling back to heuristic coefficients.",
                profile_path,
                exc,
            )

    def predict(
        self,
        *,
        num_reqs: int,
        num_new_tokens: int,
        num_prefill_tokens_in_batch: int,
    ) -> float:
        return self._delegate.predict(
            num_reqs=num_reqs,
            num_new_tokens=num_new_tokens,
            num_prefill_tokens_in_batch=num_prefill_tokens_in_batch,
        )


# ---------------------------------------------------------------------------
# 3. Online predictor
# ---------------------------------------------------------------------------

_WARMUP_SAMPLES: int = 32
_EMA_ALPHA: float = 0.05  # weight of new sample; 1-alpha for old estimate


class OnlinePredictor(LatencyPredictor):
    """EMA-updated linear regression predictor.

    Uses ``HeuristicPredictor`` during a warm-up phase (first
    ``WARMUP_SAMPLES`` observations).  After that, maintains an affine model
    fitted online via exponential moving average updates on the coefficient
    vector.

    The feature vector is ``[1, num_reqs, num_new_tokens, num_prefill_tokens]``
    and the target is ``measured_ms``.

    Thread safety: no locks — the scheduler loop is single-threaded.
    """

    def __init__(self) -> None:
        self._warmup = HeuristicPredictor()
        self._n_samples: int = 0
        # Coefficient vector: [base, coef_reqs, coef_tokens, coef_prefill]
        self._w: list[float] = [
            _DEFAULT_BASE_MS,
            _DEFAULT_COEF_REQS,
            _DEFAULT_COEF_TOKENS,
            _DEFAULT_COEF_PREFILL,
        ]

    def _features(
        self,
        num_reqs: int,
        num_new_tokens: int,
        num_prefill_tokens_in_batch: int,
    ) -> list[float]:
        return [1.0, float(num_reqs), float(num_new_tokens),
                float(num_prefill_tokens_in_batch)]

    def predict(
        self,
        *,
        num_reqs: int,
        num_new_tokens: int,
        num_prefill_tokens_in_batch: int,
    ) -> float:
        if self._n_samples < _WARMUP_SAMPLES:
            return self._warmup.predict(
                num_reqs=num_reqs,
                num_new_tokens=num_new_tokens,
                num_prefill_tokens_in_batch=num_prefill_tokens_in_batch,
            )
        x = self._features(num_reqs, num_new_tokens, num_prefill_tokens_in_batch)
        return sum(w * xi for w, xi in zip(self._w, x))

    def observe(
        self,
        *,
        num_reqs: int,
        num_new_tokens: int,
        num_prefill_tokens_in_batch: int,
        measured_ms: float,
    ) -> None:
        if math.isnan(measured_ms) or measured_ms <= 0:
            return
        self._n_samples += 1
        x = self._features(num_reqs, num_new_tokens, num_prefill_tokens_in_batch)
        pred = sum(w * xi for w, xi in zip(self._w, x))
        error = measured_ms - pred
        # Gradient descent step: w_i += alpha * error * x_i
        self._w = [
            w + _EMA_ALPHA * error * xi for w, xi in zip(self._w, x)
        ]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_latency_predictor(
    kind: str,
    profile_path: str | None = None,
) -> LatencyPredictor:
    """Create a ``LatencyPredictor`` by name.

    Parameters
    ----------
    kind:
        One of ``"heuristic"``, ``"offline"``, ``"online"``.
    profile_path:
        Path to JSON profile file.  Required when *kind* is ``"offline"``;
        ignored otherwise.
    """
    if kind == "heuristic":
        return HeuristicPredictor()
    if kind == "offline":
        return OfflineProfilePredictor(profile_path)
    if kind == "online":
        return OnlinePredictor()
    raise ValueError(
        f"Unknown latency predictor kind {kind!r}. "
        "Expected one of: 'heuristic', 'offline', 'online'."
    )
