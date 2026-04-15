# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Urgency function for SLO-aware scheduling.

Urgency U(req, now) is a dimensionless "slack ratio":
    U = slack_s / tpot_target_s

where slack_s = next_deadline_ts - now.

Convention:
  - Smaller U → more urgent (should be scheduled sooner).
  - U < 0       → deadline already missed.
  - U = +inf    → request has no SLO (deprioritised vs. any SLO'd request).

The "next deadline" is the tightest of:
  - TTFT deadline  (only active while the request has 0 output tokens)
  - TPOT deadline  (only active after the first output token)
  - E2E deadline   (always active; linearised into per-step urgency)
"""

import math
import time

_EPS = 1e-6


def request_urgency(
    req: "Request",  # type: ignore[name-defined]  # noqa: F821
    now: float,
    default_tpot_s: float = 0.05,
) -> float:
    """Return the urgency scalar for *req* at monotonic time *now*.

    Parameters
    ----------
    req:
        A ``vllm.v1.request.Request`` instance.
    now:
        Current ``time.monotonic()`` value.
    default_tpot_s:
        Fallback TPOT budget (seconds) used for normalisation when the
        request has no explicit ``slo_tpot_ms``.  Also used as the per-step
        time budget when estimating E2E urgency.

    Returns
    -------
    float
        Urgency scalar.  Smaller = more urgent.  ``math.inf`` for requests
        with no SLO.
    """
    tpot_s = req.tpot_budget_s if req.tpot_budget_s is not None else default_tpot_s

    # Collect candidate per-step deadlines.
    next_deadline_ts: float | None = None

    # --- TTFT deadline (only before first output token) ---
    if req.num_output_tokens == 0 and req.ttft_deadline_ts is not None:
        next_deadline_ts = req.ttft_deadline_ts

    # --- TPOT deadline (after first token) ---
    if req.num_output_tokens >= 1 and req.tpot_budget_s is not None:
        last_ts = req.last_token_mono_ts
        if last_ts is None:
            # Shouldn't happen, but fall back to arrival.
            last_ts = req.mono_arrival_ts
        tpot_deadline = last_ts + req.tpot_budget_s
        if next_deadline_ts is None or tpot_deadline < next_deadline_ts:
            next_deadline_ts = tpot_deadline

    # --- E2E deadline (linearised to per-step) ---
    if req.e2e_deadline_ts is not None:
        # Estimate remaining steps.  Use max_tokens if set (common case).
        max_out = req.max_tokens
        remaining_tokens = max(1, max_out - req.num_output_tokens)
        # Linearly apportion the remaining E2E budget across remaining steps.
        remaining_budget_s = req.e2e_deadline_ts - now
        per_step_deadline = now + remaining_budget_s / remaining_tokens
        if next_deadline_ts is None or per_step_deadline < next_deadline_ts:
            next_deadline_ts = per_step_deadline

    if next_deadline_ts is None:
        return math.inf

    slack_s = next_deadline_ts - now
    return slack_s / max(tpot_s, _EPS)
