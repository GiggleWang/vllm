# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SLO-aware scheduling components for the vLLM v1 scheduler."""

from vllm.v1.core.sched.slo.latency_predictor import (
    LatencyPredictor,
    create_latency_predictor,
)
from vllm.v1.core.sched.slo.urgency import request_urgency

__all__ = [
    "LatencyPredictor",
    "create_latency_predictor",
    "request_urgency",
]
