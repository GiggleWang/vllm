# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SLO diagnostic recorder.

Enabled by setting the ``VLLM_SLO_DEBUG_DIR`` environment variable to an
output directory. When unset, every hook is a single ``is None`` check and
has zero runtime cost.

The recorder writes 6 CSV files, one per diagnostic "module":

    slo_ingress.csv   Module A: did SLO fields reach the scheduler?
    slo_rank.csv      Module B: is the queue actually ordered by urgency?
    slo_predict.csv   Module C: is the latency predictor accurate?
    slo_budget.csv    Module D: is the token budget mechanism active?
    slo_preempt.csv   Module E: is admission-time preemption firing?
    slo_finish.csv    Module F: per-request outcome for attribution.

All rows carry a monotonic ``ts`` column so they can be joined across
files. ``step_id`` monotonically increases each time the scheduler is
about to run one forward pass.
"""
from __future__ import annotations

import atexit
import csv
import math
import os
import threading
import time
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.v1.request import Request


_SCHEMAS: dict[str, list[str]] = {
    "ingress": [
        "ts", "request_id", "has_sp",
        "slo_ttft_ms", "slo_tpot_ms", "slo_e2e_ms",
        "ttft_deadline_ts", "tpot_budget_s", "e2e_deadline_ts",
        "mono_arrival_ts",
    ],
    "rank": [
        "ts", "step_id",
        "n_waiting", "n_running",
        "n_waiting_inf", "n_running_inf",
        "u_wait_p10", "u_wait_p50", "u_wait_p90",
        "u_run_p10", "u_run_p50", "u_run_p90",
        "wait_head_rid", "wait_head_u", "wait_head_arrival",
        "fcfs_head_rid", "fcfs_head_arrival",
        "head_differs",
    ],
    "predict": [
        "ts", "step_id",
        "num_reqs", "num_new_tokens", "num_prefill_tokens",
        "predicted_ms", "measured_ms",
        "abs_err_ms", "rel_err",
    ],
    "budget": [
        "ts", "step_id",
        "hard_cap", "returned_budget",
        "total_reqs", "slo_reqs",
        "min_slack_s", "target_ms",
        "pred_at_hard", "pred_at_returned",
        "budget_active",
    ],
    "preempt": [
        "ts", "step_id", "kind",
        "victim_rid", "victim_u",
        "waiting_rid", "waiting_u",
        "ttft_miss_ms",
    ],
    "finish": [
        "ts", "request_id",
        "slo_ttft_ms", "slo_tpot_ms", "slo_e2e_ms",
        "num_prompt_tokens", "num_output_tokens",
        "queued_time_s", "prefill_time_s", "decode_time_s", "e2e_latency_s",
        "first_token_latency_s",
        "ttft_violated", "tpot_violated", "e2e_violated",
        "ttft_slack_s", "e2e_slack_s", "tpot_violation_count",
        "num_preemptions",
    ],
}


class _Recorder:
    def __init__(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._out_dir = out_dir
        self._files: dict[str, IO[str]] = {}
        self._writers: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._step_id: int = 0

        for name, header in _SCHEMAS.items():
            f = (out_dir / f"slo_{name}.csv").open("w", newline="", buffering=1)
            self._files[name] = f
            w = csv.writer(f)
            w.writerow(header)
            self._writers[name] = w

        atexit.register(self.close)

    # -- step id ---------------------------------------------------------

    def next_step_id(self) -> int:
        # Caller wraps this in its own lock-free context; step_id only
        # needs to be roughly-monotonic and unique per scheduler step.
        self._step_id += 1
        return self._step_id

    @property
    def step_id(self) -> int:
        return self._step_id

    # -- writing ---------------------------------------------------------

    def _write(self, module: str, row: list[Any]) -> None:
        with self._lock:
            self._writers[module].writerow(row)

    # -- Module A: ingress ----------------------------------------------

    def record_ingress(self, req: "Request") -> None:
        try:
            sp = req.sampling_params
            has_sp = sp is not None
            self._write("ingress", [
                f"{time.monotonic():.6f}",
                req.request_id,
                int(has_sp),
                sp.slo_ttft_ms if has_sp else "",
                sp.slo_tpot_ms if has_sp else "",
                sp.slo_e2e_ms if has_sp else "",
                _fmt(req.ttft_deadline_ts),
                _fmt(req.tpot_budget_s),
                _fmt(req.e2e_deadline_ts),
                f"{req.mono_arrival_ts:.6f}",
            ])
        except Exception:
            # Never let debug logging break the engine.
            pass

    # -- Module B: ranking ----------------------------------------------

    def record_rank(
        self,
        step_id: int,
        now: float,
        waiting_iter,
        running: list,
        default_tpot_s: float,
    ) -> None:
        try:
            from vllm.v1.core.sched.slo.urgency import request_urgency

            wait_reqs = list(waiting_iter)  # may re-sort internally; cheap
            u_wait = [request_urgency(r, now, default_tpot_s) for r in wait_reqs]
            u_run = [request_urgency(r, now, default_tpot_s) for r in running]

            wait_head = wait_reqs[0] if wait_reqs else None
            # What FCFS would have picked: earliest arrival_time among waiting.
            fcfs_head = (min(wait_reqs, key=lambda r: r.arrival_time)
                         if wait_reqs else None)
            head_differs = (
                1 if (wait_head is not None
                      and fcfs_head is not None
                      and wait_head.request_id != fcfs_head.request_id)
                else 0
            )

            self._write("rank", [
                f"{now:.6f}", step_id,
                len(wait_reqs), len(running),
                sum(1 for u in u_wait if math.isinf(u)),
                sum(1 for u in u_run if math.isinf(u)),
                _pct(u_wait, 10), _pct(u_wait, 50), _pct(u_wait, 90),
                _pct(u_run, 10), _pct(u_run, 50), _pct(u_run, 90),
                wait_head.request_id if wait_head else "",
                _fmt(u_wait[0]) if u_wait else "",
                f"{wait_head.arrival_time:.6f}" if wait_head else "",
                fcfs_head.request_id if fcfs_head else "",
                f"{fcfs_head.arrival_time:.6f}" if fcfs_head else "",
                head_differs,
            ])
        except Exception:
            pass

    # -- Module C: predictor --------------------------------------------

    def record_predict(
        self,
        step_id: int,
        num_reqs: int,
        num_new_tokens: int,
        num_prefill_tokens: int,
        predicted_ms: float,
        measured_ms: float,
    ) -> None:
        try:
            err = measured_ms - predicted_ms
            rel = (err / measured_ms) if measured_ms > 1e-6 else 0.0
            self._write("predict", [
                f"{time.monotonic():.6f}", step_id,
                num_reqs, num_new_tokens, num_prefill_tokens,
                f"{predicted_ms:.3f}", f"{measured_ms:.3f}",
                f"{err:.3f}", f"{rel:.4f}",
            ])
        except Exception:
            pass

    # -- Module D: budget -----------------------------------------------

    def record_budget(
        self,
        step_id: int,
        hard_cap: int,
        returned: int,
        total_reqs: int,
        slo_reqs: int,
        min_slack_s: float,
        target_ms: float,
        pred_at_hard: float,
        pred_at_returned: float,
    ) -> None:
        try:
            self._write("budget", [
                f"{time.monotonic():.6f}", step_id,
                hard_cap, returned,
                total_reqs, slo_reqs,
                _fmt(min_slack_s), _fmt(target_ms),
                _fmt(pred_at_hard), _fmt(pred_at_returned),
                int(returned < hard_cap),
            ])
        except Exception:
            pass

    # -- Module E: preempt ----------------------------------------------

    def record_preempt(
        self,
        step_id: int,
        kind: str,
        victim_rid: str,
        victim_u: float,
        waiting_rid: str = "",
        waiting_u: float = math.nan,
        ttft_miss_ms: float = math.nan,
    ) -> None:
        try:
            self._write("preempt", [
                f"{time.monotonic():.6f}", step_id, kind,
                victim_rid, _fmt(victim_u),
                waiting_rid, _fmt(waiting_u),
                _fmt(ttft_miss_ms),
            ])
        except Exception:
            pass

    # -- Module F: finish -----------------------------------------------

    def record_finish(
        self,
        request_id: str,
        req_stats,
        finished_stats,
        num_prompt_tokens: int,
        num_output_tokens: int,
        num_preemptions: int,
    ) -> None:
        try:
            queued_time = max(
                0.0,
                (req_stats.scheduled_ts - req_stats.queued_ts)
                if req_stats.queued_ts else 0.0,
            )
            prefill_time = max(
                0.0,
                (req_stats.first_token_ts - req_stats.scheduled_ts)
                if req_stats.scheduled_ts else 0.0,
            )
            decode_time = max(
                0.0,
                (req_stats.last_token_ts - req_stats.first_token_ts)
                if req_stats.first_token_ts else 0.0,
            )
            self._write("finish", [
                f"{time.monotonic():.6f}", request_id,
                req_stats.slo_ttft_ms or "",
                req_stats.slo_tpot_ms or "",
                req_stats.slo_e2e_ms or "",
                num_prompt_tokens, num_output_tokens,
                f"{queued_time:.4f}",
                f"{prefill_time:.4f}",
                f"{decode_time:.4f}",
                f"{finished_stats.e2e_latency:.4f}",
                f"{req_stats.first_token_latency:.4f}",
                int(finished_stats.ttft_violated),
                int(finished_stats.tpot_violated),
                int(finished_stats.e2e_violated),
                f"{finished_stats.ttft_slack_s:.4f}",
                f"{finished_stats.e2e_slack_s:.4f}",
                req_stats.tpot_violation_count,
                num_preemptions,
            ])
        except Exception:
            pass

    # -- shutdown -------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            for f in self._files.values():
                try:
                    f.flush()
                    f.close()
                except Exception:
                    pass
            self._files.clear()
            self._writers.clear()


def _fmt(x: float | None) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "inf" if (isinstance(x, float) and math.isinf(x) and x > 0) else (
            "-inf" if isinstance(x, float) and math.isinf(x) else "nan"
        )
    return f"{x:.6f}"


def _pct(values: list[float], q: int) -> str:
    finite = [v for v in values if not math.isinf(v) and not math.isnan(v)]
    if not finite:
        return ""
    finite.sort()
    k = (len(finite) - 1) * q / 100.0
    f = int(k)
    c = min(f + 1, len(finite) - 1)
    if f == c:
        return f"{finite[f]:.4f}"
    v = finite[f] + (finite[c] - finite[f]) * (k - f)
    return f"{v:.4f}"


# --------------------------------------------------------------------------
# Public accessor
# --------------------------------------------------------------------------

_RECORDER: _Recorder | None = None
_INIT_LOCK = threading.Lock()
_INITIALIZED = False


def get_recorder() -> _Recorder | None:
    """Return the debug recorder singleton, or None if disabled."""
    global _RECORDER, _INITIALIZED
    if _INITIALIZED:
        return _RECORDER
    with _INIT_LOCK:
        if _INITIALIZED:
            return _RECORDER
        out_dir = os.environ.get("VLLM_SLO_DEBUG_DIR")
        if out_dir:
            _RECORDER = _Recorder(Path(out_dir))
        _INITIALIZED = True
        return _RECORDER
