# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline step-latency profiler for SLO-aware scheduling.

Sweeps a grid of (num_reqs, num_new_tokens, prefill_share) combinations,
measures the GPU forward-pass latency for each, and fits an affine model:

    latency_ms ≈ base_ms
                 + coef_reqs * num_reqs
                 + coef_tokens * num_new_tokens
                 + coef_prefill * num_prefill_tokens_in_batch

Outputs:
  - A CSV file with raw measurements.
  - A JSON file with the fitted coefficients, ready for
    ``--slo-predictor offline --slo-profile-path <file>``.

Usage::

    python -m vllm.tools.profile_step_latency \\
        --model Qwen/Qwen3-0.6B \\
        --output /tmp/profile.json \\
        [--csv /tmp/profile.csv] \\
        [--warmup 5] \\
        [--repeats 20]

    # Then start vLLM with:
    vllm serve Qwen/Qwen3-0.6B \\
        --scheduling-policy slo --slo-enable \\
        --slo-predictor offline --slo-profile-path /tmp/profile.json
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

_NUM_REQS_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
_NUM_TOKENS_GRID = [1, 8, 32, 64, 128, 256, 512, 1024, 2048, 4096]
_PREFILL_SHARE_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]

_WARMUP_STEPS = 5
_MEASURE_STEPS = 20


# ---------------------------------------------------------------------------
# Simple least-squares fit (no numpy dependency required)
# ---------------------------------------------------------------------------

def _fit_affine(
    samples: list[dict[str, Any]],
) -> dict[str, float]:
    """Fit ``y = w0 + w1*x1 + w2*x2 + w3*x3`` via normal equations.

    Features: [1, num_reqs, num_new_tokens, num_prefill_tokens_in_batch]
    Target: measured_ms
    """
    # Build feature matrix X and target vector y.
    rows: list[list[float]] = []
    ys: list[float] = []
    for s in samples:
        rows.append([
            1.0,
            float(s["num_reqs"]),
            float(s["num_new_tokens"]),
            float(s["num_prefill_tokens"]),
        ])
        ys.append(float(s["mean_ms"]))

    n = len(rows)
    d = len(rows[0])

    # XtX = X^T X, Xty = X^T y
    XtX = [[0.0] * d for _ in range(d)]
    Xty = [0.0] * d
    for i in range(n):
        for j in range(d):
            Xty[j] += rows[i][j] * ys[i]
            for k in range(d):
                XtX[j][k] += rows[i][j] * rows[i][k]

    # Solve XtX * w = Xty using Gaussian elimination.
    # Augment with identity for backsubstitution.
    aug = [XtX[i][:] + [Xty[i]] for i in range(d)]
    for col in range(d):
        # Pivot.
        pivot_row = max(range(col, d), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        if abs(aug[col][col]) < 1e-12:
            continue
        scale = aug[col][col]
        aug[col] = [x / scale for x in aug[col]]
        for r in range(d):
            if r != col:
                factor = aug[r][col]
                aug[r] = [aug[r][c] - factor * aug[col][c] for c in range(d + 1)]

    w = [aug[i][d] for i in range(d)]
    return {
        "base_ms": w[0],
        "coef_reqs": w[1],
        "coef_tokens": w[2],
        "coef_prefill": w[3],
    }


# ---------------------------------------------------------------------------
# Profiling logic
# ---------------------------------------------------------------------------

def _run_profiling(
    model: str,
    warmup_steps: int,
    measure_steps: int,
    max_model_len: int,
    tensor_parallel_size: int,
) -> list[dict[str, Any]]:
    """Run the profiling sweep and return a list of measurement records."""
    import torch
    from vllm import LLM, SamplingParams

    print(f"Loading model {model!r} for latency profiling ...")
    llm = LLM(
        model=model,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=True,  # Disable CUDA graphs for more stable timing.
    )

    records: list[dict[str, Any]] = []

    grid = [
        (nr, nt, ps)
        for nr in _NUM_REQS_GRID
        for nt in _NUM_TOKENS_GRID
        for ps in _PREFILL_SHARE_GRID
    ]
    total = len(grid)

    for idx, (num_reqs, num_tokens, prefill_share) in enumerate(grid):
        # Skip if num_tokens < num_reqs: in real serving each request
        # produces at least 1 token per step, so this combination
        # cannot occur and would record misleading features.
        if num_tokens < num_reqs:
            print(f"  [{idx+1}/{total}] Skipping num_reqs={num_reqs} "
                  f"num_tokens={num_tokens}: num_tokens < num_reqs")
            continue

        # Skip if total tokens would exceed model capacity.
        tokens_per_req = max(1, num_tokens // num_reqs)
        if tokens_per_req > max_model_len - 1:
            print(f"  [{idx+1}/{total}] Skipping num_reqs={num_reqs} "
                  f"num_tokens={num_tokens}: tokens_per_req={tokens_per_req} "
                  f"> max_model_len-1={max_model_len-1}")
            continue

        num_prefill = int(num_tokens * prefill_share)

        # Build a synthetic batch: ``num_reqs`` requests each with
        # ``tokens_per_req`` prompt tokens.
        prompt_ids = list(range(1, tokens_per_req + 1))  # dummy token IDs
        prompts = [{"prompt_token_ids": prompt_ids} for _ in range(num_reqs)]
        sampling = SamplingParams(max_tokens=1, temperature=0.0)

        # Warm up.
        for _ in range(warmup_steps):
            llm.generate(prompts, sampling, use_tqdm=False)

        # Measure.
        latencies_ms: list[float] = []
        for _ in range(measure_steps):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            llm.generate(prompts, sampling, use_tqdm=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        mean_ms = sum(latencies_ms) / len(latencies_ms)
        record = {
            "num_reqs": num_reqs,
            "num_new_tokens": num_tokens,
            "num_prefill_tokens": num_prefill,
            "prefill_share": prefill_share,
            "mean_ms": mean_ms,
            "min_ms": min(latencies_ms),
            "max_ms": max(latencies_ms),
        }
        records.append(record)
        print(
            f"  [{idx+1}/{total}] num_reqs={num_reqs:3d} "
            f"num_tokens={num_tokens:5d} prefill_share={prefill_share:.1f} "
            f"-> mean={mean_ms:.2f} ms  "
            f"min={record['min_ms']:.2f} max={record['max_ms']:.2f}"
        )

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline step-latency profiler for SLO-aware scheduling.",
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Model name or path to profile.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for the fitted JSON profile "
                             "(used with --slo-profile-path).")
    parser.add_argument("--csv", type=str, default=None,
                        help="Optional path to save raw measurement CSV.")
    parser.add_argument("--warmup", type=int, default=_WARMUP_STEPS,
                        help=f"Warmup steps per grid point (default: {_WARMUP_STEPS}).")
    parser.add_argument("--repeats", type=int, default=_MEASURE_STEPS,
                        help=f"Measurement steps per grid point "
                             f"(default: {_MEASURE_STEPS}).")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="Max model sequence length (default: 4096).")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Tensor parallel size (default: 1).")
    args = parser.parse_args()

    print("=" * 60)
    print("  vLLM Step-Latency Profiler")
    print("=" * 60)

    records = _run_profiling(
        model=args.model,
        warmup_steps=args.warmup,
        measure_steps=args.repeats,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    if not records:
        print("No measurements collected (all grid points skipped).")
        return

    # Optionally save raw CSV.
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        print(f"\nRaw measurements saved to: {csv_path}")

    # Fit affine model and save JSON.
    coefficients = _fit_affine(records)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(coefficients, f, indent=2)

    print("\nFitted coefficients:")
    for k, v in coefficients.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nProfile saved to: {output_path}")
    print(
        "\nUsage:\n"
        f"  vllm serve {args.model} \\\n"
        "      --scheduling-policy slo --slo-enable \\\n"
        "      --slo-predictor offline \\\n"
        f"      --slo-profile-path {output_path}"
    )


if __name__ == "__main__":
    main()
