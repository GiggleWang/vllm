# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Analyze scheduler step CSV logs produced by --sched-log-dir.

Usage
-----
    python analyze_sched_csv.py <dir_or_csv> [<dir_or_csv> ...] [--labels A B ...]

Each positional argument can be:
  - a timestamped run directory (contains sched_steps.csv), or
  - the path to a sched_steps.csv file directly.

Example (compare compression vs baseline):
    python analyze_sched_csv.py \\
        /workspace/sched-logs/20260409_150000 \\
        /workspace/sched-logs/20260409_151500 \\
        --labels compress nocompress
"""

import argparse
import csv
import glob
import os
import statistics
from pathlib import Path


CSV_FILENAME = "sched_steps.csv"
COLUMNS = [
    "step", "timestamp",
    "running", "waiting",
    "decode_reqs", "prefill_reqs",
    "decode_toks", "prefill_toks",
    "budget_used", "free_blocks", "step_duration_ms",
]


def find_csv(path: str) -> str:
    p = Path(path)
    if p.is_file():
        return str(p)
    candidate = p / CSV_FILENAME
    if candidate.exists():
        return str(candidate)
    # try one level deeper (e.g. sched-logs/20260409_*/sched_steps.csv)
    matches = sorted(glob.glob(str(p / "*" / CSV_FILENAME)))
    if matches:
        return matches[-1]   # most recent
    raise FileNotFoundError(f"Cannot find {CSV_FILENAME} under {path}")


def load_csv(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                if k in ("timestamp", "step_duration_ms"):
                    parsed[k] = float(v)
                else:
                    parsed[k] = int(v)
            rows.append(parsed)
    return rows


def summarize(rows: list[dict], label: str) -> None:
    rows = [r for r in rows if r["budget_used"] > 0]
    if not rows:
        print(f"\n=== {label}: no data ===")
        return

    n = len(rows)
    prefill_fracs = [r["prefill_toks"] / r["budget_used"] for r in rows]
    decode_reqs   = [r["decode_reqs"]  for r in rows]
    prefill_reqs  = [r["prefill_reqs"] for r in rows]
    free_blocks   = [r["free_blocks"]  for r in rows]
    pure_decode   = sum(1 for r in rows if r["prefill_reqs"] == 0)

    # step_duration breakdown (only if column present)
    has_duration = "step_duration_ms" in rows[0]
    if has_duration:
        mixed_rows  = [r for r in rows if r["prefill_reqs"] > 0]
        pdonly_rows = [r for r in rows if r["prefill_reqs"] == 0]
        dur_mixed   = [r["step_duration_ms"] for r in mixed_rows]
        dur_pdonly  = [r["step_duration_ms"] for r in pdonly_rows]

    print(f"\n=== {label}  (n={n} steps) ===")
    print(f"  prefill token 占 budget 比例 : "
          f"mean={statistics.mean(prefill_fracs):.2%}  "
          f"median={statistics.median(prefill_fracs):.2%}")
    print(f"  每步 decode 请求数           : "
          f"mean={statistics.mean(decode_reqs):.1f}  "
          f"median={statistics.median(decode_reqs):.1f}")
    print(f"  每步 prefill 请求数          : "
          f"mean={statistics.mean(prefill_reqs):.1f}  "
          f"median={statistics.median(prefill_reqs):.1f}")
    print(f"  平均空闲 KV blocks           : "
          f"{statistics.mean(free_blocks):.0f}")
    print(f"  纯 decode 步骤占比           : "
          f"{pure_decode / n:.1%}  ({pure_decode}/{n})")
    if has_duration:
        all_dur = [r["step_duration_ms"] for r in rows]
        print(f"  step 耗时 (全部)             : "
              f"mean={statistics.mean(all_dur):.1f}ms  "
              f"median={statistics.median(all_dur):.1f}ms")
        if dur_pdonly:
            print(f"  step 耗时 (纯 decode)        : "
                  f"mean={statistics.mean(dur_pdonly):.1f}ms  "
                  f"median={statistics.median(dur_pdonly):.1f}ms  "
                  f"(n={len(dur_pdonly)})")
        if dur_mixed:
            print(f"  step 耗时 (混合 prefill)     : "
                  f"mean={statistics.mean(dur_mixed):.1f}ms  "
                  f"median={statistics.median(dur_mixed):.1f}ms  "
                  f"(n={len(dur_mixed)})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+",
                        help="Run directories or CSV file paths to analyze")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Human-readable labels for each path (optional)")
    args = parser.parse_args()

    labels = args.labels or [os.path.basename(p.rstrip("/")) for p in args.paths]
    if len(labels) != len(args.paths):
        parser.error("--labels count must match number of paths")

    for path, label in zip(args.paths, labels):
        try:
            csv_path = find_csv(path)
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
            continue
        rows = load_csv(csv_path)
        print(f"Loaded {len(rows)} rows from {csv_path}")
        summarize(rows, label)

    print()


if __name__ == "__main__":
    main()
