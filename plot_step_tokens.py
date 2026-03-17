#!/usr/bin/env python3
"""Analyze and plot per-step prefill/decode token distribution from CSV logs.

Usage:
    # Single file analysis:
    python plot_step_tokens.py steps_compressed.csv

    # Compare baseline vs compressed:
    python plot_step_tokens.py steps_baseline.csv steps_compressed.csv

Generate CSV logs by running vLLM with:
    VLLM_LOG_STEP_TOKENS=1 VLLM_LOG_STEP_TOKENS_CSV=steps.csv python ...
"""

import argparse
import sys

import matplotlib.pyplot as plt
import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize timestamp to start from 0
    df["time_s"] = df["timestamp"] - df["timestamp"].iloc[0]
    df["step"] = range(len(df))
    df["prefill_ratio"] = df["prefill_tokens"] / df["total_tokens"].clip(lower=1)
    return df


def plot_single(df: pd.DataFrame, label: str, axes):
    """Plot analysis for a single CSV."""
    ax1, ax2, ax3 = axes

    # Plot 1: Stacked bar - prefill vs decode tokens per step
    ax1.bar(df["step"], df["decode_tokens"], label="decode tokens", color="tab:blue",
            alpha=0.8, width=1.0)
    ax1.bar(df["step"], df["prefill_tokens"], bottom=df["decode_tokens"],
            label="prefill tokens", color="tab:orange", alpha=0.8, width=1.0)
    # Mark compression steps
    compress_steps = df[df["num_compressions"] > 0]
    if not compress_steps.empty:
        ax1.scatter(compress_steps["step"],
                    compress_steps["total_tokens"] + 20,
                    marker="v", color="red", s=30, zorder=5,
                    label="compression step")
    ax1.set_ylabel("Tokens")
    ax1.set_title(f"{label}: Prefill vs Decode Tokens per Step")
    ax1.legend(loc="upper right", fontsize=8)

    # Plot 2: Prefill ratio over time
    ax2.plot(df["step"], df["prefill_ratio"], color="tab:red", alpha=0.7, linewidth=0.8)
    ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    if not compress_steps.empty:
        ax2.scatter(compress_steps["step"],
                    compress_steps["prefill_ratio"],
                    marker="v", color="red", s=30, zorder=5)
    ax2.set_ylabel("Prefill Ratio")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title(f"{label}: Prefill Token Ratio per Step")

    # Plot 3: Running queue length and request counts
    ax3.plot(df["step"], df["running_queue_len"], label="running queue",
             color="tab:green", alpha=0.7)
    ax3.plot(df["step"], df["decode_reqs"], label="decode reqs",
             color="tab:blue", alpha=0.7, linestyle="--")
    ax3.plot(df["step"], df["prefill_reqs"], label="prefill reqs",
             color="tab:orange", alpha=0.7, linestyle="--")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Count")
    ax3.set_title(f"{label}: Queue Length & Request Counts")
    ax3.legend(loc="upper right", fontsize=8)


def plot_comparison(df_baseline: pd.DataFrame, df_compressed: pd.DataFrame):
    """Side-by-side comparison of baseline vs compressed."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle("Baseline vs Compressed: Per-Step Token Distribution", fontsize=14)

    plot_single(df_baseline, "Baseline", axes[:, 0])
    plot_single(df_compressed, "Compressed", axes[:, 1])

    plt.tight_layout()
    out = "step_tokens_comparison.png"
    plt.savefig(out, dpi=150)
    print(f"Saved comparison plot to {out}")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for name, df in [("Baseline", df_baseline), ("Compressed", df_compressed)]:
        print(f"\n--- {name} ---")
        print(f"  Total steps: {len(df)}")
        print(f"  Avg prefill ratio: {df['prefill_ratio'].mean():.3f}")
        print(f"  Avg decode reqs/step: {df['decode_reqs'].mean():.1f}")
        print(f"  Avg prefill reqs/step: {df['prefill_reqs'].mean():.1f}")
        print(f"  Avg total tokens/step: {df['total_tokens'].mean():.1f}")
        print(f"  Steps with compression: {(df['num_compressions'] > 0).sum()}")
        print(f"  Avg running queue: {df['running_queue_len'].mean():.1f}")

        # Decode-only steps stats
        decode_only = df[df["prefill_reqs"] == 0]
        if not decode_only.empty:
            print(f"  Decode-only steps: {len(decode_only)} "
                  f"({100*len(decode_only)/len(df):.1f}%)")
            print(f"  Avg decode tokens (decode-only): "
                  f"{decode_only['decode_tokens'].mean():.1f}")


def plot_single_file(df: pd.DataFrame, label: str):
    """Plot analysis for a single file."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f"{label}: Per-Step Token Distribution", fontsize=14)

    plot_single(df, label, axes)

    plt.tight_layout()
    out = "step_tokens_single.png"
    plt.savefig(out, dpi=150)
    print(f"Saved plot to {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_files", nargs="+",
                        help="CSV log file(s). 1 file = single plot, "
                             "2 files = baseline vs compressed comparison")
    args = parser.parse_args()

    if len(args.csv_files) == 1:
        df = load_csv(args.csv_files[0])
        plot_single_file(df, args.csv_files[0])
    elif len(args.csv_files) == 2:
        df_baseline = load_csv(args.csv_files[0])
        df_compressed = load_csv(args.csv_files[1])
        plot_comparison(df_baseline, df_compressed)
    else:
        print("Please provide 1 or 2 CSV files.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
