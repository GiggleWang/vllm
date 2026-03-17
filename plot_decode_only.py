"""Compare compress vs no-compress timing for PURE DECODE iterations only.

Filters out any iteration that contains prefill requests (num_prefill_reqs > 0),
so the plotted timing data reflects decode-only workload.

Usage:
    python plot_decode_only.py
"""

import csv
from collections import defaultdict

import matplotlib.pyplot as plt


# --- Load timing CSV ---
def load_timing_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "iteration": int(row["iteration"]),
                "module": row["module"],
                "avg_ms": float(row["avg_ms"]),
                "total_ms": float(row["total_ms"]),
            })
    return rows


# --- Load metadata CSV and find pure-decode iterations ---
def load_pure_decode_iterations(path):
    """Return set of iterations where every request is decode (no prefill)."""
    all_iters = set()
    has_prefill = set()
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            it = int(row["iteration"])
            all_iters.add(it)
            # If any request in this iteration is not decode, mark it
            if row["is_decode"] != "True":
                has_prefill.add(it)
    return all_iters - has_prefill


# --- Load metadata for decode batch size info ---
def load_decode_batch_info(path, pure_decode_iters):
    """Return {iteration: num_decode_reqs} for pure decode iterations."""
    info = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            it = int(row["iteration"])
            if it in pure_decode_iters:
                info[it] = int(row["num_decode_reqs"])
    return info


# --- Classify module name ---
def classify_module(name: str) -> str:
    parts = name.split(".")
    if "self_attn" in parts:
        idx = parts.index("self_attn")
        suffix = ".".join(parts[idx:])
    elif "mlp" in parts:
        idx = parts.index("mlp")
        suffix = ".".join(parts[idx:])
    else:
        suffix = parts[-1]

    if suffix == "self_attn.attn":
        return "attention"
    elif suffix in ("self_attn.qkv_proj", "self_attn.o_proj",
                     "mlp.gate_up_proj", "mlp.down_proj"):
        return "linear"
    elif suffix in ("input_layernorm", "post_attention_layernorm",
                     "self_attn.q_norm", "self_attn.k_norm", "model.norm"):
        return "norm"
    return "other"


# --- Filter timing rows to pure-decode iterations ---
def filter_timing(rows, pure_decode_iters):
    return [r for r in rows if r["iteration"] in pure_decode_iters]


# --- Aggregate: sum avg_ms and total_ms by (iteration, category) ---
def aggregate(rows):
    agg_avg = defaultdict(float)
    agg_total = defaultdict(float)
    for r in rows:
        cat = classify_module(r["module"])
        agg_avg[(r["iteration"], cat)] += r["avg_ms"]
        agg_total[(r["iteration"], cat)] += r["total_ms"]
    result_avg = defaultdict(dict)
    result_total = defaultdict(dict)
    for (it, cat), val in agg_avg.items():
        result_avg[cat][it] = val
    for (it, cat), val in agg_total.items():
        result_total[cat][it] = val
    return result_avg, result_total


# --- Main ---
print("Loading metadata to find pure-decode iterations...")
comp_pure = load_pure_decode_iterations("compress_metadata.csv")
nocomp_pure = load_pure_decode_iterations("no_compress_metadata.csv")
print(f"  compress:    {len(comp_pure)} pure-decode iterations")
print(f"  no-compress: {len(nocomp_pure)} pure-decode iterations")

comp_rows = filter_timing(load_timing_csv("compress_timing.csv"), comp_pure)
nocomp_rows = filter_timing(load_timing_csv("no_compress_timing.csv"), nocomp_pure)
print(f"  compress timing rows after filter:    {len(comp_rows)}")
print(f"  no-compress timing rows after filter: {len(nocomp_rows)}")

agg_comp_avg, agg_comp_total = aggregate(comp_rows)
agg_nocomp_avg, agg_nocomp_total = aggregate(nocomp_rows)

categories = ["attention", "linear", "norm"]

# ============================================================
# Figure 1: avg_ms comparison for pure-decode iterations
# ============================================================
fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))

for cat, ax in zip(categories, [axes1[0, 0], axes1[0, 1], axes1[1, 0]]):
    comp_data = agg_comp_avg.get(cat, {})
    nocomp_data = agg_nocomp_avg.get(cat, {})

    # Use sequential index (report index) for x-axis
    comp_iters = sorted(comp_data.keys())
    nocomp_iters = sorted(nocomp_data.keys())

    ax.plot(range(len(comp_iters)),
            [comp_data[i] for i in comp_iters],
            label="compress", alpha=0.8)
    ax.plot(range(len(nocomp_iters)),
            [nocomp_data[i] for i in nocomp_iters],
            label="no-compress", alpha=0.8)
    ax.set_title(f"{cat.upper()} - avg_ms (decode only)")
    ax.set_xlabel("Decode-only report index")
    ax.set_ylabel("avg_ms")
    ax.legend()
    ax.grid(True, alpha=0.3)

# Bottom-right: ratio
ax = axes1[1, 1]
for cat in categories:
    comp_data = agg_comp_avg.get(cat, {})
    nocomp_data = agg_nocomp_avg.get(cat, {})

    comp_vals = [comp_data[i] for i in sorted(comp_data.keys())]
    nocomp_vals = [nocomp_data[i] for i in sorted(nocomp_data.keys())]

    n = min(len(comp_vals), len(nocomp_vals))
    ratios = [comp_vals[i] / nocomp_vals[i]
              for i in range(n) if nocomp_vals[i] > 0]
    ax.plot(range(len(ratios)), ratios, label=cat, alpha=0.8)

ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="ratio=1")
ax.set_title("Ratio (compress / no-compress) decode only")
ax.set_xlabel("Report index")
ax.set_ylabel("Ratio")
ax.legend()
ax.grid(True, alpha=0.3)

fig1.suptitle("Decode-Only Timing: Compress vs No-Compress (avg_ms)",
              fontsize=14)
plt.tight_layout()
plt.savefig("decode_only_comparison.png", dpi=150, bbox_inches="tight")
print("Saved to decode_only_comparison.png")

# ============================================================
# Figure 2: total_ms increments per decode-only interval
# ============================================================
def step_increments(data):
    """Given {iteration: total_ms}, return (labels, incremental ms)."""
    iters = sorted(data.keys())
    labels = [f"{iters[i]}->{iters[i+1]}" for i in range(len(iters) - 1)]
    values = [data[iters[i+1]] - data[iters[i]] for i in range(len(iters) - 1)]
    return labels, values


fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))

for cat, ax in zip(categories, [axes2[0, 0], axes2[0, 1], axes2[1, 0]]):
    comp_data = agg_comp_total.get(cat, {})
    nocomp_data = agg_nocomp_total.get(cat, {})

    comp_labels, comp_inc = step_increments(comp_data)
    nocomp_labels, nocomp_inc = step_increments(nocomp_data)

    n = min(len(comp_inc), len(nocomp_inc))
    x = range(n)
    width = 0.35
    ax.bar([i - width/2 for i in x], comp_inc[:n], width,
           label="compress", alpha=0.8)
    ax.bar([i + width/2 for i in x], nocomp_inc[:n], width,
           label="no-compress", alpha=0.8)
    ax.set_title(f"{cat.upper()} - ms per interval (decode only)")
    ax.set_xlabel("Step interval")
    ax.set_ylabel("ms")
    if n <= 20:
        ax.set_xticks(list(x))
        ax.set_xticklabels(comp_labels[:n], rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

# Bottom-right: ratio per interval
ax = axes2[1, 1]
for cat in categories:
    comp_data = agg_comp_total.get(cat, {})
    nocomp_data = agg_nocomp_total.get(cat, {})

    _, comp_inc = step_increments(comp_data)
    _, nocomp_inc = step_increments(nocomp_data)

    n = min(len(comp_inc), len(nocomp_inc))
    ratios = [comp_inc[i] / nocomp_inc[i]
              for i in range(n) if nocomp_inc[i] > 0]
    ax.plot(range(len(ratios)), ratios, marker="o", label=cat, alpha=0.8)

ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="ratio=1")
ax.set_title("Ratio (compress / no-compress) per interval (decode only)")
ax.set_xlabel("Interval index")
ax.set_ylabel("Ratio")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig2.suptitle("Decode-Only: total_ms per step interval (compress vs no-compress)",
              fontsize=13)
plt.tight_layout()
plt.savefig("decode_only_diff.png", dpi=150, bbox_inches="tight")
print("Saved to decode_only_diff.png")

# ============================================================
# Figure 3: Decode batch size over time (how many reqs per iteration)
# ============================================================
comp_batch = load_decode_batch_info("compress_metadata.csv", comp_pure)
nocomp_batch = load_decode_batch_info("no_compress_metadata.csv", nocomp_pure)

fig3, ax3 = plt.subplots(figsize=(12, 5))
comp_batch_iters = sorted(comp_batch.keys())
nocomp_batch_iters = sorted(nocomp_batch.keys())

ax3.plot(range(len(comp_batch_iters)),
         [comp_batch[i] for i in comp_batch_iters],
         label="compress", alpha=0.8)
ax3.plot(range(len(nocomp_batch_iters)),
         [nocomp_batch[i] for i in nocomp_batch_iters],
         label="no-compress", alpha=0.8)
ax3.set_title("Decode Batch Size (num_decode_reqs) per Pure-Decode Iteration")
ax3.set_xlabel("Decode-only report index")
ax3.set_ylabel("num_decode_reqs")
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("decode_only_batch_size.png", dpi=150, bbox_inches="tight")
print("Saved to decode_only_batch_size.png")
