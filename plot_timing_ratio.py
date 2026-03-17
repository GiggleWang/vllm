"""Compare compress vs no-compress decode timing.

Plots the ratio (compress / no-compress) of avg_ms for each module category
(attention, linear, norm) across iterations, plus total_ms difference.

Usage:
    python plot_timing_ratio.py
"""

import csv
from collections import defaultdict

import matplotlib.pyplot as plt


# --- Load CSV ---
def load_csv(path):
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
comp_rows = load_csv("compress_timing.csv")
nocomp_rows = load_csv("no_compress_timing.csv")

agg_comp_avg, agg_comp_total = aggregate(comp_rows)
agg_nocomp_avg, agg_nocomp_total = aggregate(nocomp_rows)

categories = ["attention", "linear", "norm"]

# Figure 1: avg_ms comparison + ratio (原有图)
fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))

for cat, ax in zip(categories, [axes1[0, 0], axes1[0, 1], axes1[1, 0]]):
    comp_data = agg_comp_avg.get(cat, {})
    nocomp_data = agg_nocomp_avg.get(cat, {})

    comp_iters = sorted(comp_data.keys())
    nocomp_iters = sorted(nocomp_data.keys())

    ax.plot(range(len(comp_iters)),
            [comp_data[i] for i in comp_iters],
            label="compress", alpha=0.8)
    ax.plot(range(len(nocomp_iters)),
            [nocomp_data[i] for i in nocomp_iters],
            label="no-compress", alpha=0.8)
    ax.set_title(f"{cat.upper()} - Total avg_ms (sum over all layers)")
    ax.set_xlabel("Report index")
    ax.set_ylabel("avg_ms")
    ax.legend()
    ax.grid(True, alpha=0.3)

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
ax.set_title("Ratio (compress / no-compress)")
ax.set_xlabel("Report index")
ax.set_ylabel("Ratio")
ax.legend()
ax.grid(True, alpha=0.3)

fig1.suptitle("Decode Timing: Compress vs No-Compress (avg_ms)", fontsize=14)
plt.tight_layout()
plt.savefig("timing_comparison.png", dpi=150, bbox_inches="tight")
print("Saved to timing_comparison.png")

# Figure 2: total_ms per step interval (step[n] -> step[n+1])
# Each bar = time consumed between two consecutive reports
def step_increments(data):
    """Given {iteration: total_ms}, return (labels, incremental ms per interval)."""
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
    ax.set_title(f"{cat.upper()} - ms consumed per interval")
    ax.set_xlabel("Step interval")
    ax.set_ylabel("ms")
    ax.set_xticks(list(x))
    ax.set_xticklabels(comp_labels[:n], rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

# Bottom-right: ratio compress/no-compress per interval
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
ax.set_title("Ratio (compress / no-compress) per interval")
ax.set_xlabel("Interval index")
ax.set_ylabel("Ratio")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig2.suptitle("total_ms consumed per step interval (compress vs no-compress)",
              fontsize=13)
plt.tight_layout()
plt.savefig("timing_diff.png", dpi=150, bbox_inches="tight")
print("Saved to timing_diff.png")
