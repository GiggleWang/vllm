import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Load data
files = {
    'baseline': ('/workspace/vllm/steps_baseline.csv', 'baseline (unlimited)'),
    'new_adaptive': ('/workspace/vllm/steps_compressed_new.csv', 'new_adaptive (adaptive)'),
    'old_thresh_10': ('/workspace/vllm/steps_compressed_old_10.csv', 'old_thresh_10'),
    'old_thresh_20': ('/workspace/vllm/steps_compressed_old_20.csv', 'old_thresh_20'),
}

data = {}
for key, (path, label) in files.items():
    df = pd.read_csv(path)
    df['label'] = label
    data[key] = df

keys_ordered = ['baseline', 'new_adaptive', 'old_thresh_10', 'old_thresh_20']
labels = {k: files[k][1] for k in keys_ordered}
colors_line = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

OUT = '/workspace/vllm/analysis_plots'

# ============================================================
# Plot 1: prefill_vs_decode_tokens.png
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
axes = axes.flatten()

y_max = 0
for k in keys_ordered:
    df = data[k]
    y_max = max(y_max, (df['decode_tokens'] + df['prefill_tokens']).max())

for idx, k in enumerate(keys_ordered):
    ax = axes[idx]
    df = data[k]
    steps = np.arange(len(df))
    ax.fill_between(steps, 0, df['decode_tokens'], alpha=0.7, color='#1f77b4',
                    label='decode_tokens')
    ax.fill_between(steps, df['decode_tokens'], df['decode_tokens'] + df['prefill_tokens'],
                    alpha=0.7, color='#ff7f0e', label='prefill_tokens')
    ax.set_title(f"Prefill vs Decode Tokens - {labels[k]}", fontsize=11)
    ax.set_xlabel('Step', fontsize=10)
    ax.set_ylabel('Tokens', fontsize=10)
    ax.set_ylim(0, y_max * 1.05)
    ax.legend(fontsize=8, loc='upper right')

plt.tight_layout()
fig.savefig(f'{OUT}/prefill_vs_decode_tokens.png')
plt.close(fig)
print("Plot 1 done")

# ============================================================
# Plot 2: running_queue_len.png
# ============================================================
fig, ax = plt.subplots(figsize=(14, 7), dpi=150)

for idx, k in enumerate(keys_ordered):
    df = data[k]
    t = df['timestamp'] - df['timestamp'].iloc[0]
    ax.plot(t, df['running_queue_len'], color=colors_line[idx], alpha=0.75,
            label=labels[k], linewidth=1.2)

ax.set_xlabel('Relative Time (seconds)', fontsize=12)
ax.set_ylabel('running_queue_len', fontsize=12)
ax.set_title('Running Queue Length Over Time', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f'{OUT}/running_queue_len.png')
plt.close(fig)
print("Plot 2 done")

# ============================================================
# Plot 3: decode_ratio_distribution.png
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
axes = axes.flatten()

for idx, k in enumerate(keys_ordered):
    ax = axes[idx]
    df = data[k]
    ratio = df['decode_ratio'].dropna()
    ax.hist(ratio, bins=20, range=(0, 1), color=colors_line[idx], alpha=0.7, edgecolor='black', linewidth=0.5)
    mean_val = ratio.mean()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
               label=f'mean={mean_val:.3f}')
    ax.set_title(f"decode_ratio Distribution - {labels[k]}", fontsize=11)
    ax.set_xlabel('decode_ratio', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(f'{OUT}/decode_ratio_distribution.png')
plt.close(fig)
print("Plot 3 done")

# ============================================================
# Plot 4: compression_impact.png
# ============================================================
fig, ax = plt.subplots(figsize=(14, 8), dpi=150)

PRE = 5
POST = 15

for idx, k in enumerate(keys_ordered):
    df = data[k].reset_index(drop=True)
    comp_indices = df.index[df['num_compressions'] > 0].tolist()
    if not comp_indices:
        continue

    all_windows = []
    for ci in comp_indices:
        start = ci - PRE
        end = ci + POST + 1
        if start < 0 or end > len(df):
            continue
        window = df['prefill_tokens'].iloc[start:end].values
        all_windows.append(window)

    if not all_windows:
        continue

    avg_curve = np.mean(all_windows, axis=0)
    x = np.arange(-PRE, POST + 1)
    ax.plot(x, avg_curve, color=colors_line[idx], linewidth=2, marker='o', markersize=3,
            label=f"{labels[k]} (n={len(all_windows)})")

ax.axvline(0, color='gray', linestyle='--', linewidth=2, alpha=0.7,
           label='Compression event')
ax.set_xlabel('Steps Relative to Compression', fontsize=12)
ax.set_ylabel('Avg prefill_tokens', fontsize=12)
ax.set_title('Compression Impact on Prefill Tokens', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f'{OUT}/compression_impact.png')
plt.close(fig)
print("Plot 4 done")

# ============================================================
# Plot 5: budget_floor_effect.png (new_adaptive only)
# ============================================================
df = data['new_adaptive'].reset_index(drop=True)
fig, ax1 = plt.subplots(figsize=(14, 7), dpi=150)

steps = np.arange(len(df))

ax1.plot(steps, df['budget_floor'], color='#1f77b4', linewidth=1.2, label='budget_floor')
ax1.set_xlabel('Step', fontsize=12)
ax1.set_ylabel('budget_floor', color='#1f77b4', fontsize=12)
ax1.tick_params(axis='y', labelcolor='#1f77b4')

ax2 = ax1.twinx()
ax2.bar(steps, df['prefill_tokens'], color='#ff7f0e', alpha=0.4, width=1.0, label='prefill_tokens')
ax2.set_ylabel('prefill_tokens', color='#ff7f0e', fontsize=12)
ax2.tick_params(axis='y', labelcolor='#ff7f0e')

comp_steps = df.index[df['num_compressions'] > 0].tolist()
for i, cs in enumerate(comp_steps):
    ax1.axvline(cs, color='red', linestyle='--', alpha=0.6, linewidth=1.0,
                label='Compression' if i == 0 else None)

ax1.set_title('Budget Floor Effect (new_adaptive)', fontsize=14)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper left')

ax1.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f'{OUT}/budget_floor_effect.png')
plt.close(fig)
print("Plot 5 done")

print("\nAll plots saved to /workspace/vllm/analysis_plots/")
