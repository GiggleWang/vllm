# Scheduler Step Profiling

This feature records per-step scheduling statistics to a CSV file so you can
analyze how KV cache compression (or any other configuration change) affects
the balance between prefill and decode work.

## Motivation

After enabling KV cache compression (e.g. `--kv-compression-policy streaming_llm`),
TTFT typically improves because the KV cache footprint is smaller and more
requests are admitted sooner.  However, TPOT often degrades at the same time.

The hypothesis being tested is:

```
压缩 → KV block 释放 → Scheduler 接纳更多 waiting 请求
     → 更多 prefill-in-progress 请求与 decode 请求共享 token budget
     → decode 请求每步能处理的 token 变少 → TPOT 升高
```

Per-step CSV logging makes this directly observable.

## Usage

### 1. Start vLLM with logging enabled

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B \
    --no-enable-prefix-caching \
    --kv-compression-policy streaming_llm \
    --kv-compression-ratio 0.2 \
    --kv-compression-keep-first 4 \
    --kv-compression-keep-last 4 \
    --max-num-batched-tokens 16384 \
    --sched-log-dir /workspace/sched-logs \
    --port 8000
```

A timestamped subdirectory is created automatically, e.g.:

```
/workspace/sched-logs/
  20260409_150012/
    sched_steps.csv
```

Each restart creates a fresh subdirectory, so runs are never overwritten.

When `--sched-log-dir` is **not** specified, no logging overhead is incurred.

### 2. Run your benchmark

```bash
python benchmark/test-code/lmdeploy/benchmark_test.py \
    -d benchmark/longbench-v2_3k/test-data/benchmark_3000.jsonl \
    -u http://localhost:8000/v1/completions \
    -m Qwen/Qwen3-4B -c 8 -n 800 --max-input-length 40000
```

Stop the server after the benchmark completes (Ctrl-C) — this flushes and
closes the CSV file cleanly.

Repeat the same steps for the baseline (no compression) run, using the same
`--sched-log-dir` so both runs land in sibling subdirectories.

### 3. Analyze the results

```bash
python analyze_sched_csv.py \
    /workspace/sched-logs/20260409_150012 \
    /workspace/sched-logs/20260409_152034 \
    --labels compress nocompress
```

Sample output:

```
Loaded 12483 rows from .../20260409_150012/sched_steps.csv

=== compress  (n=12483 steps) ===
  prefill token 占 budget 比例 : mean=68.41%  median=71.20%
  每步 decode 请求数           : mean=6.2  median=6.0
  每步 prefill 请求数          : mean=2.8  median=3.0
  平均空闲 KV blocks           : 1842
  纯 decode 步骤占比           : 31.2%  (3894/12483)

Loaded 11201 rows from .../20260409_152034/sched_steps.csv

=== nocompress  (n=11201 steps) ===
  prefill token 占 budget 比例 : mean=22.14%  median=0.00%
  每步 decode 请求数           : mean=7.1  median=7.0
  每步 prefill 请求数          : mean=0.4  median=0.0
  平均空闲 KV blocks           : 312
  纯 decode 步骤占比           : 71.8%  (8042/11201)
```

## CSV Schema

| Column | Type | Description |
|--------|------|-------------|
| `step` | int | Step index (1-based, monotonically increasing) |
| `timestamp` | float | `time.monotonic()` at scheduling time |
| `running` | int | Number of requests in the running queue |
| `waiting` | int | Number of requests in the waiting queue |
| `decode_reqs` | int | Running requests in decode phase this step |
| `prefill_reqs` | int | Running requests in prefill (chunked) phase this step |
| `decode_toks` | int | Tokens scheduled for decode (== `decode_reqs`) |
| `prefill_toks` | int | Tokens scheduled for prefill |
| `budget_used` | int | Total tokens scheduled (`decode_toks + prefill_toks`) |
| `free_blocks` | int | Free KV cache blocks at end of scheduling |

## Interpretation

| Metric | Hypothesis confirmed if … |
|--------|--------------------------|
| `prefill_toks / budget_used` | compression >> baseline |
| `prefill_reqs` per step | compression >> baseline |
| pure-decode step fraction | compression << baseline |
| `free_blocks` | compression >> baseline |
