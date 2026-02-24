```
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B \
    --kv-compression-policy knorm \
    --kv-compression-ratio 0.5 \
    --kv-compression-keep-first 4 \
    --kv-compression-keep-last 4 \
    --port 8000 --host 0.0.0.0
```

```
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B \
    --kv-compression-policy streaming_llm \
    --kv-compression-ratio 0.5 \
    --kv-compression-keep-first 4 \
    --kv-compression-keep-last 4 \
    --port 8000 --host 0.0.0.0
```

```
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B \
    --kv-compression-policy expected_attention \
    --kv-compression-ratio 0.5 \
    --kv-compression-keep-first 4 \
    --kv-compression-keep-last 4 \
    --port 8000 --host 0.0.0.0
```

```
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B \
    --port 8000 --host 0.0.0.0
```