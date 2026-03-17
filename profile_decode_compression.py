# SPDX-License-Identifier: Apache-2.0
"""Profile decode-phase operator execution time: KV compression vs no compression.

Uses vLLM's built-in torch.profiler integration to compare CUDA kernel timings
between baseline (no compression) and streaming_llm KV cache compression.

Usage:
    python profile_decode_compression.py

Output:
    - ./profile_baseline/   : TensorBoard trace files (no compression)
    - ./profile_compressed/ : TensorBoard trace files (with compression)

View results:
    tensorboard --logdir=./profile_baseline
    tensorboard --logdir=./profile_compressed
    # Or open the .json trace files in chrome://tracing
"""

import gc
import time

import torch

from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# Target ~34000 tokens. A rough heuristic: 1 token ≈ 4 characters for English.
# We build a long prompt by repeating paragraphs, then append a question.
# ---------------------------------------------------------------------------
TARGET_TOKENS = 34000
CHARS_PER_TOKEN = 4  # conservative estimate

_PARAGRAPH_BLOCK = (
    "The field of artificial intelligence was founded in 1956 at a conference "
    "at Dartmouth College, where researchers gathered to discuss the "
    "possibility of creating machines that could think. Early pioneers "
    "included John McCarthy, who coined the term 'artificial intelligence', "
    "Marvin Minsky, Allen Newell, and Herbert Simon. In the early years, "
    "researchers were optimistic that machines would soon be able to perform "
    "any intellectual task that a human could. They developed programs that "
    "could prove mathematical theorems, play checkers, and solve algebra "
    "problems. However, progress was slower than expected, and by the 1970s, "
    "funding for AI research was cut dramatically in what became known as the "
    "'AI winter'. During this period, many researchers left the field and "
    "public interest waned. The field experienced a revival in the 1980s "
    "with the development of expert systems, which were programs designed "
    "to mimic the decision-making abilities of human experts. Companies "
    "invested heavily in these systems, but they proved brittle and "
    "difficult to maintain. A second AI winter followed in the late 1980s "
    "and early 1990s. The modern era of AI began in the late 1990s and "
    "early 2000s, driven by increases in computing power, the availability "
    "of large datasets, and advances in machine learning algorithms. "
    "Deep learning, a subset of machine learning based on neural networks "
    "with many layers, achieved breakthroughs in image recognition, speech "
    "recognition, and natural language processing. In 2012, a deep learning "
    "model called AlexNet won the ImageNet competition by a large margin, "
    "sparking a revolution in computer vision. Since then, AI has been "
    "applied to a wide range of problems, from autonomous vehicles to drug "
    "discovery to language translation. Large language models such as GPT "
    "and BERT have transformed natural language processing, enabling machines "
    "to generate human-like text, answer questions, and summarize documents. "
    "Today, AI is one of the most active areas of research in computer "
    "science, with applications in nearly every industry.\n\n"
)

_SUFFIX = (
    "Based on all of the above, what are the three most important "
    "milestones in AI history? Explain each briefly:"
)


def _build_long_prompt(target_tokens: int = TARGET_TOKENS) -> str:
    """Build a prompt of approximately `target_tokens` tokens."""
    target_chars = target_tokens * CHARS_PER_TOKEN
    block_len = len(_PARAGRAPH_BLOCK)
    repeats = max(1, target_chars // block_len)

    header = "Below is a very detailed and repeated summary of AI history.\n\n"
    body = _PARAGRAPH_BLOCK * repeats
    prompt = header + body + _SUFFIX

    # Trim to approximate target length
    max_chars = target_chars + len(_SUFFIX) + len(header)
    if len(prompt) > max_chars:
        prompt = prompt[:max_chars]

    return prompt


LONG_PROMPT = _build_long_prompt()

MODEL = "Qwen/Qwen3-4B"
SAMPLING_PARAMS = SamplingParams(temperature=0.0, max_tokens=200)


def run_profiling(
    label: str,
    trace_dir: str,
    compression_policy: str | None = None,
    compression_ratio: float = 0.5,
):
    """Run a single profiling round."""
    print("=" * 70)
    print(f"Profiling: {label}")
    print(f"Trace dir: {trace_dir}")
    print("=" * 70)

    kwargs = dict(
        model=MODEL,
        enforce_eager=True,
        gpu_memory_utilization=0.9,
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": trace_dir,
        },
    )

    if compression_policy is not None:
        kwargs["kv_compression_policy"] = compression_policy
        kwargs["kv_compression_ratio"] = compression_ratio
        print(f"KV compression: {compression_policy}, ratio={compression_ratio}")
    else:
        print("KV compression: disabled")

    llm = LLM(**kwargs)

    # Start profiling, run inference, stop profiling
    llm.start_profile()
    outputs = llm.generate([LONG_PROMPT], SAMPLING_PARAMS)
    llm.stop_profile()

    # Print generated text
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"\nGenerated ({len(generated_text)} chars): {generated_text[:200]}...")
    print()

    # Cleanup GPU memory for next round
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    # Wait for profiler background writes to complete
    time.sleep(5)


def main():
    # Round 1: Baseline (no compression)
    run_profiling(
        label="Baseline (no KV compression)",
        trace_dir="./profile_baseline",
    )

    # Round 2: With streaming_llm compression
    run_profiling(
        label="With KV compression (streaming_llm, ratio=0.5)",
        trace_dir="./profile_compressed",
        compression_policy="streaming_llm",
        compression_ratio=0.5,
    )

    print("=" * 70)
    print("Profiling complete!")
    print()
    print("View results:")
    print("  tensorboard --logdir=./profile_baseline")
    print("  tensorboard --logdir=./profile_compressed")
    print()
    print("Or open the .json trace files in chrome://tracing")
    print("=" * 70)


if __name__ == "__main__":
    main()
