# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test script for KV cache compression.

Runs offline inference with a long prompt to exercise the KV cache
compression path (prefill -> compress -> decode).  The scheduler
compresses the KV cache between prefill and decode, freeing GPU
memory blocks while (ideally) preserving generation quality.

Usage:
    python kv_compression_test.py
"""

import gc

import torch

from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# Long prompt (~300 tokens) so that compression is meaningful.
# With keep_ratio=0.5 the scheduler will try to halve the KV cache after
# prefill, freeing roughly half the blocks.
# ---------------------------------------------------------------------------
LONG_PROMPT = (
    "Below is a detailed summary of the history of artificial intelligence.\n\n"
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
    "Based on the summary above, what are the three most important "
    "milestones in AI history? Explain each briefly:"
)

# A second shorter prompt to verify batching still works.
SHORT_PROMPT = "What is the capital of France?"

PROMPTS = [LONG_PROMPT, SHORT_PROMPT]


def main():
    sampling_params = SamplingParams(temperature=0.0, max_tokens=100)

    # Use lower GPU memory utilization to allow multiple LLM instances
    # in sequence without running out of memory.
    gpu_mem_util = 0.9  # Only use 90% of GPU memory

    # --- Run WITHOUT compression (baseline) ---
    print("=" * 70)
    print("Run 1: WITHOUT KV cache compression (baseline)")
    print("=" * 70)

    llm_baseline = LLM(
        model="facebook/opt-125m",
        enforce_eager=True,
        gpu_memory_utilization=gpu_mem_util,
    )
    outputs_baseline = llm_baseline.generate(PROMPTS, sampling_params)

    for output in outputs_baseline:
        prompt = output.prompt[:80] + "..." if len(output.prompt) > 80 else output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:  {prompt!r}")
        print(f"Output:  {generated_text!r}")
        print("-" * 70)

    del llm_baseline
    # Force garbage collection and CUDA cache cleanup
    gc.collect()
    torch.cuda.empty_cache()
    print("\n[Cleaned up GPU memory]\n")

    # --- Run WITH compression ---
    print("=" * 70)
    print("Run 2: WITH KV cache compression (streaming_llm)")
    print("=" * 70)

    llm_compressed = LLM(
        model="facebook/opt-125m",
        enforce_eager=True,
        gpu_memory_utilization=gpu_mem_util,
        kv_compression_policy="streaming_llm",
        kv_compression_ratio=0.5,
        kv_compression_keep_first=0,
        kv_compression_keep_last=0,
    )
    outputs_compressed = llm_compressed.generate(PROMPTS, sampling_params)

    for output in outputs_compressed:
        prompt = output.prompt[:80] + "..." if len(output.prompt) > 80 else output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:  {prompt!r}")
        print(f"Output:  {generated_text!r}")
        print("-" * 70)

    del llm_compressed
    gc.collect()
    torch.cuda.empty_cache()

    # --- Compare ---
    print("\n" + "=" * 70)
    print("Comparison")
    print("=" * 70)
    for i, (b, c) in enumerate(zip(outputs_baseline, outputs_compressed)):
        b_text = b.outputs[0].text
        c_text = c.outputs[0].text
        match = "MATCH" if b_text == c_text else "DIFFER"
        print(f"Prompt {i}: [{match}]")
        if b_text != c_text:
            print(f"  Baseline:   {b_text[:120]!r}")
            print(f"  Compressed: {c_text[:120]!r}")


if __name__ == "__main__":
    main()
