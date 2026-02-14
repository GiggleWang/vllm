# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""全面测试vLLM支持的所有KV cache压缩策略。

此脚本测试以下6种压缩策略：
1. streaming_llm - 保留首尾token（attention sinks）
2. knorm - 基于key向量L2范数选择重要token
3. expected_attention - 基于预期注意力权重估计选择token
4. think - 基于key通道重要性选择token
5. keydiff - 保留与邻居差异最大的token
6. lagkv - 基于滞后key相关性选择token

运行方式:
    python all_policies.py
"""

import gc
from typing import Dict, List

import torch

from vllm import LLM, SamplingParams

# 长提示词（约300个token）用于测试压缩效果
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

# 短提示词用于验证批处理
SHORT_PROMPT = "What is the capital of France?"

PROMPTS = [LONG_PROMPT, SHORT_PROMPT]

# 所有支持的压缩策略
COMPRESSION_POLICIES = [
    "streaming_llm",
    "knorm",
    "expected_attention",
    "think",
    "keydiff",
    "lagkv",
]


def run_without_compression(
    model_name: str = "facebook/opt-125m",
    gpu_mem_util: float = 0.9
) -> List:
    """运行无压缩的基准测试。"""
    print("=" * 80)
    print("基准测试: 无KV cache压缩")
    print("=" * 80)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=100)

    llm = LLM(
        model=model_name,
        enforce_eager=True,
        gpu_memory_utilization=gpu_mem_util,
    )
    outputs = llm.generate(PROMPTS, sampling_params)

    for i, output in enumerate(outputs):
        prompt = output.prompt[:80] + "..." if len(output.prompt) > 80 else output.prompt
        generated_text = output.outputs[0].text
        print(f"\n提示词 {i}: {prompt!r}")
        print(f"输出: {generated_text[:150]!r}...")
        print("-" * 80)

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print("\n[已清理GPU内存]\n")

    return outputs


def run_with_compression(
    policy_name: str,
    model_name: str = "facebook/opt-125m",
    gpu_mem_util: float = 0.9,
    compression_ratio: float = 0.5,
    keep_first: int = 0,
    keep_last: int = 0,
) -> List:
    """使用指定的压缩策略运行测试。"""
    print("=" * 80)
    print(f"压缩策略: {policy_name}")
    print(f"  - 保留比例: {compression_ratio}")
    print(f"  - 保留首部token数: {keep_first}")
    print(f"  - 保留尾部token数: {keep_last}")
    print("=" * 80)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=100)

    llm = LLM(
        model=model_name,
        enforce_eager=True,
        gpu_memory_utilization=gpu_mem_util,
        kv_compression_policy=policy_name,
        kv_compression_ratio=compression_ratio,
        kv_compression_keep_first=keep_first,
        kv_compression_keep_last=keep_last,
    )
    outputs = llm.generate(PROMPTS, sampling_params)

    for i, output in enumerate(outputs):
        prompt = output.prompt[:80] + "..." if len(output.prompt) > 80 else output.prompt
        generated_text = output.outputs[0].text
        print(f"\n提示词 {i}: {prompt!r}")
        print(f"输出: {generated_text[:150]!r}...")
        print("-" * 80)

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print("\n[已清理GPU内存]\n")

    return outputs


def compare_outputs(baseline_outputs: List, compressed_outputs_dict: Dict[str, List]):
    """比较基准输出和各压缩策略的输出。"""
    print("\n" + "=" * 80)
    print("输出对比分析")
    print("=" * 80)

    for policy_name, compressed_outputs in compressed_outputs_dict.items():
        print(f"\n策略: {policy_name}")
        print("-" * 80)

        for i, (baseline, compressed) in enumerate(zip(baseline_outputs, compressed_outputs)):
            baseline_text = baseline.outputs[0].text
            compressed_text = compressed.outputs[0].text

            # 判断是否完全匹配
            if baseline_text == compressed_text:
                status = "✓ 完全匹配"
            else:
                # 计算相似度（简单字符匹配）
                min_len = min(len(baseline_text), len(compressed_text))
                matching_chars = sum(
                    1 for j in range(min_len)
                    if baseline_text[j] == compressed_text[j]
                )
                similarity = matching_chars / max(len(baseline_text), len(compressed_text)) * 100
                status = f"✗ 不同 (相似度: {similarity:.1f}%)"

            print(f"  提示词 {i}: {status}")

            if baseline_text != compressed_text:
                print(f"    基准输出:   {baseline_text[:80]!r}...")
                print(f"    压缩输出:   {compressed_text[:80]!r}...")


def test_different_compression_ratios(
    policy_name: str = "knorm",
    model_name: str = "facebook/opt-125m",
    ratios: List[float] = [0.3, 0.5, 0.7, 0.9]
):
    """测试同一策略在不同压缩比下的表现。"""
    print("\n" + "=" * 80)
    print(f"测试不同压缩比 - 策略: {policy_name}")
    print("=" * 80)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=100)
    gpu_mem_util = 0.9

    results = {}

    for ratio in ratios:
        print(f"\n压缩比: {ratio}")
        print("-" * 80)

        llm = LLM(
            model=model_name,
            enforce_eager=True,
            gpu_memory_utilization=gpu_mem_util,
            kv_compression_policy=policy_name,
            kv_compression_ratio=ratio,
            kv_compression_keep_first=4,
            kv_compression_keep_last=64,
        )
        outputs = llm.generate([LONG_PROMPT], sampling_params)

        generated_text = outputs[0].outputs[0].text
        print(f"输出: {generated_text[:150]!r}...")

        results[ratio] = generated_text

        del llm
        gc.collect()
        torch.cuda.empty_cache()

    # 对比不同压缩比的输出
    print("\n" + "=" * 80)
    print("不同压缩比输出对比")
    print("=" * 80)

    ratio_list = sorted(results.keys())
    for i in range(len(ratio_list) - 1):
        ratio1, ratio2 = ratio_list[i], ratio_list[i + 1]
        text1, text2 = results[ratio1], results[ratio2]

        match = "匹配" if text1 == text2 else "不同"
        print(f"压缩比 {ratio1} vs {ratio2}: {match}")


def main():
    """主测试函数。"""
    model_name = "facebook/opt-125m"
    gpu_mem_util = 0.9

    # 1. 运行基准测试（无压缩）
    baseline_outputs = run_without_compression(model_name, gpu_mem_util)

    # 2. 测试所有压缩策略
    compressed_outputs_dict = {}

    for policy in COMPRESSION_POLICIES:
        try:
            outputs = run_with_compression(
                policy_name=policy,
                model_name=model_name,
                gpu_mem_util=gpu_mem_util,
                compression_ratio=0.5,
                keep_first=0,
                keep_last=0,
            )
            compressed_outputs_dict[policy] = outputs
        except Exception as e:
            print(f"错误: 策略 '{policy}' 测试失败: {e}\n")

    # 3. 对比所有策略的输出
    if compressed_outputs_dict:
        compare_outputs(baseline_outputs, compressed_outputs_dict)

    # 4. 测试不同压缩比（使用knorm策略）
    print("\n" + "=" * 80)
    print("附加测试: 不同压缩比的影响")
    print("=" * 80)
    test_different_compression_ratios(
        policy_name="knorm",
        model_name=model_name,
        ratios=[0.3, 0.5, 0.7, 0.9]
    )

    print("\n" + "=" * 80)
    print("所有测试完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
