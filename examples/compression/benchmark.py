# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache压缩策略性能基准测试。

此脚本测量各压缩策略的：
1. 生成速度（tokens/秒）
2. 内存使用情况
3. 输出质量（与基准的相似度）

运行方式:
    python benchmark.py
"""

import gc
import time
from typing import Dict, List, Tuple

import torch

from vllm import LLM, SamplingParams

# 用于性能测试的提示词 - 不同长度以测试压缩在各场景下的表现
BENCHMARK_PROMPTS = [
    # 短提示词 (~50 tokens)
    "Explain the theory of relativity in simple terms. Include both "
    "special and general relativity, and discuss how they changed our "
    "understanding of space and time:",
    # 中等长度提示词 (~150 tokens)
    "Machine learning and deep learning are both subfields of "
    "artificial intelligence, but they have important differences. "
    "Machine learning involves algorithms that can learn patterns from "
    "data without being explicitly programmed. It includes techniques "
    "like decision trees, random forests, and support vector machines. "
    "Deep learning, on the other hand, is a subset of machine learning "
    "that uses neural networks with multiple layers (hence 'deep') to "
    "learn hierarchical representations of data. What are the key "
    "differences between these two approaches, and when should each "
    "be used?",
    # 长提示词 (~300 tokens)
    "The water cycle, also known as the hydrological cycle, is a "
    "continuous process by which water circulates between Earth's "
    "oceans, atmosphere, and land. It involves several key processes: "
    "evaporation, where water from oceans, lakes, and rivers turns "
    "into vapor; transpiration, where plants release water vapor; "
    "condensation, where water vapor forms clouds; precipitation, "
    "where water falls as rain or snow; and collection, where water "
    "gathers in bodies of water. This cycle is crucial for maintaining "
    "Earth's ecosystems because it distributes fresh water across the "
    "planet, regulates temperature, supports plant growth, and enables "
    "life as we know it. Climate change is affecting the water cycle, "
    "leading to more extreme weather events, changing precipitation "
    "patterns, and altering the availability of fresh water in many "
    "regions. The cycle also plays a vital role in nutrient "
    "distribution, soil formation, and maintaining atmospheric "
    "composition. Describe the water cycle in detail and explain its "
    "importance:",
    # 超长提示词 (~500 tokens)
    "Blockchain technology is a revolutionary distributed ledger system "
    "that has transformed how we think about data storage, "
    "transactions, and trust in digital systems. At its core, a "
    "blockchain is a chain of blocks, where each block contains a list "
    "of transactions, a timestamp, and a cryptographic hash of the "
    "previous block. This structure makes it extremely difficult to "
    "alter historical records because changing one block would require "
    "changing all subsequent blocks. The technology was first "
    "introduced in 2008 by an anonymous person or group known as "
    "Satoshi Nakamoto as the underlying technology for Bitcoin, the "
    "first cryptocurrency. However, blockchain's applications extend "
    "far beyond cryptocurrency. The key innovation of blockchain is "
    "that it enables trustless transactions between parties who don't "
    "know or trust each other, without requiring a central authority. "
    "This is achieved through consensus mechanisms like Proof of Work "
    "or Proof of Stake, where network participants (miners or "
    "validators) verify and validate transactions. Each participant "
    "maintains a copy of the entire ledger, making the system highly "
    "resilient to failure or attack. Smart contracts, self-executing "
    "contracts with the terms directly written into code, run on "
    "blockchain platforms like Ethereum and enable complex automated "
    "transactions. Blockchain technology has potential applications in "
    "supply chain management, healthcare records, voting systems, "
    "intellectual property rights, and financial services. However, it "
    "also faces challenges including scalability issues, high energy "
    "consumption (especially for Proof of Work systems), regulatory "
    "uncertainty, and the need for widespread adoption to realize its "
    "full potential. How does blockchain technology work, and what are "
    "its main concepts, advantages, and limitations?",
]

# 所有支持的压缩策略
COMPRESSION_POLICIES = [
    "streaming_llm",
    "knorm",
    "expected_attention",
    "think",
    "keydiff",
    "lagkv",
]


class BenchmarkResult:
    """基准测试结果。"""

    def __init__(self, policy_name: str):
        self.policy_name = policy_name
        self.total_time = 0.0
        self.total_tokens = 0
        self.outputs = []
        self.gpu_memory_allocated = 0.0
        self.gpu_memory_reserved = 0.0

    @property
    def tokens_per_second(self) -> float:
        """计算tokens/秒。"""
        if self.total_time > 0:
            return self.total_tokens / self.total_time
        return 0.0

    def __str__(self):
        return (
            f"策略: {self.policy_name}\n"
            f"  总时间: {self.total_time:.2f}秒\n"
            f"  总tokens: {self.total_tokens}\n"
            f"  速度: {self.tokens_per_second:.2f} tokens/秒\n"
            f"  GPU内存 (已分配): {self.gpu_memory_allocated:.2f} MB\n"
            f"  GPU内存 (已保留): {self.gpu_memory_reserved:.2f} MB"
        )


def benchmark_policy(
    policy_name: str | None,
    model_name: str = "facebook/opt-125m",
    prompts: List[str] = None,
    max_tokens: int = 100,
    compression_ratio: float = 0.5,
) -> BenchmarkResult:
    """对单个策略进行基准测试。"""

    if prompts is None:
        prompts = BENCHMARK_PROMPTS

    result = BenchmarkResult(policy_name or "baseline")

    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    # 创建LLM配置
    llm_kwargs = {
        "model": model_name,
        "enforce_eager": True,
        "gpu_memory_utilization": 0.9,
    }

    if policy_name is not None:
        llm_kwargs.update(
            {
                "kv_compression_policy": policy_name,
                "kv_compression_ratio": compression_ratio,
                "kv_compression_keep_first": 4,
                "kv_compression_keep_last": 64,
            }
        )

    # 创建LLM并生成
    llm = LLM(**llm_kwargs)

    # 预热
    _ = llm.generate([prompts[0]], sampling_params)

    # 重置GPU内存统计
    torch.cuda.reset_peak_memory_stats()

    # 开始基准测试
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()

    # 记录结果
    result.total_time = end_time - start_time
    result.outputs = outputs
    result.total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    result.gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
    result.gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**2  # MB

    # 清理
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return result


def calculate_similarity(text1: str, text2: str) -> float:
    """计算两个文本的相似度（简单字符匹配）。"""
    if not text1 or not text2:
        return 0.0

    min_len = min(len(text1), len(text2))
    matching_chars = sum(1 for i in range(min_len) if text1[i] == text2[i])
    return matching_chars / max(len(text1), len(text2)) * 100


def compare_quality(
    baseline_result: BenchmarkResult, compressed_results: Dict[str, BenchmarkResult]
):
    """比较输出质量。"""
    print("\n" + "=" * 80)
    print("输出质量对比（与基准的相似度）")
    print("=" * 80)

    for policy_name, result in compressed_results.items():
        similarities = []

        for baseline_out, compressed_out in zip(
            baseline_result.outputs, result.outputs
        ):
            baseline_text = baseline_out.outputs[0].text
            compressed_text = compressed_out.outputs[0].text

            similarity = calculate_similarity(baseline_text, compressed_text)
            similarities.append(similarity)

        avg_similarity = sum(similarities) / len(similarities)
        print(f"\n策略: {policy_name}")
        print(f"  平均相似度: {avg_similarity:.1f}%")
        print(f"  各提示词相似度: {[f'{s:.1f}%' for s in similarities]}")


def print_benchmark_summary(
    baseline_result: BenchmarkResult, compressed_results: Dict[str, BenchmarkResult]
):
    """打印基准测试摘要。"""
    print("\n" + "=" * 80)
    print("性能基准测试摘要")
    print("=" * 80)

    # 打印基准结果
    print(f"\n{baseline_result}")

    # 打印压缩策略结果
    for policy_name, result in sorted(compressed_results.items()):
        print(f"\n{result}")

        # 计算相对性能
        print(f"  相对基准:")

        if baseline_result.tokens_per_second > 0:
            speedup = (
                result.tokens_per_second / baseline_result.tokens_per_second
            )
            print(f"    速度倍数: {speedup:.2f}x")
        else:
            print("    速度倍数: N/A")

        if baseline_result.gpu_memory_allocated > 0:
            memory_reduction = (
                1 - result.gpu_memory_allocated /
                baseline_result.gpu_memory_allocated
            ) * 100
            print(f"    内存减少: {memory_reduction:.1f}%")
        else:
            print("    内存减少: N/A")


def main():
    """主函数。"""
    model_name = "facebook/opt-125m"
    compression_ratio = 0.5

    print("=" * 80)
    print("KV Cache压缩策略性能基准测试")
    print("=" * 80)
    print(f"模型: {model_name}")
    print(f"压缩比: {compression_ratio}")
    print(f"测试提示词数: {len(BENCHMARK_PROMPTS)}")
    print("=" * 80)

    # 1. 运行基准测试（无压缩）
    print("\n正在运行基准测试（无压缩）...")
    baseline_result = benchmark_policy(
        policy_name=None,
        model_name=model_name,
        prompts=BENCHMARK_PROMPTS,
        compression_ratio=compression_ratio,
    )

    # 2. 测试所有压缩策略
    compressed_results = {}

    for policy in COMPRESSION_POLICIES:
        print(f"\n正在测试策略: {policy}...")
        try:
            result = benchmark_policy(
                policy_name=policy,
                model_name=model_name,
                prompts=BENCHMARK_PROMPTS,
                compression_ratio=compression_ratio,
            )
            compressed_results[policy] = result
        except Exception as e:
            print(f"错误: 策略 '{policy}' 测试失败: {e}")

    # 3. 打印结果摘要
    print_benchmark_summary(baseline_result, compressed_results)

    # 4. 比较输出质量
    if compressed_results:
        compare_quality(baseline_result, compressed_results)

    print("\n" + "=" * 80)
    print("基准测试完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
