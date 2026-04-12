"""Benchmarks for agent workload patterns with context composition.

This module provides benchmarks to measure performance characteristics of
the Virtual Context Manager (VCM) layer, particularly focusing on:

1. Context composition efficiency (shared base pages + agent-specific suffixes)
2. Concurrency patterns (multiple agents on same base page)
3. Memory efficiency (KV cache utilization vs theoretical)
4. Throughput under various loads

Run benchmarks:
    python -m polymathera.colony.vcm.benchmarks

Requirements:
    - Ray cluster with GPU nodes
    - vLLM deployment with prefix caching enabled
    - Test model (e.g., TinyLlama-1.1B-Chat-v1.0)
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class BenchmarkResult:
    """Results from a benchmark run.

    Attributes:
        name: Benchmark name
        duration_s: Total duration in seconds
        num_requests: Number of requests processed
        throughput_rps: Requests per second
        avg_latency_ms: Average latency per request
        p50_latency_ms: 50th percentile latency
        p95_latency_ms: 95th percentile latency
        p99_latency_ms: 99th percentile latency
        memory_efficiency: Memory efficiency score (1.0 = perfect sharing)
        concurrent_peak: Peak concurrent requests
        queue_time_ms: Average queue time
        errors: Number of errors encountered
        metadata: Additional benchmark-specific metadata
    """

    name: str
    duration_s: float
    num_requests: int
    throughput_rps: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    memory_efficiency: float = 1.0
    concurrent_peak: int = 0
    queue_time_ms: float = 0.0
    errors: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Pretty-print benchmark results."""
        return f"""
{'='*80}
Benchmark: {self.name}
{'='*80}
Duration:           {self.duration_s:.2f}s
Requests:           {self.num_requests}
Throughput:         {self.throughput_rps:.2f} req/s
Latency (avg):      {self.avg_latency_ms:.2f}ms
Latency (p50/p95/p99): {self.p50_latency_ms:.0f}/{self.p95_latency_ms:.0f}/{self.p99_latency_ms:.0f}ms
Queue Time (avg):   {self.queue_time_ms:.2f}ms
Memory Efficiency:  {self.memory_efficiency:.2%}
Peak Concurrency:   {self.concurrent_peak}
Errors:             {self.errors}
{'='*80}
"""


class ContextCompositionBenchmarks:
    """Benchmarks for context composition patterns.

    These benchmarks measure the performance characteristics of the
    infer_with_suffix() API under realistic agent workload patterns.
    """

    def __init__(self, vllm_deployment_handle, base_page_tokens: list[int], num_tokens_in_base: int = 10000):
        """Initialize benchmarks.

        Args:
            vllm_deployment_handle: Ray handle to VLLMDeployment
            base_page_tokens: Base page tokens to use for benchmarks
            num_tokens_in_base: Number of tokens in the base page (for metrics)
        """
        self.vllm = vllm_deployment_handle
        self.base_page_tokens = base_page_tokens
        self.num_tokens_in_base = num_tokens_in_base

    async def benchmark_sequential_agents(
        self,
        num_agents: int = 10,
        suffix_size: int = 500,
    ) -> BenchmarkResult:
        """Benchmark sequential agent execution (baseline).

        Measures performance when agents run one after another, providing
        a baseline for comparison with concurrent execution.

        Args:
            num_agents: Number of sequential agents to run
            suffix_size: Size of agent-specific suffix in tokens

        Returns:
            Benchmark results
        """
        from ..vcm.models import VirtualContextPage
        from ..cluster.models import InferenceRequest

        # Create base page
        base_page = VirtualContextPage(
            page_id="benchmark-sequential-base",
            tokens=self.base_page_tokens[:self.num_tokens_in_base],
            size=self.num_tokens_in_base,
        )

        # Load base page
        await self.vllm.load_page(base_page)

        # Prepare agent suffixes
        agent_suffixes = [
            list(range(i * 100, i * 100 + suffix_size))
            for i in range(num_agents)
        ]

        # Run agents sequentially
        latencies = []
        start_time = time.time()

        for i, suffix in enumerate(agent_suffixes):
            request = InferenceRequest(
                request_id=f"seq-agent-{i}",
                prompt="",  # Will be overridden by composition
                max_tokens=50,
            )

            req_start = time.time()
            try:
                response = await self.vllm.infer_with_suffix(
                    base_page_id=base_page.page_id,
                    suffix_tokens=suffix,
                    request=request,
                )
                latencies.append((time.time() - req_start) * 1000)
            except Exception as e:
                print(f"Error in agent {i}: {e}")
                latencies.append(0)

        duration = time.time() - start_time

        # Calculate statistics
        latencies_np = np.array([l for l in latencies if l > 0])
        return BenchmarkResult(
            name="Sequential Agents (Baseline)",
            duration_s=duration,
            num_requests=num_agents,
            throughput_rps=num_agents / duration,
            avg_latency_ms=float(np.mean(latencies_np)),
            p50_latency_ms=float(np.percentile(latencies_np, 50)),
            p95_latency_ms=float(np.percentile(latencies_np, 95)),
            p99_latency_ms=float(np.percentile(latencies_np, 99)),
            memory_efficiency=1.0,  # Perfect - only one active at a time
            concurrent_peak=1,
            errors=len([l for l in latencies if l == 0]),
            metadata={
                "num_agents": num_agents,
                "suffix_size": suffix_size,
                "base_size": self.num_tokens_in_base,
            },
        )

    async def benchmark_concurrent_agents(
        self,
        num_agents: int = 10,
        suffix_size: int = 500,
        max_concurrent_per_page: int | None = None,
    ) -> BenchmarkResult:
        """Benchmark concurrent agent execution.

        Measures performance when multiple agents run concurrently on the
        same base page with different suffixes. This tests vLLM's prefix
        caching effectiveness and concurrency control.

        Args:
            num_agents: Number of concurrent agents
            suffix_size: Size of agent-specific suffix in tokens
            max_concurrent_per_page: Override per-page concurrency limit

        Returns:
            Benchmark results
        """
        from ..vcm.models import VirtualContextPage
        from ..cluster.models import InferenceRequest

        # Create base page
        base_page = VirtualContextPage(
            page_id="benchmark-concurrent-base",
            tokens=self.base_page_tokens[:self.num_tokens_in_base],
            size=self.num_tokens_in_base,
        )

        # Load base page
        await self.vllm.load_page(base_page)

        # Prepare agent suffixes
        agent_suffixes = [
            list(range(i * 100, i * 100 + suffix_size))
            for i in range(num_agents)
        ]

        # Define agent task
        async def run_agent(agent_id: int, suffix: list[int]):
            request = InferenceRequest(
                request_id=f"concurrent-agent-{agent_id}",
                prompt="",
                max_tokens=50,
            )

            req_start = time.time()
            try:
                response = await self.vllm.infer_with_suffix(
                    base_page_id=base_page.page_id,
                    suffix_tokens=suffix,
                    request=request,
                    max_concurrent_per_page=max_concurrent_per_page,
                )
                return (time.time() - req_start) * 1000, None
            except Exception as e:
                return 0, str(e)

        # Run all agents concurrently
        start_time = time.time()
        results = await asyncio.gather(*[
            run_agent(i, suffix)
            for i, suffix in enumerate(agent_suffixes)
        ])
        duration = time.time() - start_time

        # Extract latencies and errors
        latencies = [r[0] for r in results if r[0] > 0]
        errors = [r[1] for r in results if r[1] is not None]

        # Theoretical memory efficiency
        # Without sharing: num_agents * (base + suffix)
        # With sharing: base + (num_agents * suffix)
        without_sharing = num_agents * (self.num_tokens_in_base + suffix_size)
        with_sharing = self.num_tokens_in_base + (num_agents * suffix_size)
        memory_efficiency = with_sharing / without_sharing if without_sharing > 0 else 1.0

        latencies_np = np.array(latencies)
        return BenchmarkResult(
            name=f"Concurrent Agents (n={num_agents})",
            duration_s=duration,
            num_requests=num_agents,
            throughput_rps=num_agents / duration,
            avg_latency_ms=float(np.mean(latencies_np)) if len(latencies_np) > 0 else 0,
            p50_latency_ms=float(np.percentile(latencies_np, 50)) if len(latencies_np) > 0 else 0,
            p95_latency_ms=float(np.percentile(latencies_np, 95)) if len(latencies_np) > 0 else 0,
            p99_latency_ms=float(np.percentile(latencies_np, 99)) if len(latencies_np) > 0 else 0,
            memory_efficiency=memory_efficiency,
            concurrent_peak=num_agents,
            errors=len(errors),
            metadata={
                "num_agents": num_agents,
                "suffix_size": suffix_size,
                "base_size": self.num_tokens_in_base,
                "theoretical_memory_saved": f"{(1 - memory_efficiency) * 100:.1f}%",
                "max_concurrent_per_page": max_concurrent_per_page,
                "error_messages": errors[:5] if errors else [],
            },
        )

    async def benchmark_scaling_concurrency(
        self,
        concurrency_levels: list[int] = [1, 2, 5, 10, 20, 50],
        suffix_size: int = 500,
    ) -> list[BenchmarkResult]:
        """Benchmark performance across different concurrency levels.

        Measures how throughput and latency scale with increasing concurrency.
        Helps determine optimal concurrency settings.

        Args:
            concurrency_levels: List of concurrency levels to test
            suffix_size: Size of agent-specific suffix

        Returns:
            List of benchmark results, one per concurrency level
        """
        results = []

        for concurrency in concurrency_levels:
            print(f"\nTesting concurrency level: {concurrency}")
            result = await self.benchmark_concurrent_agents(
                num_agents=concurrency,
                suffix_size=suffix_size,
            )
            results.append(result)
            print(result)

            # Brief pause between tests
            await asyncio.sleep(2)

        return results

    async def benchmark_varying_suffix_sizes(
        self,
        suffix_sizes: list[int] = [100, 500, 1000, 2000, 5000],
        num_agents: int = 10,
    ) -> list[BenchmarkResult]:
        """Benchmark performance with varying suffix sizes.

        Tests how suffix size affects performance and memory efficiency.
        Larger suffixes mean more unique KV blocks per request.

        Args:
            suffix_sizes: List of suffix sizes to test
            num_agents: Number of concurrent agents per test

        Returns:
            List of benchmark results, one per suffix size
        """
        results = []

        for suffix_size in suffix_sizes:
            print(f"\nTesting suffix size: {suffix_size} tokens")
            result = await self.benchmark_concurrent_agents(
                num_agents=num_agents,
                suffix_size=suffix_size,
            )
            results.append(result)
            print(result)

            await asyncio.sleep(2)

        return results

    async def benchmark_mixed_workload(
        self,
        num_unique_suffixes: int = 3,
        requests_per_suffix: int = 10,
    ) -> BenchmarkResult:
        """Benchmark mixed workload with repeated suffixes.

        Simulates realistic scenarios where multiple agents use the same
        task templates (e.g., code review, refactoring, optimization).

        Args:
            num_unique_suffixes: Number of unique agent task types
            requests_per_suffix: Number of requests per task type

        Returns:
            Benchmark results
        """
        from ..vcm.models import VirtualContextPage
        from ..cluster.models import InferenceRequest

        # Create base page
        base_page = VirtualContextPage(
            page_id="benchmark-mixed-base",
            tokens=self.base_page_tokens[:self.num_tokens_in_base],
            size=self.num_tokens_in_base,
        )

        await self.vllm.load_page(base_page)

        # Create unique suffixes (agent templates)
        unique_suffixes = [
            list(range(i * 1000, i * 1000 + 500))
            for i in range(num_unique_suffixes)
        ]

        # Generate requests (cycle through suffixes)
        tasks = []
        for i in range(num_unique_suffixes * requests_per_suffix):
            suffix = unique_suffixes[i % num_unique_suffixes]

            async def run_request(req_id: int, suffix_tokens: list[int]):
                request = InferenceRequest(
                    request_id=f"mixed-{req_id}",
                    prompt="",
                    max_tokens=50,
                )

                req_start = time.time()
                try:
                    await self.vllm.infer_with_suffix(
                        base_page_id=base_page.page_id,
                        suffix_tokens=suffix_tokens,
                        request=request,
                    )
                    return (time.time() - req_start) * 1000, None
                except Exception as e:
                    return 0, str(e)

            tasks.append(run_request(i, suffix))

        # Run all requests concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time

        latencies = [r[0] for r in results if r[0] > 0]
        errors = [r[1] for r in results if r[1] is not None]

        # Memory efficiency calculation
        # With perfect suffix sharing: base + (num_unique_suffixes * suffix_size)
        # Without sharing: (num_unique_suffixes * requests_per_suffix) * (base + suffix)
        total_requests = num_unique_suffixes * requests_per_suffix
        without_sharing = total_requests * (self.num_tokens_in_base + 500)
        with_sharing = self.num_tokens_in_base + (num_unique_suffixes * 500)
        memory_efficiency = with_sharing / without_sharing if without_sharing > 0 else 1.0

        latencies_np = np.array(latencies)
        return BenchmarkResult(
            name=f"Mixed Workload ({num_unique_suffixes} templates × {requests_per_suffix} requests)",
            duration_s=duration,
            num_requests=total_requests,
            throughput_rps=total_requests / duration,
            avg_latency_ms=float(np.mean(latencies_np)) if len(latencies_np) > 0 else 0,
            p50_latency_ms=float(np.percentile(latencies_np, 50)) if len(latencies_np) > 0 else 0,
            p95_latency_ms=float(np.percentile(latencies_np, 95)) if len(latencies_np) > 0 else 0,
            p99_latency_ms=float(np.percentile(latencies_np, 99)) if len(latencies_np) > 0 else 0,
            memory_efficiency=memory_efficiency,
            concurrent_peak=total_requests,
            errors=len(errors),
            metadata={
                "num_unique_suffixes": num_unique_suffixes,
                "requests_per_suffix": requests_per_suffix,
                "total_requests": total_requests,
                "base_size": self.num_tokens_in_base,
            },
        )


async def run_all_benchmarks(vllm_deployment_handle, base_page_tokens: list[int]):
    """Run complete benchmark suite.

    Args:
        vllm_deployment_handle: Ray handle to VLLMDeployment
        base_page_tokens: Base page tokens for benchmarks
    """
    print("\n" + "="*80)
    print("PHYSICAL CONTEXT MANAGER - BENCHMARK SUITE")
    print("="*80)

    benchmarks = ContextCompositionBenchmarks(
        vllm_deployment_handle=vllm_deployment_handle,
        base_page_tokens=base_page_tokens,
        num_tokens_in_base=10000,
    )

    all_results = []

    # Benchmark 1: Sequential baseline
    print("\n[1/5] Running sequential agents benchmark...")
    result = await benchmarks.benchmark_sequential_agents(num_agents=10)
    print(result)
    all_results.append(result)

    # Benchmark 2: Concurrent agents
    print("\n[2/5] Running concurrent agents benchmark...")
    result = await benchmarks.benchmark_concurrent_agents(num_agents=10)
    print(result)
    all_results.append(result)

    # Benchmark 3: Scaling concurrency
    print("\n[3/5] Running scaling concurrency benchmark...")
    scaling_results = await benchmarks.benchmark_scaling_concurrency(
        concurrency_levels=[1, 2, 5, 10, 20],
    )
    all_results.extend(scaling_results)

    # Benchmark 4: Varying suffix sizes
    print("\n[4/5] Running varying suffix sizes benchmark...")
    suffix_results = await benchmarks.benchmark_varying_suffix_sizes(
        suffix_sizes=[100, 500, 1000, 2000],
        num_agents=10,
    )
    all_results.extend(suffix_results)

    # Benchmark 5: Mixed workload
    print("\n[5/5] Running mixed workload benchmark...")
    result = await benchmarks.benchmark_mixed_workload(
        num_unique_suffixes=3,
        requests_per_suffix=10,
    )
    print(result)
    all_results.append(result)

    # Summary
    print("\n" + "="*80)
    print("BENCHMARK SUITE COMPLETE")
    print("="*80)
    print(f"\nTotal benchmarks run: {len(all_results)}")
    print("\nKey findings:")
    print(f"  - Best throughput: {max(r.throughput_rps for r in all_results):.2f} req/s")
    print(f"  - Best latency (p50): {min(r.p50_latency_ms for r in all_results if r.p50_latency_ms > 0):.0f}ms")
    print(f"  - Best memory efficiency: {max(r.memory_efficiency for r in all_results):.2%}")

    return all_results


if __name__ == "__main__":
    print("To run benchmarks, use the following pattern:")
    print("""
    import ray
    from polymathera.colony import VLLMDeployment
    from polymathera.colony.vcm.benchmarks import run_all_benchmarks

    # Get deployment handle
    vllm = ray.get_actor("vllm-deployment-name")

    # Prepare base page tokens (example)
    base_tokens = list(range(10000))

    # Run benchmarks
    results = await run_all_benchmarks(vllm, base_tokens)
    """)