"""GPU utilities for Polymathera.

This package provides centralized GPU management including:
- GPU metrics collection (PyTorch CUDA, pynvml, nvidia-smi fallback)
- Memory management and optimization
- Device detection and configuration
"""

from .memory import (
    clear_gpu_cache,
    estimate_model_memory,
    get_gpu_devices,
    get_gpu_memory_info,
    get_gpu_memory_ratio,
    get_optimal_device,
    is_cuda_available,
    manage_gpu_memory,
)
from .metrics import GPUMetrics, GPUMetricsBackend, GPUMetricsCollector

__all__ = [
    # Metrics
    "GPUMetrics",
    "GPUMetricsBackend",
    "GPUMetricsCollector",
    # Memory management
    "is_cuda_available",
    "get_gpu_memory_info",
    "get_gpu_memory_ratio",
    "clear_gpu_cache",
    "get_gpu_devices",
    "manage_gpu_memory",
    "get_optimal_device",
    "estimate_model_memory",
]