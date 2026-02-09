"""GPU metrics monitoring for LLM cluster.

This module provides GPU metrics collection with multiple fallback mechanisms:
1. PyTorch CUDA (fastest, most accurate for PyTorch workloads)
2. pynvml (NVIDIA Management Library)
3. nvidia-smi (command-line fallback)

The module automatically selects the best available method and provides
a unified interface for GPU metrics.
"""

import logging
import subprocess
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Any

logger = logging.getLogger(__name__)


class GPUMetricsBackend(Enum):
    """Available GPU metrics backends."""
    PYTORCH_CUDA = auto()
    PYNVML = auto()
    NVIDIA_SMI = auto()
    NONE = auto()


@dataclass
class GPUMetrics:
    """GPU metrics snapshot."""

    gpu_id: int
    gpu_name: str

    # Memory metrics (in MB)
    memory_used_mb: float
    memory_total_mb: float
    memory_free_mb: float
    memory_utilization_pct: float

    # PyTorch CUDA-specific memory metrics (in bytes, None if not available)
    memory_allocated_bytes: float | None = None
    memory_reserved_bytes: float | None = None
    memory_cached_bytes: float | None = None

    # Compute metrics
    gpu_utilization_pct: float = 0.0
    temperature_celsius: Optional[float] = None
    power_draw_watts: Optional[float] = None

    process_count: int | None = None  # Number of active GPU processes

    # Backend used to collect these metrics
    backend: GPUMetricsBackend = GPUMetricsBackend.NONE

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            'gpu_id': self.gpu_id,
            'gpu_name': self.gpu_name,
            'memory_used_mb': self.memory_used_mb,
            'memory_total_mb': self.memory_total_mb,
            'memory_free_mb': self.memory_free_mb,
            'memory_utilization_pct': self.memory_utilization_pct,
            'gpu_utilization_pct': self.gpu_utilization_pct,
            'temperature_celsius': self.temperature_celsius,
            'power_draw_watts': self.power_draw_watts,
            'process_count': self.process_count,
            'backend': self.backend.name,
        }


class GPUMetricsCollector:
    """Collects GPU metrics using the best available backend.

    Attempts backends in order:
    1. PyTorch CUDA - fastest, best for PyTorch workloads
    2. pynvml - comprehensive NVIDIA metrics
    3. nvidia-smi - command-line fallback

    If all backends fail, returns None and logs warnings.
    """

    def __init__(self, gpu_id: int = 0):
        """Initialize metrics collector.

        Args:
            gpu_id: GPU device ID to monitor (default: 0)
        """
        self.gpu_id = gpu_id
        self.backend = self._detect_backend()

        # Initialize backend-specific resources
        if self.backend == GPUMetricsBackend.PYNVML:
            self._init_pynvml()

        logger.info(f"GPU metrics collector initialized with backend: {self.backend.name}")

    def _detect_backend(self) -> GPUMetricsBackend:
        """Detect the best available GPU metrics backend."""
        # Try PyTorch CUDA first
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("Using PyTorch CUDA for GPU metrics")
                return GPUMetricsBackend.PYTORCH_CUDA
        except ImportError:
            pass

        # Try pynvml
        try:
            import pynvml
            pynvml.nvmlInit()
            logger.info("Using pynvml for GPU metrics")
            return GPUMetricsBackend.PYNVML
        except (ImportError, Exception) as e:
            logger.debug(f"pynvml not available: {e}")

        # Try nvidia-smi
        try:
            result = subprocess.run(
                ['nvidia-smi', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info("Using nvidia-smi for GPU metrics")
                return GPUMetricsBackend.NVIDIA_SMI
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.debug(f"nvidia-smi not available: {e}")

        logger.warning("No GPU metrics backend available")
        return GPUMetricsBackend.NONE

    def _init_pynvml(self):
        """Initialize pynvml resources."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self.pynvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
        except Exception as e:
            logger.error(f"Failed to initialize pynvml: {e}")
            self.backend = GPUMetricsBackend.NVIDIA_SMI  # Fallback

    def collect(self) -> Optional[GPUMetrics]:
        """Collect current GPU metrics.

        Returns:
            GPUMetrics snapshot, or None if collection fails
        """
        if self.backend == GPUMetricsBackend.PYTORCH_CUDA:
            return self._collect_pytorch()
        elif self.backend == GPUMetricsBackend.PYNVML:
            return self._collect_pynvml()
        elif self.backend == GPUMetricsBackend.NVIDIA_SMI:
            return self._collect_nvidia_smi()
        else:
            return None

    def _collect_pytorch(self) -> Optional[GPUMetrics]:
        """Collect metrics using PyTorch CUDA."""
        try:
            import torch

            # Memory stats (in bytes for PyTorch-specific metrics)
            memory_allocated_bytes = torch.cuda.memory_allocated(self.gpu_id)
            memory_reserved_bytes = torch.cuda.memory_reserved(self.gpu_id)
            memory_cached_bytes = torch.cuda.memory_reserved(self.gpu_id)
            memory_total_bytes = torch.cuda.get_device_properties(self.gpu_id).total_memory

            # Convert to MB for standard metrics
            memory_allocated_mb = memory_allocated_bytes / (1024 ** 2)
            memory_total_mb = memory_total_bytes / (1024 ** 2)
            memory_free_mb = memory_total_mb - memory_allocated_mb

            # GPU utilization (PyTorch doesn't provide this directly)
            # We'll estimate based on memory usage as a proxy
            gpu_utilization = (memory_allocated_mb / memory_total_mb) * 100 if memory_total_mb > 0 else 0

            return GPUMetrics(
                gpu_id=self.gpu_id,
                gpu_name=torch.cuda.get_device_name(self.gpu_id),
                memory_used_mb=memory_allocated_mb,
                memory_total_mb=memory_total_mb,
                memory_free_mb=memory_free_mb,
                memory_utilization_pct=(memory_allocated_mb / memory_total_mb * 100) if memory_total_mb > 0 else 0,
                memory_allocated_bytes=memory_allocated_bytes,
                memory_reserved_bytes=memory_reserved_bytes,
                memory_cached_bytes=memory_cached_bytes,
                gpu_utilization_pct=gpu_utilization,
                backend=GPUMetricsBackend.PYTORCH_CUDA,
            )

        except Exception as e:
            logger.error(f"Failed to collect PyTorch metrics: {e}")
            return None

    def _collect_pynvml(self) -> Optional[GPUMetrics]:
        """Collect metrics using pynvml."""
        try:
            import pynvml

            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.pynvml_handle)
            memory_used_mb = mem_info.used / (1024 ** 2)
            memory_total_mb = mem_info.total / (1024 ** 2)
            memory_free_mb = mem_info.free / (1024 ** 2)

            # GPU utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.pynvml_handle)
            gpu_utilization = utilization.gpu

            # Temperature
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(
                    self.pynvml_handle,
                    pynvml.NVML_TEMPERATURE_GPU
                )
            except:
                temperature = None

            # Power draw
            try:
                power_draw = pynvml.nvmlDeviceGetPowerUsage(self.pynvml_handle) / 1000.0  # mW to W
            except:
                power_draw = None

            # Process count
            try:
                processes = self.pynvml.nvmlDeviceGetComputeRunningProcesses(self.pynvml_handle)
            except:
                processes = None

            # GPU name
            gpu_name = pynvml.nvmlDeviceGetName(self.pynvml_handle)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode('utf-8')

            return GPUMetrics(
                gpu_id=self.gpu_id,
                gpu_name=gpu_name,
                memory_used_mb=memory_used_mb,
                memory_total_mb=memory_total_mb,
                memory_free_mb=memory_free_mb,
                memory_utilization_pct=(memory_used_mb / memory_total_mb * 100) if memory_total_mb > 0 else 0,
                gpu_utilization_pct=gpu_utilization,
                temperature_celsius=temperature,
                power_draw_watts=power_draw,
                process_count=len(processes) if processes is not None else None,
                backend=GPUMetricsBackend.PYNVML,
            )

        except Exception as e:
            logger.error(f"Failed to collect pynvml metrics: {e}")
            return None

    def _collect_nvidia_smi(self) -> Optional[GPUMetrics]:
        """Collect metrics using nvidia-smi."""
        try:
            # Query nvidia-smi for metrics
            result = subprocess.run(
                [
                    'nvidia-smi',
                    '--query-gpu=index,name,memory.used,memory.total,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,processes.count',
                    '--format=csv,noheader,nounits',
                    '-i', str(self.gpu_id)
                ],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                logger.error(f"nvidia-smi failed: {result.stderr}")
                return None

            # Parse output
            parts = result.stdout.strip().split(', ')
            if len(parts) < 8:
                logger.error(f"Unexpected nvidia-smi output: {result.stdout}")
                return None

            gpu_id = int(parts[0])
            gpu_name = parts[1]
            memory_used_mb = float(parts[2])
            memory_total_mb = float(parts[3])
            memory_free_mb = float(parts[4])
            gpu_utilization = float(parts[5])
            mem_utilization = float(parts[6]) if parts[6] != '[Not Supported]' else 0
            temperature = float(parts[7]) if parts[7].strip() not in ['N/A', ''] else None
            power_draw = float(parts[8]) if parts[8].strip() not in ['N/A', ''] else None
            process_count = int(parts[9]) if parts[9] != '[Not Supported]' else 0

            return GPUMetrics(
                gpu_id=gpu_id,
                gpu_name=gpu_name,
                memory_used_mb=memory_used_mb,
                memory_total_mb=memory_total_mb,
                memory_free_mb=memory_free_mb,
                memory_utilization_pct=(memory_used_mb / memory_total_mb * 100) if memory_total_mb > 0 else 0,  # mem_utilization
                gpu_utilization_pct=gpu_utilization,
                temperature_celsius=temperature,
                power_draw_watts=power_draw,
                process_count=process_count,
                backend=GPUMetricsBackend.NVIDIA_SMI,
            )

        except Exception as e:
            logger.error(f"Failed to collect nvidia-smi metrics: {e}")
            return None

    def cleanup(self):
        """Cleanup resources."""
        if self.backend == GPUMetricsBackend.PYNVML:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except:
                pass


