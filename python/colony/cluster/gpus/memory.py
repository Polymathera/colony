"""GPU memory management utilities.

This module provides GPU memory management functions including:
- Memory allocation and monitoring
- Cache clearing and optimization
- Device detection and configuration
"""

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def is_cuda_available() -> bool:
    """Check if CUDA is available with detailed diagnostics.

    Returns:
        True if CUDA is available and functional, False otherwise
    """
    cuda_available = torch.cuda.is_available()
    logger.info(f"torch.cuda.is_available(): {cuda_available}")

    # Debug CUDA configuration
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    cuda_home = os.environ.get("CUDA_HOME", "")
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")

    logger.info(f"CUDA_VISIBLE_DEVICES: '{cuda_visible_devices}'")
    logger.info(f"CUDA_HOME: '{cuda_home}'")
    logger.info(f"LD_LIBRARY_PATH: '{ld_library_path}'")

    # If CUDA_VISIBLE_DEVICES is set but CUDA not available, try to force init
    if cuda_visible_devices and not cuda_available:
        logger.warning(
            "CUDA_VISIBLE_DEVICES is set but CUDA not available. "
            "Attempting to initialize CUDA context..."
        )
        try:
            torch.cuda.init()
            cuda_available = torch.cuda.is_available()
            logger.info(f"After torch.cuda.init(): torch.cuda.is_available() = {cuda_available}")
        except Exception as e:
            logger.warning(f"Failed to initialize CUDA context: {e}")

    return cuda_available



def get_gpu_memory_info(device_id: int = 0) -> dict[str, int]:
    """Get GPU memory information for a specific device.

    Args:
        device_id: GPU device ID (default: 0)

    Returns:
        Dictionary with memory info in bytes:
        - allocated: Currently allocated memory
        - reserved: Reserved memory by allocator
        - cached: Cached memory
        - total: Total GPU memory
        - free: Free memory
    """
    if not torch.cuda.is_available():
        return {
            'allocated': 0,
            'reserved': 0,
            'cached': 0,
            'total': 0,
            'free': 0,
        }

    with torch.cuda.device(device_id):
        allocated = torch.cuda.memory_allocated(device_id)
        reserved = torch.cuda.memory_reserved(device_id)
        cached = torch.cuda.memory_reserved(device_id)
        total = torch.cuda.get_device_properties(device_id).total_memory
        free = total - allocated

        return {
            'allocated': allocated,
            'reserved': reserved,
            'cached': cached,
            'total': total,
            'free': free,
        }


def get_gpu_memory_ratio(device_id: int = 0) -> float:
    """Get GPU memory utilization ratio.

    Args:
        device_id: GPU device ID (default: 0)

    Returns:
        Memory utilization ratio (0.0 to 1.0)
    """
    memory_info = get_gpu_memory_info(device_id)
    if memory_info['total'] == 0:
        return 0.0

    return memory_info['allocated'] / memory_info['total']


def clear_gpu_cache(device_id: Optional[int] = None):
    """Clear GPU memory cache.

    Args:
        device_id: GPU device ID to clear, or None to clear all devices
    """
    if not torch.cuda.is_available():
        return

    if device_id is not None:
        with torch.cuda.device(device_id):
            torch.cuda.empty_cache()
        logger.debug(f"Cleared GPU {device_id} cache")
    else:
        torch.cuda.empty_cache()
        logger.debug("Cleared all GPU caches")

def initialize_nvml(self):
    """Initialize NVML."""
    # Try to initialize NVIDIA ML regardless of PyTorch CUDA availability
    try:
        import pynvml
        pynvml.nvmlInit()
        pynvml_lib = pynvml
        nvml_available = True
        logger.info("NVIDIA ML monitoring initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize NVIDIA ML: {e}")
        pynvml_lib = None
        nvml_available = False
    return pynvml_lib, nvml_available


def get_gpu_devices() -> list[dict]:
    """Get information about all available GPU devices.

    Returns:
        List of dictionaries with device info:
        - id: Device ID
        - name: Device name
        - total_memory: Total memory in bytes
        - major: Compute capability major version
        - minor: Compute capability minor version
        - multi_processor_count: Number of multiprocessors
    """
    pynvml_lib, nvml_available = initialize_nvml()

    cuda_available = torch.cuda.is_available()
    devices = []
    if cuda_available:
        device_count = torch.cuda.device_count()
        if device_count > 0:
            logger.info(f"torch.cuda.device_count(): {device_count}")
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            logger.info(f"Current CUDA device: {current_device}, name: {device_name}")
            gpu_monitoring_available = True
        else:
            logger.warning("CUDA is available but no GPU devices found")

        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            device_info = {
                'id': i,
                'name': props.name,
                'total_memory': props.total_memory,
                'major': props.major,
                'minor': props.minor,
                'multi_processor_count': props.multi_processor_count,
            }
            devices.append(device_info)
            logger.info(f"GPU {i}: {devices[-1]}")

        return devices
    else:
        logger.warning("CUDA is not available in this Ray actor context")
        # If PyTorch CUDA failed but NVIDIA ML works, try to get basic device info
        if nvml_available:
            try:
                nvml_device_count = pynvml_lib.nvmlDeviceGetCount()
                logger.info(f"NVIDIA ML device count: {nvml_device_count}")

                # Create basic device info from NVIDIA ML
                for i in range(nvml_device_count):
                    handle = pynvml_lib.nvmlDeviceGetHandleByIndex(i)
                    name_raw = pynvml_lib.nvmlDeviceGetName(handle)
                    # Handle both string and bytes return types
                    name = name_raw.decode('utf-8') if isinstance(name_raw, bytes) else name_raw
                    memory_info = pynvml_lib.nvmlDeviceGetMemoryInfo(handle)

                    device_info = {
                        'id': i,
                        'name': name,
                        'total_memory': memory_info.total,
                        'major': 0,  # Not available from NVIDIA ML
                        'minor': 0,  # Not available from NVIDIA ML
                        'multi_processor_count': 0  # Not available from NVIDIA ML
                    }
                    devices.append(device_info)
                    logger.info(f"GPU {i} (via NVIDIA ML): {device_info}")

                if nvml_device_count > 0:
                    gpu_monitoring_available = True
                    logger.info(f"Using NVIDIA ML for GPU monitoring: {nvml_device_count} devices")
            except Exception as e:
                logger.warning(f"Failed to get device info from NVIDIA ML: {e}")

        if not gpu_monitoring_available:
            logger.warning("No GPU devices detected due to CUDA unavailability")

    logger.info(f"GPU monitoring initialization complete. Available: {gpu_monitoring_available}, Devices: {len(devices)}")


def manage_gpu_memory(
    threshold: float = 0.9,
    device_id: int = 0,
    clear_cache_on_high: bool = True,
) -> bool:
    """Manage GPU memory and take action if threshold exceeded.

    Args:
        threshold: Memory ratio threshold (0.0 to 1.0)
        device_id: GPU device ID
        clear_cache_on_high: Whether to clear cache if threshold exceeded

    Returns:
        True if memory is within threshold, False if exceeded
    """
    if not torch.cuda.is_available():
        return True

    memory_ratio = get_gpu_memory_ratio(device_id)

    if memory_ratio > threshold:
        logger.warning(
            f"GPU {device_id} memory usage high: {memory_ratio:.2%} "
            f"(threshold: {threshold:.2%})"
        )

        if clear_cache_on_high:
            clear_gpu_cache(device_id)
            new_ratio = get_gpu_memory_ratio(device_id)
            logger.info(
                f"GPU {device_id} memory after cache clear: {new_ratio:.2%} "
                f"(was: {memory_ratio:.2%})"
            )

        return False

    return True


def get_optimal_device(prefer_gpu: bool = True) -> str:
    """Get the optimal device for computation.

    Args:
        prefer_gpu: Whether to prefer GPU if available

    Returns:
        Device string ("cuda:0", "cuda:1", or "cpu")
    """
    if not prefer_gpu or not torch.cuda.is_available():
        return "cpu"

    # Select GPU with most free memory
    devices = get_gpu_devices()
    if not devices:
        return "cpu"

    best_device = 0
    max_free = 0

    for device in devices:
        memory_info = get_gpu_memory_info(device['id'])
        if memory_info['free'] > max_free:
            max_free = memory_info['free']
            best_device = device['id']

    return f"cuda:{best_device}"


def estimate_model_memory(
    num_parameters: int,
    bits_per_param: int = 16,
    overhead_ratio: float = 0.2,
) -> int:
    """Estimate GPU memory needed for a model.

    Args:
        num_parameters: Number of model parameters
        bits_per_param: Bits per parameter (16 for fp16, 32 for fp32, 8 for int8, 4 for int4)
        overhead_ratio: Additional overhead ratio (default 20%)

    Returns:
        Estimated memory in bytes
    """
    bytes_per_param = bits_per_param / 8
    model_memory = num_parameters * bytes_per_param
    total_memory = model_memory * (1 + overhead_ratio)

    return int(total_memory)