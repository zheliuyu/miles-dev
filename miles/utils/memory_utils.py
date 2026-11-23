import gc
import logging

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _device_memory():
    """Stats from the live device module.

    torch.accelerator's memory queries require a c10 DeviceAllocator. torch_npu's
    caching allocator is not one, so those calls assert on NPU and this module is
    the fallback.
    """
    return getattr(torch, torch.accelerator.current_accelerator().type)


def empty_cache() -> None:
    try:
        torch.accelerator.empty_cache()
    except RuntimeError as exc:
        if "not a DeviceAllocator" not in str(exc):
            raise
        _device_memory().empty_cache()


def clear_memory(clear_host_memory: bool = False):
    torch.accelerator.synchronize()
    gc.collect()
    empty_cache()
    if clear_host_memory:
        torch._C._host_emptyCache()


def available_memory():
    device = torch.accelerator.current_device_index()
    try:
        free, total = torch.accelerator.get_memory_info(device)
        allocated = torch.accelerator.memory_allocated(device)
        reserved = torch.accelerator.memory_reserved(device)
    except RuntimeError as exc:
        if "not a DeviceAllocator" not in str(exc):
            raise
        backend = _device_memory()
        free, total = backend.mem_get_info(device)
        allocated = backend.memory_allocated(device)
        reserved = backend.memory_reserved(device)
    return {
        "gpu": str(device),
        "total_GB": _byte_to_gb(total),
        "free_GB": _byte_to_gb(free),
        "used_GB": _byte_to_gb(total - free),
        "allocated_GB": _byte_to_gb(allocated),
        "reserved_GB": _byte_to_gb(reserved),
    }


def _byte_to_gb(n: int):
    return round(n / (1024**3), 2)


def print_memory(msg, clear_before_print: bool = False):
    if clear_before_print:
        clear_memory()

    memory_info = available_memory()
    # Need to print for all ranks, b/c different rank can have different behaviors
    logger.info(
        f"[Rank {dist.get_rank()}] Memory-Usage {msg}{' (cleared before print)' if clear_before_print else ''}: {memory_info}"
    )
    return memory_info
