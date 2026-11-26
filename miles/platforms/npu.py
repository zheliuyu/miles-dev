"""Ascend NPU through torch_npu."""

import torch

try:
    import torch_npu  # noqa: F401  registers the npu device with torch
except ImportError as exc:
    raise ImportError("MILES_ACCELERATOR=npu requires torch_npu, which is not installed.") from exc

from miles.platforms import Platform


class NpuPlatform(Platform):
    device_type = "npu"
    visible_devices_env = "ASCEND_RT_VISIBLE_DEVICES"
    # the 950 multimodal attention backend selection, same as the working probe D launch
    extra_process_env = {"ASCEND_USE_FIA": "1"}
    sglang_attention_backend = "ascend"
    # No 950 compile backend is selected. dynamic_shapes is False, so the shared dynamic helpers stay eager;
    # a caller that still compiles gets eager execution rather than inductor.
    compile_backend = "eager"
    triton_kernels = False
    flex_attention = False
    # Probe B measured HCCL dist.reduce. PREMUL_SUM is unmeasured, so a non-default gradient divide stays on all_reduce.
    rooted_reduce = True
    premul_sum_reduce = False
    # The HCCL weight group is the only transfer mode this path uses; it was measured on 2 and 4 cards.
    weight_transfer_modes = frozenset({"broadcast"})
    process_group_backend = "hccl"
    # SGLang 0.5.23 has no --enable-prefill-weight-versions.
    prefill_weight_versions = False
    dynamic_shapes = False
    nvml_numa_affinity = False
    # Keep the rollout KV cache in bf16 so its logprobs match the bf16 trainer.
    rollout_kv_cache_dtype = "bfloat16"

    def empty_cache(self) -> None:
        # torch.accelerator's memory calls assert: the npu allocator is not a c10 DeviceAllocator
        torch.npu.empty_cache()

    def device_memory(self) -> tuple[int, int, int, int]:
        device = torch.npu.current_device()
        free, total = torch.npu.mem_get_info(device)
        return free, total, torch.npu.memory_allocated(device), torch.npu.memory_reserved(device)

    def device_name(self) -> str | None:
        return torch.npu.get_device_name() if torch.npu.is_available() else None
