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
    # the attention backend the 950 serves with
    extra_process_env = {"ASCEND_USE_FIA": "1"}
    sglang_attention_backend = "ascend"
    # No 950 compile backend is selected. dynamic_shapes is False, so the shared dynamic helpers stay eager;
    # a caller that still compiles gets eager execution rather than inductor.
    compile_backend = "eager"
    triton_kernels = False
    flex_attention = False
    # Measured on the 950: HCCL dist.reduce. PREMUL_SUM was not measured, so a non-default gradient divide stays on all_reduce.
    rooted_reduce = True
    premul_sum_reduce = False
    # The HCCL weight group is the only transfer mode this path uses; it was measured on 2 and 4 cards.
    weight_transfer_modes = frozenset({"broadcast"})
    process_group_backend = "hccl"
    # The launcher does not pass --sglang-enable-prefill-weight-versions, so the engine reports no per-prompt weight versions.
    prefill_weight_versions = False
    dynamic_shapes = False
    nvml_numa_affinity = False
    # Keep the rollout KV cache in bf16 so its logprobs match the bf16 trainer.
    rollout_kv_cache_dtype = "bfloat16"

    def device_name(self) -> str | None:
        return torch.npu.get_device_name() if torch.npu.is_available() else None
