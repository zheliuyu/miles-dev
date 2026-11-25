"""Ascend NPU through torch_npu."""

import torch

try:
    import torch_npu  # noqa: F401  registers the npu device with torch
except ImportError as exc:
    raise ImportError("MILES_ACCELERATOR=npu requires torch_npu, which is not installed.") from exc

from miles.platforms import Platform


class NpuPlatform(Platform):
    device_type = "npu"
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

    def bind_device_before_process_group(self, local_rank: int) -> None:
        # HCCL reads the current device at init_process_group, unlike TPU's delayed rendezvous.
        torch.npu.set_device(local_rank)

    def bind_device_after_process_group(self) -> None:
        pass

    def device_name(self) -> str | None:
        return torch.npu.get_device_name() if torch.npu.is_available() else None
