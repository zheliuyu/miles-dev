"""The training process's platform: CUDA, NPU (torch_npu), and TPU (torch_tpu), chosen once by MILES_ACCELERATOR.

Everything `torch.accelerator` covers (synchronize, empty_cache, memory stats, the device index) is called directly.
"""

import functools
import importlib
import os

import torch


class Platform:
    """CUDA, and the base other platforms override: every method is the call main makes on CUDA."""

    device_type = "cuda"
    # the environment variable that narrows the process to a set of devices
    visible_devices_env = "CUDA_VISIBLE_DEVICES"
    # environment a launcher sets for every job on this platform
    extra_process_env: dict[str, str] = {}
    # the SGLang attention backend this platform serves with; None keeps SGLang's choice
    sglang_attention_backend: str | None = None
    # the torch.compile backend for model code
    compile_backend = "inductor"
    # Triton kernels (e.g. --log-probs-backend fused) run here
    triton_kernels = True
    # FlexAttention compiles to a block-sparse kernel here
    flex_attention = True
    # a multimodal model's encoders train here; elsewhere the trainer builds its language model alone
    trains_multimodal_encoders = True
    # the process group has a rooted reduce
    rooted_reduce = True
    # the process group has PREMUL_SUM, which FSDP picks for a gradient divide factor other than the group size
    premul_sum_reduce = True
    # the --update-weight-transfer-mode values an engine on this platform can receive; None: all of them
    weight_transfer_modes: frozenset[str] | None = None
    # backend of the process group that broadcasts trainer weights to the rollout engines
    process_group_backend = "nccl"
    # the engine reports the weight version each prompt's KV was computed under (--sglang-enable-prefill-weight-versions)
    prefill_weight_versions = True
    # torch.compile(dynamic=True) works here; a platform without it runs dynamic-shape helpers eager
    dynamic_shapes = True
    # NVML sees the devices, so each training rank is pinned to its GPU's NUMA node
    nvml_numa_affinity = True
    # the rollout engines' KV cache dtype when --sglang-kv-cache-dtype is left at "auto"; None keeps the engine's choice
    rollout_kv_cache_dtype: str | None = None
    # under --true-on-policy-mode the engine returns log-probs in the trainer's precision (bf16); True: in fp32
    true_on_policy_fp32_logprobs = False

    def bind_device_before_process_group(self, local_rank: int) -> None:
        # the vendor reads the current device at init_process_group; torch.accelerator maps local_rank to it
        torch.accelerator.set_device_index(local_rank)

    def bind_device_after_process_group(self) -> None:
        pass

    def current_device(self) -> torch.device:
        return torch.device(self.device_type, torch.accelerator.current_device_index())

    @functools.cache  # noqa: B019 -- one instance per process
    def device_name(self) -> str | None:
        """The CUDA device's marketing name (for peak-FLOPs lookup); None without a CUDA device."""
        return torch.cuda.get_device_name() if torch.cuda.is_available() else None


# where each non-CUDA platform lives; a module imports its vendor runtime only when it is selected
_PLATFORM_MODULES = {
    "npu": ("miles.platforms.npu", "NpuPlatform"),
    "tpu": ("miles.platforms.tpu", "TpuPlatform"),
}


def _select_platform() -> Platform:
    device_type = os.environ.get("MILES_ACCELERATOR", "cuda")
    if device_type == "cuda":
        return Platform()
    target = _PLATFORM_MODULES.get(device_type)
    if target is None:
        raise ValueError(f"MILES_ACCELERATOR={device_type!r}; expected cuda or one of {sorted(_PLATFORM_MODULES)}")
    module_name, class_name = target
    return getattr(
        importlib.import_module(module_name), class_name
    )()  # config-access-exempt: class chosen from _PLATFORM_MODULES


current_platform = _select_platform()
# TODO: the TPU stack's remaining call-site checks become platform capabilities, then this goes
IS_TPU = current_platform.device_type == "tpu"


def torch_compile(*, dynamic: bool):
    """`torch.compile(dynamic=dynamic)` where the platform supports it; a dynamic helper runs eager elsewhere.

    Only for helpers without data-dependent control flow (`compute_approx_kl`, `compute_policy_loss`, the entropy's
    `mul_reduce`). Eager results are numerically close to the compiled ones, not bit-identical: fusion can change
    reduction order and intermediate rounding. Eager `mul_reduce` also materializes the [rows, vocab] product the
    fused kernel avoids, one more fp32 transient per log-prob chunk.
    """

    def wrap(fn):
        if dynamic and not current_platform.dynamic_shapes:
            return fn
        return torch.compile(dynamic=dynamic)(fn)

    return wrap
