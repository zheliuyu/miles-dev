"""Precision policy for the FSDP backend.

Resolves the FSDP ``MixedPrecisionPolicy`` dtypes and whether to keep an fp32 master copy. The master copy is enabled by default for bit-exact weight sync and can be disabled for memory-constrained runs that only require forward accuracy.
"""

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

# Safetensors header names for the dtypes a checkpoint may store per tensor. from_pretrained
# follows config.dtype and can collapse these (Qwen3.5 keeps A_log and the GDN norm in fp32).
_TORCH_DTYPE_BY_SAFETENSORS = {
    "F32": torch.float32,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F64": torch.float64,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


@dataclass
class PrecisionPolicy:
    param_dtype: torch.dtype  # FSDP MixedPrecisionPolicy compute dtype
    reduce_dtype: torch.dtype  # gradient all-reduce dtype
    keep_fp32_master: bool = True
    autocast_dtype: torch.dtype | None = None
    sync_dtype_resolver: Callable[[str, torch.dtype], torch.dtype] | None = None


def default_precision_policy(args) -> PrecisionPolicy:
    """The backend's compute, reduction and master-weight precision; an ``ArchAdapter`` may adjust it."""
    return PrecisionPolicy(
        param_dtype=torch.float16 if args.backend.fp16 else torch.bfloat16,
        reduce_dtype=torch.float32,
        keep_fp32_master=args.backend.keep_fp32_master,
    )


def precision_forward_context(policy: PrecisionPolicy):
    if policy.autocast_dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=policy.autocast_dtype)


def apply_fp32_master(
    model,
    sync_dtype_resolver: Callable[[str, torch.dtype], torch.dtype] | None = None,
):
    """Convert ``model`` to an fp32 master and record each parameter's outbound sync dtype.

    The checkpoint dtype is the default. A model-specific precision policy may override it when the
    rollout contract stores selected parameters at a different dtype.
    """
    sync_dtypes = {}
    for name, param in model.state_dict().items():
        checkpoint_dtype = param.dtype
        sync_dtypes[name] = (
            sync_dtype_resolver(name, checkpoint_dtype) if sync_dtype_resolver is not None else checkpoint_dtype
        )
    model = model.to(torch.float32)
    model._fsdp_sync_dtypes = sync_dtypes
    return model


def _safetensors_dtypes(checkpoint_path: str) -> dict[str, torch.dtype]:
    """Per-tensor dtypes declared in the checkpoint files, before from_pretrained casts them."""
    dtypes = {}
    for path in sorted(Path(checkpoint_path).glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as tensors:
            for name in tensors.keys():
                dtype = _TORCH_DTYPE_BY_SAFETENSORS.get(tensors.get_slice(name).get_dtype())
                if dtype is not None:
                    dtypes[name] = dtype
    return dtypes


def align_sync_dtypes_to_checkpoint(model, checkpoint_path: str) -> int:
    """Point outbound weight sync at the checkpoint file dtype.

    ``from_pretrained`` applies ``config.dtype`` and can turn an fp32 tensor into bf16 before
    ``apply_fp32_master`` records it. Disk-delta then rejects the publish, because it diffs raw
    safetensors bytes. Returns how many recorded dtypes were corrected.
    """
    sync_dtypes = getattr(model, "_fsdp_sync_dtypes", None)
    if not sync_dtypes:
        return 0
    corrected = 0
    for name, dtype in _safetensors_dtypes(checkpoint_path).items():
        if name in sync_dtypes and sync_dtypes[name] != dtype:
            sync_dtypes[name] = dtype
            corrected += 1
    return corrected
