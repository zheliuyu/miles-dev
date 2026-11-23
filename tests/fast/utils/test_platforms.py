import builtins
import sys
from types import ModuleType

import pytest

from miles import platforms


def test_an_unknown_accelerator_is_refused(monkeypatch):
    """Any value but cuda, npu, or tpu would otherwise run the CUDA platform silently."""
    monkeypatch.setenv("MILES_ACCELERATOR", "TPU")
    with pytest.raises(ValueError, match="expected cuda, npu, or tpu"):
        platforms._select_platform()


def test_the_default_accelerator_is_cuda_and_does_not_load_the_npu_platform(monkeypatch):
    """A CUDA process must keep working on a machine where torch_npu is not installed."""
    monkeypatch.delenv("MILES_ACCELERATOR", raising=False)
    sys.modules.pop("miles.platforms.npu", None)
    platform = platforms._select_platform()
    assert platform.device_type == "cuda"
    assert platform.process_group_backend == "nccl"
    assert "miles.platforms.npu" not in sys.modules


def test_npu_without_torch_npu_names_the_missing_package(monkeypatch):
    """The failure points at torch_npu. Selecting npu must not break the CUDA import of this package."""
    monkeypatch.setenv("MILES_ACCELERATOR", "npu")
    sys.modules.pop("miles.platforms.npu", None)
    monkeypatch.delitem(sys.modules, "torch_npu", raising=False)
    real_import = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch_npu":
            raise ModuleNotFoundError("No module named 'torch_npu'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(ImportError, match="torch_npu"):
        platforms._select_platform()


def test_npu_selects_the_torch_npu_platform(monkeypatch):
    """The NPU facts are readable without a device: disk-delta and broadcast, no Triton, no NVML, bf16 KV."""
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    sys.modules.pop("miles.platforms.npu", None)
    monkeypatch.setenv("MILES_ACCELERATOR", "npu")
    platform = platforms._select_platform()
    assert platform.device_type == "npu"
    assert platform.compile_backend == "eager"
    assert platform.triton_kernels is False
    assert platform.flex_attention is False
    assert platform.rooted_reduce is False
    assert platform.premul_sum_reduce is False
    assert platform.weight_transfer_modes == frozenset({"disk-delta", "broadcast"})
    assert platform.process_group_backend == "hccl"
    assert platform.prefill_weight_versions is False
    assert platform.dynamic_shapes is False
    assert platform.nvml_numa_affinity is False
    assert platform.rollout_kv_cache_dtype == "bfloat16"


def test_a_dynamic_helper_runs_eager_where_the_platform_compiles_static_shapes_only(monkeypatch):
    """The TPU compile backend rejects dynamic shapes, so the helper must stay the plain function there."""

    def helper(x):
        return x + 1

    monkeypatch.setattr(platforms.current_platform, "dynamic_shapes", False)
    assert platforms.torch_compile(dynamic=True)(helper) is helper

    monkeypatch.setattr(platforms.current_platform, "dynamic_shapes", True)
    assert platforms.torch_compile(dynamic=True)(helper) is not helper
