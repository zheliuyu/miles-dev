#!/bin/bash
# 探针 A：单卡算子。ASCEND_RT_VISIBLE_DEVICES=0，bf16 矩阵乘 + 一次反向，结果拉回 CPU。
set -u
ENV=/root/miniconda3/envs/miles-950-py312
export PATH="$ENV/bin:/root/miniconda3/bin:$PATH"
# shellcheck disable=SC1091
source /home/pkg/cann920/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0

echo "=== 探针 A start $(date -Is) ==="
echo "visible=$ASCEND_RT_VISIBLE_DEVICES"
timeout 600 python - <<'PY'
import torch
import torch_npu

print("torch", torch.__version__, "| torch_npu", torch_npu.__version__)
print("device_count", torch.npu.device_count())
torch.npu.set_device(0)
print("device_name", torch.npu.get_device_name(0))

torch.npu.reset_peak_memory_stats()
base_alloc = torch.npu.memory_allocated() / 1024**2

n = 4096
a = torch.randn(n, n, dtype=torch.bfloat16, device="npu", requires_grad=True)
b = torch.randn(n, n, dtype=torch.bfloat16, device="npu", requires_grad=True)

out = a @ b
loss = out.float().pow(2).mean()
loss.backward()

print("loss", float(loss.detach().cpu()))
print("grad a norm", float(a.grad.float().norm().cpu()))
print("grad b norm", float(b.grad.float().norm().cpu()))
print("finite outputs:", bool(torch.isfinite(out.float()).all().cpu()))

peak = torch.npu.max_memory_allocated() / 1024**2
cur = torch.npu.memory_allocated() / 1024**2
print(f"HBM alloc MB: base={base_alloc:.0f} peak={peak:.0f} current={cur:.0f}")

del a, b, out, loss
torch.npu.empty_cache()
after = torch.npu.memory_allocated() / 1024**2
print(f"HBM after free={after:.0f}")
print("PROBE_A_OK")
PY
rc=$?
echo "=== 探针 A exit=$rc $(date -Is) ==="
