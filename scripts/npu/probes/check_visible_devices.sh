#!/bin/bash
# Verify how visible-device selection behaves on this box (Ascend 950DT, CANN 9.2.0).
set -u
ENV=/root/miniconda3/envs/miles-950-py312
export PATH="$ENV/bin:/root/miniconda3/bin:$PATH"
# shellcheck disable=SC1091
source /home/pkg/cann920/ascend-toolkit/set_env.sh

probe() {
  # $1 = label, remaining env already exported by caller
  python - <<'PY'
import os, torch, torch_npu
vis = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "<unset>")
legacy = os.environ.get("ASCEND_VISIBLE_DEVICES", "<unset>")
n = torch.npu.device_count()
print(f"  ASCEND_RT_VISIBLE_DEVICES={vis}  ASCEND_VISIBLE_DEVICES={legacy}")
print(f"  device_count={n}")
if n:
    torch.npu.set_device(0)
    a = torch.ones(2, 2).npu()
    print(f"  logical npu:0 -> tensor on {a.device}, device_name={torch.npu.get_device_name(0)}")
PY
}

echo "=== 1) single card: ASCEND_RT_VISIBLE_DEVICES=2 ==="
ASCEND_RT_VISIBLE_DEVICES=2 probe

echo "=== 2) two cards: ASCEND_RT_VISIBLE_DEVICES=2,3 ==="
ASCEND_RT_VISIBLE_DEVICES=2,3 probe

echo "=== 3) legacy name only: ASCEND_VISIBLE_DEVICES=3 ==="
ASCEND_VISIBLE_DEVICES=3 probe

echo "=== 4) unset (all cards) ==="
probe

echo "=== 5) what Miles launch code expects ==="
grep -rn -E 'ASCEND_RT_VISIBLE_DEVICES' /home/weichunyu/run_qwen3_5_4b_npu.py /home/weichunyu/miles-dev/miles 2>/dev/null | head -5
echo "=== done ==="
