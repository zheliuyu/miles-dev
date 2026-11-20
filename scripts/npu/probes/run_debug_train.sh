#!/bin/bash
# 阶段 4 首次尝试：debug-train，只占 NPU 1，不接 SGLang。
set -u
cd /home/weichunyu/miles-dev
export PATH=/root/miniconda3/envs/miles-950-py312/bin:$PATH
# shellcheck disable=SC1091
source /home/pkg/cann920/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=1
echo "=== stage4 debug-train start $(date -Is) ==="
echo "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
python scripts/npu/run_qwen3_5_4b_fsdp.py debug-train --trainer-device 1
rc=$?
echo "=== stage4 debug-train exit=$rc $(date -Is) ==="
