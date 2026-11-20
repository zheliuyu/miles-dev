#!/bin/bash
# 探针 B 启动器（真正执行 probe_b_run.py）
set -u
ENV=/root/miniconda3/envs/miles-950-py312
export PATH="$ENV/bin:/root/miniconda3/bin:$PATH"
# shellcheck disable=SC1091
source /home/pkg/cann920/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1
export HCCL_CONNECT_TIMEOUT=120
export HCCL_EXEC_TIMEOUT=120
echo "=== 探针 B start $(date -Is) visible=$ASCEND_RT_VISIBLE_DEVICES ==="
timeout 900 python /tmp/probe_b_run.py
rc=$?
echo "=== 探针 B exit=$rc $(date -Is) ==="
