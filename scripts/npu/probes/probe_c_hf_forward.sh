#!/bin/bash
# 探针 C 启动器
set -u
ENV=/root/miniconda3/envs/miles-950-py312
export PATH="$ENV/bin:/root/miniconda3/bin:$PATH"
# shellcheck disable=SC1091
source /home/pkg/cann920/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export ASCEND_USE_FIA=1
echo "=== 探针 C start $(date -Is) ==="
timeout 1800 python /tmp/probe_c_run.py
rc=$?
echo "=== 探针 C exit=$rc $(date -Is) ==="
