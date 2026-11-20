#!/bin/bash
# Acceptance: Gemma RMSNorm on npu:0 + the doc's probe D (SGLang generates on NPU 1).
set -u
ENV=/root/miniconda3/envs/miles-950-py312
export PATH="$ENV/bin:/root/miniconda3/bin:$PATH"
# shellcheck disable=SC1091
source /home/pkg/cann920/ascend-toolkit/set_env.sh
export ASCEND_USE_FIA=1

echo "=== gemma rmsnorm shape check $(date -Is) ==="
export ASCEND_RT_VISIBLE_DEVICES=0
timeout 300 python - <<'PY'
import torch
torch.npu.set_device(0)
from sgl_kernel_npu.norm.gemma_rmsnorm import npu_gemma_rms_norm
x = torch.randn(2, 8, dtype=torch.bfloat16).npu()
w = torch.ones(8, dtype=torch.bfloat16).npu()
out = npu_gemma_rms_norm(x, w)
if isinstance(out, tuple):
    print("returned tuple of", len(out), "->", [getattr(o, "shape", type(o)) for o in out])
    t = out[0]
else:
    print("returned", type(out), out.shape)
    t = out
print("device:", t.device, "absmean:", t.float().abs().mean().item())
PY
echo "gemma exit=$?"

echo "=== probe D: SGLang on NPU 1 $(date -Is) ==="
export ASCEND_RT_VISIBLE_DEVICES=1
python -m sglang.launch_server \
  --model-path /mnt/weight/Qwen3.5-4B \
  --device npu --attention-backend ascend --language-only \
  --dtype bfloat16 --mamba-ssm-dtype bfloat16 \
  --tp-size 1 --mem-fraction-static 0.5 --context-length 2048 \
  --max-running-requests 1 \
  --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled \
  --host 127.0.0.1 --port 31001 > /tmp/probe_d_sglang.log 2>&1 &
SRV=$!
echo "server pid=$SRV"

up=0
for i in $(seq 1 120); do
  sleep 5
  if curl -s -o /dev/null --max-time 3 http://127.0.0.1:31001/health; then up=1; break; fi
  if ! kill -0 $SRV 2>/dev/null; then echo "server died"; break; fi
done
echo "health up=$up after $((i * 5))s"

if [ "$up" = 1 ]; then
  echo "=== generate ==="
  curl -s --max-time 120 http://127.0.0.1:31001/generate \
    -H 'Content-Type: application/json' \
    -d '{"text":"用一句话回答：1+1等于几？","sampling_params":{"max_new_tokens":48,"temperature":0}}'
  echo
else
  echo "=== server log tail ==="
  tail -30 /tmp/probe_d_sglang.log
fi

kill $SRV 2>/dev/null || true
sleep 5
kill -9 $SRV 2>/dev/null || true
echo "=== probe D done $(date -Is) ==="
