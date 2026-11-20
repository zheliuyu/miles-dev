# miles on Ascend 950DT (NPU)

GRPO training on one Ascend 950DT host: the FSDP trainer runs as one rank on one card and drives an external SGLang
server on another. The two jobs are separate processes, each pinned with `ASCEND_RT_VISIBLE_DEVICES`.

## Environment

`miles-950-py312` (Python 3.12.16) with torch 2.10.0+cpu, torch_npu 2.10.0.post7, triton-ascend 3.6.0, SGLang 0.5.23,
`sgl-kernel-npu` 2026.11.0.post4 and CANN 9.2.0 from `/home/pkg/cann920`. Activating the environment sources
`/home/pkg/cann920/ascend-toolkit/set_env.sh` through `etc/conda/activate.d/ascend.sh`:

```bash
conda activate miles-950-py312
```

## Run

One launcher, three subcommands plus a debug pair:

| Command | What it does |
|---|---|
| `prepare` | checks the local checkpoint and dataset (nothing is downloaded when both exist) |
| `sglang` | serves the policy on `--engine-device` in the foreground |
| `train` | trains on `--trainer-device` against that server, `--update-weight-transfer-mode disk-delta` |
| `debug-data` | writes a fixed rollout-data `.pt` from the dataset's first prompts |
| `debug-train` | trains one rollout on that batch with **no** SGLang server (`--load-debug-rollout-data`) |

```bash
# first training check, one card, no engine
python scripts/npu/run_qwen3_5_4b_fsdp.py debug-data
python scripts/npu/run_qwen3_5_4b_fsdp.py debug-train --trainer-device 0

# the closed loop: engine on one card, trainer on another
python scripts/npu/run_qwen3_5_4b_fsdp.py sglang --engine-device 1     # foreground shell
python scripts/npu/run_qwen3_5_4b_fsdp.py train  --trainer-device 0
```

Pick free cards first with `npu-smi info`; the launcher only writes the ids you pass into `ASCEND_RT_VISIBLE_DEVICES`.

## What differs from the CUDA recipe

- `MILES_ACCELERATOR=npu` selects `miles.platforms.npu.NpuPlatform`: eager compile backend, no Triton kernels, no
  flex attention, `disk-delta` weight transfer only, bf16 rollout KV cache, device bound before the process group.
- `--distributed-backend hccl`; no CPU offload, so the backend never becomes `cpu:gloo,npu:hccl`.
- The engine needs `ASCEND_USE_FIA=1`, `--device npu --attention-backend ascend` and `--mamba-ssm-dtype bfloat16`,
  and the 950 `sgl-kernel-npu` wheel for Qwen3.5-4B's GDN and Gemma RMSNorm kernels.
- Ray is not asked for an `NPU` resource: the launcher's `ray start --num-gpus 1` declares a synthetic slot, the same
  way the TPU launcher does. The device masks come from the flags above, not from Ray.
