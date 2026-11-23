# miles on Ascend 950DT (NPU)

GRPO training on one Ascend 950DT host. The FSDP trainer runs as one rank on one card and drives an **external**
SGLang server on another card. The two jobs are separate processes, each pinned by its own
`ASCEND_RT_VISIBLE_DEVICES`; weights travel trainer → engine through `--update-weight-transfer-mode disk-delta`
(the only mode `NpuPlatform` declares).

This mirrors `scripts/tpu/`, which does the same thing on TPU with an external engine and ICI weight sync.

## Environment

`miles-950-py312` (Python 3.12.16, `/root/miniconda3/envs/miles-950-py312`) with torch 2.10.0+cpu, torch_npu
2.10.0.post7, torchvision 0.25.0, torchaudio 2.10.0, triton-ascend 3.6.0, SGLang 0.5.23, `sgl-kernel-npu`
2026.11.0.post4, transformers 5.19.0, numpy 1.26.4 and CANN 9.2.0 from `/home/pkg/cann920`.

Activating the environment sources `/home/pkg/cann920/ascend-toolkit/set_env.sh` through
`etc/conda/activate.d/ascend.sh` (`ASCEND_HOME_PATH` ends up as `/home/pkg/cann920/cann-9.2.0`):

```bash
conda activate miles-950-py312
```

**Activate before running anything.** `import torch` auto-loads the `torch_npu` backend, which needs the CANN
libraries; without the activation every subcommand fails with
`ImportError: libhccl.so: cannot open shared object file`. `--help` is the one exception (the launcher defers its
torch import).

## Inputs and defaults

The defaults point at data that already exists on this host, so `prepare` normally downloads nothing:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model-path` | `/mnt/weight/Qwen3.5-4B` | local HF checkpoint; 4.21B params, bf16 |
| `--data-path` | `/mnt/share/dataset/dapo-math-17k/dapo-math-17k.jsonl` | prompt dataset for the rollout data source |
| `--trainer-device` | `0` | physical Ascend card for the FSDP trainer |
| `--engine-device` | `1` | physical Ascend card for the SGLang server |
| `--sglang-port` | `31001` | engine port; `train` connects to `http://127.0.0.1:<port>` |
| `--seq-length` | `2048` | becomes `--max-tokens-per-gpu` and sizes the engine's context |
| `--num-rollout` | `3` | closed-loop rollouts |
| `--rollout-batch-size` / `--n-samples-per-prompt` | `4` / `4` | 16 samples per rollout |
| `--save-dir` | `/home/weichunyu/miles-950/checkpoints` | checkpoint root |
| `--save-interval` | `3` | save every N rollouts (see **Disk**) |
| `--weight-disk-dir` | `/home/weichunyu/miles-950/weight-delta` | shared disk-delta directory; `train` passes it as `--update-weight-disk-dir` |
| `--local-checkpoint-dir` | `/home/weichunyu/miles-950/rollout-checkpoint` | engine-local checkpoint; `train` passes it as `--update-weight-local-checkpoint-dir` |
| `--debug-data-dir` | `/home/weichunyu/miles-950/debug-rollout` | where `debug-data` writes `0.pt` |
| `--extra-args` | empty | extra `train.py` flags, appended last |

## Run

| Command | What it does |
| --- | --- |
| `prepare` | checks the local checkpoint and dataset (nothing is downloaded when both exist) |
| `sglang` | serves the policy on `--engine-device` in the foreground |
| `train` | trains on `--trainer-device` against that server, `disk-delta` weight sync |
| `debug-data` | writes a fixed rollout-data `.pt` from the dataset's first prompts |
| `debug-train` | trains one rollout on that batch with **no** SGLang server (`--load-debug-rollout-data`) |

```bash
# 1. first training check: one card, no engine (~2.5 min, 51 GB checkpoint)
python scripts/npu/run_qwen3_5_4b_fsdp.py debug-data
python scripts/npu/run_qwen3_5_4b_fsdp.py debug-train --trainer-device 1

# 2. the closed loop: engine on one card, trainer on another
python scripts/npu/run_qwen3_5_4b_fsdp.py sglang --engine-device 7     # keep this shell open
python scripts/npu/run_qwen3_5_4b_fsdp.py train  --trainer-device 1    # other shell
```

Pick free cards first:

```bash
npu-smi info            # cards with processes are taken; do not kill other runs
```

The launcher only writes the ids you pass into `ASCEND_RT_VISIBLE_DEVICES`; it never picks cards itself.

## What to expect

- **Engine**: starts in ~50 s (probe D) once the 4B weights load; `sglang` stays in the foreground.
- **Trainer**: Ray comes up first, then the FSDP actor loads the checkpoint; the debug run needed ~2.5 min wall for
  4 optimizer steps (`actor_train` 20.5 s, ~4–6 s per step with gradient checkpointing).
- **Debug loss is 0.0.** GRPO requires one reward per rollout and the generated batch uses a uniform reward, so the
  advantage and both `train/loss` and `grad_norm` are 0. That is expected: the debug run proves the loop (forward,
  backward, optimizer, checkpoint), not the reward signal. `train/train_rollout_logprob_abs_diff` and
  `train/train_rollout_kl` *are* non-zero (≈5.5 and ≈4.2) because the generated `.pt` carries placeholder
  `rollout_log_probs` while the trainer computes real ones; ignore those two until the engine supplies them.
- **Checkpoint**: `iter_<n>/` holds `model/` (20 GB), `optimizer/` (32 GB), `lr_scheduler/`, `rng.pt`,
  `meta.json` (`iteration`, `next_rollout_id`, `world_size`).

## Disk

One 4B checkpoint costs **~51 GB** (Adam keeps fp32 exp-avg/exp-avg-sq). The root filesystem had ~122 GB free, so:

- keep `--save-interval` at 3 or higher, and delete old `iter_*` directories after a run;
- a debug run also drops its own checkpoint; remove it if the next step does not need to resume.

## What differs from the CUDA recipe

- `MILES_ACCELERATOR=npu` selects `miles.platforms.npu.NpuPlatform`: eager compile backend, no Triton kernels, no
  flex attention, `disk-delta` weight transfer only, bf16 rollout KV cache, and the device is bound *before* the
  process group because HCCL reads it at `init_process_group` (TPU binds after).
- `--distributed-backend hccl`; CPU offload stays off, so the backend never becomes `cpu:gloo,npu:hccl`.
- The engine needs `ASCEND_USE_FIA=1`, `--device npu --attention-backend ascend`, `--mamba-ssm-dtype bfloat16`
  (Mamba state must match the bf16 keys, otherwise `chunk_gated_delta_rule` rejects it), and the 950
  `sgl-kernel-npu` wheel for Qwen3.5-4B's GDN and Gemma RMSNorm kernels.
- **Ray is not asked for an `NPU` resource.** The launcher's `ray start --num-gpus 1` declares a synthetic GPU slot,
  exactly as the TPU launcher does, so `miles/ray/*` needs no change; the price is that Ray no longer sets
  `ASCEND_RT_VISIBLE_DEVICES` per worker, which is why each job sets its own mask.
- The launcher passes `graceful_ray_stop=True`: the backend's default cleanup runs `pkill -9 sglang`, which would
  kill the engine started in the other shell.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `ImportError: libhccl.so: cannot open shared object file` | the environment is not activated; `conda activate miles-950-py312` |
| `pyarrow requires NumPy 2.0 or newer, found 1.26.4` | env drift; `uv pip install 'pyarrow==20.0.0' 'datasets<5'` (triton-ascend pins numpy 1.26.4) |
| `all samples in a rollout must share one reward` | the debug `.pt` mixes rewards; regenerate with `debug-data` |
| `unrecognized arguments: --seq-length` | that is a torchtitan flag; the FSDP backend takes `--max-tokens-per-gpu` |
| Ray job hangs before any log | the cluster has no `GPU` resource; use the launcher (it starts Ray with `--num-gpus`), not a bare `ray start` |
| `Address already in use` on the engine port | an old engine is alive; `pkill -f sglang.launch_server` or pass `--sglang-port` |

## Local records (delete before the PR)

These are kept for the audit trail of the first 950DT run, not for upstream:

| Path | What |
| --- | --- |
| `docs/hardware-platforms/ascend-npu-run-record.md` | the full run record: hardware facts, why the in-tree NPU patch is unusable, the chosen route, probes, commits, stage-4 and stage-5 results, remaining plan |
| `scripts/npu/probes/` | the probe A/B/C/D scripts and the debug-training launcher as they were run |
| `scripts/npu/legacy/run_qwen3_5_4b_npu.py` | the earlier draft launcher (previous host), superseded by `run_qwen3_5_4b_fsdp.py` |
| `scripts/npu/evidence/` | build logs, probe logs, and the stage-4 checkpoint metadata plus step metrics |

Before opening the PRs in §7 of the run record, delete this section, `probes/`, `legacy/`, `evidence/`, and the run record itself.
