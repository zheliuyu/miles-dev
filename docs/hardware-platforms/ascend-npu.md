---
title: Ascend NPUs
description: Run Miles on one Ascend 950DT host with the FSDP trainer and an external SGLang engine.
---
Miles runs a Qwen3.5-4B GRPO loop on one Ascend 950DT host. The FSDP trainer and the SGLang
rollout engine are separate processes on separate cards. Weights move trainer to engine with
`--update-weight-transfer-mode disk-delta`. Megatron-LM, colocated CUDA IPC, and the in-tree
`docker/npu_patch/` tree are not this path. The launcher is
[`scripts/npu/run_qwen3_5_4b_fsdp.py`](https://github.com/radixark/miles/blob/main/scripts/npu/run_qwen3_5_4b_fsdp.py).

This is one host and one model, not a claim that every Ascend generation runs Miles.

## Stack

The loop below was run with Python 3.12, torch 2.10.0+cpu, torch_npu 2.10.0.post7, CANN 9.2.0,
SGLang 0.5.23, sgl-kernel-npu 2026.11.0.post4, and transformers 5.19.0. Activating the environment
must source the CANN `set_env.sh` before `import torch`, or HCCL fails to load. The host-specific
paths, the conda environment, and the card-picking rules are in
[`scripts/npu/README.md`](https://github.com/radixark/miles/blob/main/scripts/npu/README.md).

## Launch training

`prepare` checks the local Hugging Face checkpoint and the prompt file. `sglang` serves one card
in the foreground. `train` starts Ray and the FSDP trainer against that server. Pick cards that
`npu-smi info` shows as empty. Do not stop another user's process to free a card.

```bash
python scripts/npu/run_qwen3_5_4b_fsdp.py prepare
python scripts/npu/run_qwen3_5_4b_fsdp.py sglang --engine-device 1          # keep this shell open
python scripts/npu/run_qwen3_5_4b_fsdp.py train  --trainer-device 0
```

The defaults are 3 rollouts, batch 4, 4 samples per prompt, response cap 1024, and a checkpoint
every 3 rollouts. One 4B Adam checkpoint is about 51 GB. Point `--save-dir`,
`--weight-disk-dir`, and `--local-checkpoint-dir` at a filesystem that can hold that, not at a
small root disk.

To resume, pass the checkpoint root and a higher rollout count. `--num-rollout 4` after a 3-rollout
save runs rollout 3:

```bash
python scripts/npu/run_qwen3_5_4b_fsdp.py train --trainer-device 0 --engine-device 1 \
    --num-rollout 4 --extra-args "--load /path/to/checkpoints"
```

## Four cards

`--trainer-device` and `--engine-device` take comma-separated card ids. The trainer count is the
FSDP world size. `sglang` still accepts one card, so start one process per engine card. The train
command addresses engine `i` at `--sglang-port + i`, in the order of `--engine-device`.

```bash
python scripts/npu/run_qwen3_5_4b_fsdp.py sglang --engine-device 5 --sglang-port 31001
python scripts/npu/run_qwen3_5_4b_fsdp.py sglang --engine-device 6 --sglang-port 31002
python scripts/npu/run_qwen3_5_4b_fsdp.py train --trainer-device 3,7 --engine-device 5,6
```

Both engines receive the same `--update-weight-local-checkpoint-dir` and apply the same xor delta.
The second pull of one version must be a no-op. Stock SGLang 0.5.23 does not record that, so two
engines on one shared directory need a local change under `/pull_weights`: remember the applied
version and skip a repeat. That change lives in the SGLang install, not in Miles, and a reinstall
drops it. Give each engine its own local directory if you do not have it.

A 2-trainer plus 2-engine disk-delta run on this stack finished 3 rollouts and one resume at
`world_size` 2. Most answers still hit the 1024 cap and the math reward stayed near 0. That checks
the loop, not that the policy improved.

## Weight transfer

The launcher defaults to `disk-delta`. That is the path measured for both the 2-card loop and the
4-card loop. `NpuPlatform` also declares `broadcast`. On one trainer card and one engine the HCCL
group came up (`backend=hccl`), the engine accepted the tensors, and the following rollout and
checkpoint completed. Pass it explicitly:

```bash
python scripts/npu/run_qwen3_5_4b_fsdp.py train --trainer-device 0 --engine-device 1 \
    --extra-args "--update-weight-transfer-mode broadcast"
```

A 2-engine broadcast has not been run. Colocate stays off: it depends on CUDA IPC.
