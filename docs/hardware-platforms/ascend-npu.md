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

## Image

There is no Miles image for Ascend. Start from the TorchNPU image that matches the stack this loop
was run on: Ascend 950, arm64, openEuler 24.03, Python 3.12, CANN 9.2.0, torch 2.10.0, and
torch_npu 2.10.0.post7.

```bash
docker pull ascendai/torch-npu:2.10.0.post7-cann9.2.0-950-openeuler24.03-py3.12
```

`ascendai/torch-npu:2.10.0.post7-cann9.2.0-950-ubuntu22.04-py3.12` is the same stack on Ubuntu 22.04.
The image does not include triton-ascend, SGLang, sgl-kernel-npu, or Miles.

## Start the container

On the host, pass one `/dev/davinciN` for each card the jobs will use. The example below is the
2-card layout, trainer on card 0 and the engine on card 1:

```bash
docker run --rm \
  --device /dev/davinci0 --device /dev/davinci1 \
  --device /dev/davinci_manager --device /dev/hisi_hdc --device /dev/devmm_svm \
  --device /dev/ummu --device /dev/uburma \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  --shm-size 128G --network=host \
  -it ascendai/torch-npu:2.10.0.post7-cann9.2.0-950-openeuler24.03-py3.12 /bin/bash
```

**Everything from here on runs inside the container.**

## Install the rest

Source CANN before installing or importing torch. Without `set_env.sh`, `import torch` fails with
`ImportError: libhccl.so: cannot open shared object file`.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python -m pip install torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install triton-ascend==3.6.0 \
  --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple
python -m pip install --no-deps sglang==0.5.23

wget https://github.com/sgl-project/sgl-kernel-npu/releases/download/2026.11.0.post10/sgl-kernel-npu-2026.11.0.post10-torch2.10.0-py312-cann9.1.0-950-aarch64.zip
unzip sgl-kernel-npu-2026.11.0.post10-torch2.10.0-py312-cann9.1.0-950-aarch64.zip
python -m pip install --no-deps sgl_kernel_npu*.whl
```

That kernel archive is labeled CANN 9.1.0. The wheel inside is sgl-kernel-npu 2026.11.0.post4, and
it loaded against this image's CANN 9.2.0. Install Miles from a checkout of this repo:

```bash
python -m pip install -e /path/to/miles
python -m pip install transformers==5.19.0 "numpy==1.26.4"
```

SGLang 0.5.23 has no disk-delta receiver. The default weight path needs a `/pull_weights` handler in
that install: version 0 copies the HF checkpoint into the engine-local directory, and later versions
apply the published zstd deltas. Miles then calls the stock `/update_weights_from_disk`. Two engines
that share one local directory also have to record the applied version and skip a repeat, or the
second pull xors the same delta again. That handler is not part of Miles, and reinstalling SGLang
removes it. `broadcast` does not use it.

## Verify

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.is_available())"
npu-smi info
```

`npu-smi info` shows which cards already have a process. The launcher writes only the ids you pass
into `ASCEND_RT_VISIBLE_DEVICES`. Leave a card that already has a process alone.

## Launch training

Paths are relative to the working directory: `--model-path Qwen3.5-4B`, `--data-path
dapo-math-17k.jsonl`, `--save-dir checkpoints`, `--weight-disk-dir weight-delta`, and
`--local-checkpoint-dir rollout-checkpoint`. `prepare` checks the checkpoint and the prompt file and
does not download them. One 4B Adam checkpoint is about 51 GB, so run from a filesystem that can hold
it. Pick cards that `npu-smi info` shows as empty.

```bash
python scripts/npu/run_qwen3_5_4b_fsdp.py prepare
python scripts/npu/run_qwen3_5_4b_fsdp.py sglang --engine-device 1    # keep this shell open
python scripts/npu/run_qwen3_5_4b_fsdp.py train --trainer-device 0 --engine-device 1
```

The defaults are 3 rollouts, batch 4, 4 samples per prompt, and a response cap of 1024. To resume
after that save, raise the rollout count. `--num-rollout 4` runs rollout 3:

```bash
python scripts/npu/run_qwen3_5_4b_fsdp.py train --trainer-device 0 --engine-device 1 \
    --num-rollout 4 --extra-args "--load checkpoints"
```

Four cards use the same launcher: cards 0 and 1 train, and cards 2 and 3 each run an engine. `sglang`
accepts one card, so start one process per engine. Add those cards' `/dev/davinciN` devices when you
start the container. `train` dials engine `i` at `--sglang-port + i`.

```bash
python scripts/npu/run_qwen3_5_4b_fsdp.py sglang --engine-device 2 --sglang-port 31001
python scripts/npu/run_qwen3_5_4b_fsdp.py sglang --engine-device 3 --sglang-port 31002
python scripts/npu/run_qwen3_5_4b_fsdp.py train --trainer-device 0,1 --engine-device 2,3
```

`disk-delta` is the measured path for the 2-card loop and the 4-card loop. `broadcast` was measured
with one trainer card and one engine. Pass `--extra-args "--update-weight-transfer-mode broadcast"`
to use it. A 2-engine broadcast has not been run. Colocate stays off: it depends on CUDA IPC.

## Next steps

- [`scripts/npu/README.md`](https://github.com/radixark/miles/blob/main/scripts/npu/README.md) — card selection, the four-card ports, and the operating notes.
