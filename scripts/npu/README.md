# miles on Ascend NPU

GRPO on one Ascend 950DT host: the FSDP trainer and an external SGLang server run as separate
processes, pinned with `ASCEND_RT_VISIBLE_DEVICES`. Weights reach the engine over an HCCL
`broadcast` group. One `sglang` process serves one engine card; the `--trainer-device` list is the
FSDP world size.

## Prerequisites

- One Ascend 950DT host with the driver under `/usr/local/Ascend/driver` and DCMI under
  `/usr/local/dcmi`; the CANN toolkit ships in the image.
- Docker with `--privileged` support, run as root or a member of the `docker` group.
- At least two idle cards (`npu-smi info` on the host), or four for the 4-card example.
- About 20 GB for the image and 51 GB per checkpoint.
- `Qwen/Qwen3.5-4B` under `--model-dir` (default `/mnt/weight`) and `zhuzilin/dapo-math-17k`
  under `--data-dir` (default `/mnt/share/dataset`):

  ```bash
  hf download Qwen/Qwen3.5-4B --local-dir /mnt/weight/Qwen3.5-4B
  hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /mnt/share/dataset/dapo-math-17k
  ```

  If they live elsewhere, mount that directory and pass the same path with `--model-dir` /
  `--data-dir`.

`run_qwen3_4b_megatron.py` is the older Megatron/MindSpeed entry point for the A2/A3 stack; it
needs the patches under `docker/npu_patch/` and is separate from the FSDP flow below.

## Environment

Start from the 950 image, which carries torch 2.13.0, torch_npu 2.13.0, CANN 9.2.0 and Python 3.12:
`ascendai/torch-npu:2.13.0-cann9.2.0-950-openeuler24.03-py3.12`. Install on top:

| Package | Version / source |
|---|---|
| torchvision | 0.28.0 (torch 2.13's pair) |
| triton / triton-ascend | 3.6.0 / 3.6.0; the Ascend wheel carries the patched `libtriton.so` and is installed after triton |
| sgl-kernel-npu wheel | `2026.11.0.post15-torch2.13.0-py312-cann9.2.0-950-aarch64` |
| SGLang | main or nightly containing `sgl-project/sglang#51818`, pinned at `8a834e3` |
| miles | `pip install -e .` from this checkout |

torch/torch_npu below 2.13 do not provide `torch.accelerator`'s memory APIs, which this target
calls; keep the 2.13 pair.

## Container

Pull the base image, then build from the repository root. The build needs PyPI, the Ascend PyPI
index, the PyTorch CPU index and GitHub (or mirrors for all four):

```bash
docker pull ascendai/torch-npu:2.13.0-cann9.2.0-950-openeuler24.03-py3.12
docker build -f docker/Dockerfile.npu -t miles-npu:2.13 .
```

Pass `--build-arg PIP_INDEX_URL=...` (and `ASCEND_INDEX` / `TORCH_CPU_INDEX` / `SGLANG_REPO` /
`SGL_KERNEL_ZIP`) when the default endpoints are unreachable. Later rebuilds reuse the cached
layers; only the final checkout layer is rebuilt after a code change.

Run it with the 950 host runtime mounted; the Ascend container needs `--privileged` plus the
driver, DCMI and install-info mounts:

```bash
docker run -it --rm --name miles-npu \
  --privileged \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /mnt/weight:/mnt/weight \
  -v /mnt/share/dataset:/mnt/share/dataset \
  -v "$HOME/miles-checkpoints:/checkpoints" \
  -v "$PWD":/workspace/miles -w /workspace/miles \
  --ulimit nofile=1048576:1048576 \
  --ipc=host --network host \
  miles-npu:2.13 bash
```

Inside the container, source CANN once and follow the Run steps:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

The image already has miles; if the checkout changed after the build, run `pip install -e .` once.
Without HuggingFace access, export `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; the launcher
forwards both into the Ray job.

## Run

`run_qwen3_5_4b_fsdp.py` is the launcher. Defaults: checkpoint `/mnt/weight/Qwen3.5-4B`
(`--model-dir` + `--model-name`), prompts `/mnt/share/dataset/dapo-math-17k/dapo-math-17k.jsonl`
(`--data-dir`). The launcher's `--save-dir` defaults to `{--output-dir}/checkpoints`, which is
inside the container; pass `--save-dir /checkpoints` so the mounted host directory holds the
51 GB checkpoint. The steps use one trainer card and one engine card.

1. On the host, check for idle cards (`npu-smi info`). In the container, source CANN (see above)
   and check the stack:

   ```bash
   python -c 'import torch, torch_npu, sglang, miles; print(torch.__version__, torch.npu.is_available())'
   ```

2. Start Ray before the engine; `train` reuses it when `MILES_SCRIPT_EXTERNAL_RAY=1` is set.
   Starting Ray while the engine runs can abort it ("An unexpected system state"):

   ```bash
   ray start --head --node-ip-address 127.0.0.1 --num-gpus 1 --disable-usage-stats
   ```

3. Check the model and the dataset:

   ```bash
   python scripts/npu/run_qwen3_5_4b_fsdp.py prepare
   ```

4. Serve the policy on one card, in its own shell. The launcher exports `SGLANG_ADMIN_API_KEY`
   (default `miles-local-admin`) to the engine and the trainer; export your own in both shells to
   override:

   ```bash
   python scripts/npu/run_qwen3_5_4b_fsdp.py sglang --engine-device 1
   ```

5. Train on another card: 3 rollouts, batch 4, 4 samples per prompt, response cap 1024.
   `MILES_SCRIPT_EXTERNAL_RAY=1` makes the launcher reuse the Ray cluster instead of restarting it.

   ```bash
   MILES_SCRIPT_EXTERNAL_RAY=1 python scripts/npu/run_qwen3_5_4b_fsdp.py train --trainer-device 0 --engine-device 1 \
       --save-dir /checkpoints
   ```

   wandb is on when `WANDB_API_KEY` is set. Extra `train.py` flags go through `--extra-args`. To
   resume from the mounted save root after a 3-rollout save:

   ```bash
   MILES_SCRIPT_EXTERNAL_RAY=1 python scripts/npu/run_qwen3_5_4b_fsdp.py train --trainer-device 0 --engine-device 1 \
       --num-rollout 4 --save-dir /checkpoints --extra-args "--load /checkpoints"
   ```

Four cards: trainers 0,1 and one engine each on 2,3. Start Ray with one slot per trainer card;
`train` dials engine `i` at `--sglang-port + i`:

```bash
ray start --head --node-ip-address 127.0.0.1 --num-gpus 2 --disable-usage-stats
python scripts/npu/run_qwen3_5_4b_fsdp.py sglang --engine-device 2 --sglang-port 31001
python scripts/npu/run_qwen3_5_4b_fsdp.py sglang --engine-device 3 --sglang-port 31002
MILES_SCRIPT_EXTERNAL_RAY=1 python scripts/npu/run_qwen3_5_4b_fsdp.py train --trainer-device 0,1 --engine-device 2,3 \
    --save-dir /checkpoints
```

## Notes

- **Pick idle cards on the host.** Run `npu-smi info` before `docker run`; the container does not
  carry the host's `npu-smi`, and the launcher writes only the ids you pass into
  `ASCEND_RT_VISIBLE_DEVICES`.
- **`broadcast` is the only weight path.** `disk-delta` needs an SGLang `/pull_weights` receiver that
  these builds do not run.
- **A 4B Adam checkpoint is ~51 GB.** Mount a host directory at `/checkpoints` and pass
  `--save-dir /checkpoints`; the launcher saves every 3 rollouts, so delete old `iter_*` directories.
- **Ray slots are synthetic.** `--num-gpus N` declares one slot per trainer card;
  `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` keeps ranks from collapsing to local id 0.
- **A busy engine port** means an old engine is still running; stop it or pass `--sglang-port`.

## Troubleshooting

- `SetDevice ... TsdOpen failed`: rerun with `--privileged` and the driver, DCMI and
  `/etc/ascend_install.info` mounts.
- `epoll: Too many open files`: keep the container's `--ulimit nofile=1048576:1048576`.
- Weight updates answer 403: export the same `SGLANG_ADMIN_API_KEY` in the engine and train shells.
- The engine fails on the Qwen3.5 import or HuggingFace errors: export `HF_HUB_OFFLINE=1` and
  `TRANSFORMERS_OFFLINE=1`, and check the model path.
