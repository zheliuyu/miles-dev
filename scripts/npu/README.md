# miles on Ascend NPU

GRPO training on one Ascend 950DT host. The FSDP trainer and an external SGLang server run as separate
processes, each pinned by its own `ASCEND_RT_VISIBLE_DEVICES`. The default is one trainer card and one
engine card. Comma-separated ids run more than one of each: the trainer list is the FSDP world size, and
each engine card is its own `sglang` process. Weights move from trainer to engine over the HCCL `broadcast`
weight group, which is `NpuPlatform`'s default transfer mode.

## Environment

The image and the packages installed on top of it are in
`docs/hardware-platforms/ascend-npu.md` on branch `docs/ascend-950-support`. The base image is
`ascendai/torch-npu:2.10.0.post7-cann9.2.0-950-openeuler24.03-py3.12` (CANN 9.2.0, Python 3.12,
torch 2.10.0, torch_npu 2.10.0.post7). triton-ascend 3.6.0, SGLang 0.5.23, sgl-kernel-npu
2026.11.0.post4, and Miles are installed afterwards.

Source CANN's `set_env.sh` before running anything. `import torch` loads the `torch_npu` backend,
which needs the CANN libraries; without them every subcommand fails with
`ImportError: libhccl.so: cannot open shared object file`. `--help` is the exception: the launcher
defers its torch import.

## Run

One launcher, `run_qwen3_5_4b_fsdp.py`, for Qwen3.5-4B with FSDP and an external SGLang engine. Paths
are relative to the working directory: `--model-path Qwen3.5-4B`, `--data-path dapo-math-17k.jsonl`,
`--save-dir checkpoints`, `--weight-disk-dir weight-delta`, and `--local-checkpoint-dir
rollout-checkpoint`. `prepare` checks the checkpoint and the prompt file and does not download them.
The steps below use one trainer card and one engine card. Pick cards that `npu-smi info` shows as empty.

1. Check the model and the dataset:

   ```bash
   python scripts/npu/run_qwen3_5_4b_fsdp.py prepare
   ```

2. Start SGLang on one card, in its own shell. It stays in the foreground:

   ```bash
   python scripts/npu/run_qwen3_5_4b_fsdp.py sglang --engine-device 1
   ```

3. Train on another card. The default is 3 rollouts, batch 4, 4 samples per prompt, and a response cap of
   1024:

   ```bash
   python scripts/npu/run_qwen3_5_4b_fsdp.py train --trainer-device 0 --engine-device 1
   ```

   Pass extra `train.py` flags through `--extra-args`. To resume, pass the checkpoint root and a higher
   rollout count. `--num-rollout 4` after a 3-rollout save runs rollout 3:

   ```bash
   python scripts/npu/run_qwen3_5_4b_fsdp.py train --trainer-device 0 --engine-device 1 \
       --num-rollout 4 --extra-args "--load checkpoints"
   ```

Four cards use the same launcher: cards 0 and 1 train, and cards 2 and 3 each run an engine.
`sglang` accepts one card, so start one process per engine card. `train` dials engine `i` at
`--sglang-port + i`, in `--engine-device` order. The default first port is 31001:

```bash
python scripts/npu/run_qwen3_5_4b_fsdp.py sglang --engine-device 2 --sglang-port 31001
python scripts/npu/run_qwen3_5_4b_fsdp.py sglang --engine-device 3 --sglang-port 31002
python scripts/npu/run_qwen3_5_4b_fsdp.py train --trainer-device 0,1 --engine-device 2,3
```

`debug-data` then `debug-train` runs one training check on a fixed batch with no SGLang server. That batch
uses a uniform reward, so `train/loss` and `grad_norm` are 0. The check covers the training loop.

## Notes

- **Pick free cards.** `npu-smi info` shows which cards have processes. The launcher writes only the ids
  you pass into `ASCEND_RT_VISIBLE_DEVICES`. Leave a card that already has a process alone.
- **`broadcast` is the default.** The trainer and the engine build a HCCL weight group; one trainer card
  plus one engine has been measured (`init custom process group ... backend=hccl`, 17 buckets in 1.6 s).
  A 2-engine broadcast has not been run.
- **`disk-delta` is the optional alternative and needs a `/pull_weights` handler in the SGLang install.**
  SGLang 0.5.23 does not ship one. Version 0 copies the HF checkpoint into the engine-local directory, and
  later versions apply the zstd deltas. Two engines that share that directory must record the applied
  version and skip a repeat, or the second pull xors the same delta again. The handler is outside Miles and
  disappears if SGLang is reinstalled. Without it, keep the default `broadcast`.
- **One 4B Adam checkpoint is about 51 GB.** Point `--save-dir` at a filesystem that can hold it. Keep
  `--save-interval` at 3 or higher, and delete old `iter_*` directories after a run.
- **Ray receives synthetic GPU slots.** `ray start --num-gpus N` declares one slot per trainer card.
  `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` keeps Ray from pinning every rank to local id 0. The
  launcher sets `graceful_ray_stop=True`, so cleanup leaves the engine in the other shell running.
- **An engine port already in use** means an old engine is still alive. Stop that process, or pass
  `--sglang-port`.
