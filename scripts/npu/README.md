# miles on Ascend NPU

GRPO training on one Ascend 950DT host. The FSDP trainer and an external SGLang server run as separate
processes, each pinned by its own `ASCEND_RT_VISIBLE_DEVICES`. The default is one trainer card and one
engine card. Comma-separated ids run more than one of each: the trainer list is the FSDP world size, and
each engine card is its own `sglang` process. Weights move from trainer to engine through
`--update-weight-transfer-mode disk-delta`.

## Environment

The engine and the trainer share one Python 3.12 environment. The loop was tested with torch 2.10.0+cpu,
torch_npu 2.10.0.post7, torchvision 0.25.0, torchaudio 2.10.0, triton-ascend 3.6.0, SGLang 0.5.23,
sgl-kernel-npu 2026.11.0.post4, transformers 5.19.0, numpy 1.26.4, and CANN 9.2.0.

Activate that environment, and source CANN's `set_env.sh`, before running anything. `import torch` loads
the `torch_npu` backend, which needs the CANN libraries; without them every subcommand fails with
`ImportError: libhccl.so: cannot open shared object file`. `--help` is the exception: the launcher defers
its torch import.

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
- **Two engines share one local checkpoint directory** and both apply the same xor delta. The second pull
  of one version has to be a no-op. Stock SGLang 0.5.23 does not record the applied version, so a 2-engine
  run needs a local `/pull_weights` change in the SGLang install: record the version under the existing
  lock, skip a repeat, and refuse a gap. That change is outside Miles and disappears if SGLang is
  reinstalled. Without it, give each engine its own `--local-checkpoint-dir`.
- **One 4B Adam checkpoint is about 51 GB.** Point `--save-dir` at a filesystem that can hold it. Keep
  `--save-interval` at 3 or higher, and delete old `iter_*` directories after a run.
- **`broadcast` is declared, and one trainer card plus one engine has been measured.** The launcher still
  defaults to `disk-delta`. Pass `--extra-args "--update-weight-transfer-mode broadcast"` to use the HCCL
  weight group. A 2-engine broadcast has not been run.
- **Ray receives synthetic GPU slots.** `ray start --num-gpus N` declares one slot per trainer card.
  `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` keeps Ray from pinning every rank to local id 0. The
  launcher sets `graceful_ray_stop=True`, so cleanup leaves the engine in the other shell running.
- **An engine port already in use** means an old engine is still alive. Stop that process, or pass
  `--sglang-port`.
