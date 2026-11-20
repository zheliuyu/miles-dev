"""Qwen3.5-4B GRPO on one Ascend 950DT host: FSDP on NPU 0, SGLang on NPU 1.

The first loop is 3 rollouts of 16 samples. Eval stays off. Weights return through disk-delta.

    conda activate miles-950-py312
    python /home/weichunyu/run_qwen3_5_4b_npu.py sglang
    python /home/weichunyu/run_qwen3_5_4b_npu.py train
"""

import os
import sys
from pathlib import Path

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig

MODEL_PATH = "/mnt/weight/Qwen3.5-4B"
DATA_PATH = "/mnt/share/dataset/dapo-math-17k/dapo-math-17k.jsonl"
RUN_DIR = Path("/home/weichunyu/miles-950")
DISK_DIR = RUN_DIR / "weight-delta"
LOCAL_CHECKPOINT_DIR = RUN_DIR / "rollout-checkpoint"
SAVE_DIR = RUN_DIR / "checkpoints"
SGLANG_PORT = 31001
TRAIN_SCRIPT = "/home/weichunyu/miles-dev/train.py"


def _use_this_interpreter() -> None:
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")


def sglang() -> None:
    """Serve the policy on NPU 1. This process stays in the foreground."""
    _use_this_interpreter()
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "1"
    os.environ["ASCEND_USE_FIA"] = "1"
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            MODEL_PATH,
            "--device",
            "npu",
            "--attention-backend",
            "ascend",
            "--language-only",
            "--dtype",
            "bfloat16",
            "--mamba-ssm-dtype",
            "bfloat16",
            "--tp-size",
            "1",
            "--mem-fraction-static",
            "0.5",
            "--context-length",
            "4096",
            "--max-running-requests",
            "4",
            "--cuda-graph-backend-decode",
            "disabled",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--host",
            "127.0.0.1",
            "--port",
            str(SGLANG_PORT),
        ],
    )


def train() -> None:
    """Train on NPU 0 against the SGLang server."""
    _use_this_interpreter()
    for path in (DISK_DIR, LOCAL_CHECKPOINT_DIR, SAVE_DIR):
        path.mkdir(parents=True, exist_ok=True)
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
    os.environ["MILES_ACCELERATOR"] = "npu"
    # skips the launcher's nvidia-smi probe
    os.environ["NCCL_NVLS_ENABLE"] = "0"

    train_args = " ".join(
        [
            f"--hf-checkpoint {MODEL_PATH}",
            f"--prompt-data {DATA_PATH}",
            "--input-key prompt",
            "--label-key label",
            "--apply-chat-template",
            # a thinking trace often fills 1024 tokens before the boxed answer, so every reward is 0
            """--apply-chat-template-kwargs '{"enable_thinking":false}'""",
            "--rollout-shuffle",
            "--rm-type math",
            "--num-rollout 3",
            "--rollout-batch-size 4",
            "--n-samples-per-prompt 4",
            "--rollout-max-response-len 1024",
            "--rollout-temperature 1",
            "--global-batch-size 16",
            "--advantage-estimator grpo",
            "--entropy-coef 0.00",
            "--eps-clip 0.2",
            "--eps-clip-high 0.28",
            "--optimizer adam",
            "--lr 1e-6",
            "--lr-decay-style constant",
            "--weight-decay 0.1",
            "--adam-beta1 0.9",
            "--adam-beta2 0.98",
            "--log-probs-backend torch",
            f"--rollout-external-engine-addrs http://127.0.0.1:{SGLANG_PORT}",
            "--rollout-num-gpus 1",
            "--rollout-num-gpus-per-engine 1",
            "--update-weight-transfer-mode disk-delta",
            f"--update-weight-disk-dir {DISK_DIR}",
            f"--update-weight-local-checkpoint-dir {LOCAL_CHECKPOINT_DIR}",
            "--train-backend fsdp",
            "--attn-implementation flash_attention_2",
            "--distributed-backend hccl",
            "--micro-batch-size 1",
            "--gradient-checkpointing",
            "--use-dynamic-batch-size",
            "--max-tokens-per-gpu 4096",
            "--actor-num-nodes 1",
            "--actor-num-gpus-per-node 1",
            "--num-gpus-per-node 1",
            f"--save {SAVE_DIR}",
            "--save-interval 3",
        ]
    )

    config = ExecuteTrainConfig(graceful_ray_stop=True)
    config.create_backend().execute_train(
        train_args=train_args,
        num_gpus_per_node=1,
        megatron_model_type=None,
        train_script=TRAIN_SCRIPT,
        extra_env_vars={
            "MILES_ACCELERATOR": "npu",
            "ASCEND_RT_VISIBLE_DEVICES": "0",
            "NCCL_NVLS_ENABLE": "0",
        },
    )


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ("sglang", "train"):
        raise SystemExit("usage: run_qwen3_5_4b_npu.py {sglang|train}")
    {"sglang": sglang, "train": train}[sys.argv[1]]()


if __name__ == "__main__":
    main()
