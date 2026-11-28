"""Qwen3.5-4B GRPO on one Ascend 950DT host: FSDP trainer and external SGLang.

The jobs are separate processes that the launcher pins with the platform visible-devices variable. `--trainer-device` and
`--engine-device` each take one card (`0`) or a comma-separated list (`0,1`). One `sglang` process serves one
card; start one process per engine card. Weights return through the platform's default `broadcast` mode, a HCCL
weight group between the trainer and the engine.

Run `prepare` once (it only checks the local model and dataset, no download when they already exist), then `sglang`
in its own shell (it stays in the foreground), then `train`.

Args:
  --model-dir / --model-name: directory holding the local checkpoint and its subdirectory. Default to
    `/mnt/weight/Qwen3.5-4B`, the layout of the 950 host.
  --data-dir: directory holding `dapo-math-17k/dapo-math-17k.jsonl`. Defaults to `/mnt/share/dataset`.
  --trainer-device / --engine-device: physical card ids, written into the platform visible-devices variable.
  --sglang-port: port of the engine `train` connects to.
  --num-rollout: how many rollouts the closed loop runs.
  --save-dir: checkpoint root; defaults to `{--output-dir}/checkpoints`.
  --extra-args: more train.py flags, appended last.

Examples:
  python scripts/npu/run_qwen3_5_4b_fsdp.py sglang --engine-device 1
  python scripts/npu/run_qwen3_5_4b_fsdp.py train --trainer-device 0
"""

import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

import typer

from miles.utils.external_utils import command_utils

SEQ_LEN = 2048
ROLLOUT_BATCH_SIZE = 4
N_SAMPLES_PER_PROMPT = 4
ROLLOUT_MAX_RESPONSE_LEN = 1024
SAVE_INTERVAL = 3
DATASET = "dapo-math-17k"
# NpuPlatform.weight_transfer_modes; broadcast is the only mode this path uses
WEIGHT_TRANSFER_MODE = "broadcast"
# SGLang main gates the weight-update endpoints behind an admin key; the engine and miles' admin
# client both read it from SGLANG_ADMIN_API_KEY, so the launcher exports the same value for both.
ADMIN_API_KEY = os.environ.get("SGLANG_ADMIN_API_KEY", "miles-local-admin")


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    run_id: str = command_utils.create_run_id()
    model_dir: str = "/mnt/weight"
    model_name: str = "Qwen3.5-4B"
    data_dir: str = "/mnt/share/dataset"
    # physical cards; the launcher writes them into the platform visible-devices variable for each job
    trainer_device: str = "0"
    engine_device: str = "1"
    sglang_port: int = 31001
    num_rollout: int = 3
    save_dir: str | None = None
    extra_args: str = ""
    # the engine runs in its own shell; a run must not SIGKILL it, which the backend's default cleanup does
    graceful_ray_stop: bool = True

    def __post_init__(self) -> None:
        self.save_dir = self.save_dir or f"{self.output_dir}/checkpoints"

    @property
    def model_path(self) -> str:
        return f"{self.model_dir}/{self.model_name}"

    @property
    def data_path(self) -> str:
        return f"{self.data_dir}/{DATASET}/{DATASET}.jsonl"


def _require(path: str, what: str) -> None:
    if not Path(path).exists():
        raise typer.BadParameter(f"{what} not found: {path}")


def _cards(value: str) -> list[str]:
    cards = [part.strip() for part in value.split(",") if part.strip()]
    if not cards or any(not card.isdigit() for card in cards):
        raise typer.BadParameter(f"expected comma-separated card ids, got {value!r}")
    return cards


DEVICE_TYPE = "npu"
VISIBLE_DEVICES_ENV = "ASCEND_RT_VISIBLE_DEVICES"
SGLANG_ATTENTION_BACKEND = "ascend"
EXTRA_PROCESS_ENV = {"ASCEND_USE_FIA": "1"}


def _trainer_env(args: ScriptArgs) -> dict[str, str]:
    env = {
        "MILES_ACCELERATOR": DEVICE_TYPE,
        # Ray's GPU slot is synthetic. This mask is the physical cards, in local-rank order.
        VISIBLE_DEVICES_ENV: ",".join(_cards(args.trainer_device)),
        # Ray must not pin CUDA_VISIBLE_DEVICES to one id: that collapses every rank's local id to 0.
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
        # skips the launcher's nvidia-smi probe, exactly as the TPU launcher does
        "NCCL_NVLS_ENABLE": "0",
        **EXTRA_PROCESS_ENV,
        "SGLANG_ADMIN_API_KEY": ADMIN_API_KEY,
    }
    # Ray copies only these variables into the job; config parsing hits the HF hub without them
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if value := os.environ.get(key):
            env[key] = value
    return env


def _dataset_args(args: ScriptArgs) -> str:
    # the rollout executor builds its data source even when the samples come from --load-debug-rollout-data
    return (
        f"--prompt-data {args.data_path} "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        """--apply-chat-template-kwargs '{"enable_thinking":false}' """
        "--rollout-shuffle "
    )


def _recipe_args(
    *,
    rollout_batch_size: int = ROLLOUT_BATCH_SIZE,
    n_samples_per_prompt: int = N_SAMPLES_PER_PROMPT,
    max_response_len: int = ROLLOUT_MAX_RESPONSE_LEN,
) -> str:
    return (
        f"--rollout-batch-size {rollout_batch_size} "
        f"--n-samples-per-prompt {n_samples_per_prompt} "
        f"--rollout-max-response-len {max_response_len} "
        f"--rollout-temperature 1 "
        f"--global-batch-size {rollout_batch_size * n_samples_per_prompt} "
        "--advantage-estimator grpo "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )


def _train_backend_args() -> str:
    return (
        "--train-backend fsdp "
        "--micro-batch-size 1 "
        "--gradient-checkpointing "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {SEQ_LEN} "
        "--log-probs-backend torch "
        "--distributed-backend hccl "
    )


def _misc_args(args: ScriptArgs, *, save_interval: int = SAVE_INTERVAL) -> str:
    trainer_cards = len(_cards(args.trainer_device))
    return (
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {trainer_cards} "
        f"--num-gpus-per-node {trainer_cards} "
        f"--save {args.save_dir} "
        f"--save-interval {save_interval} "
    )


def _serve_sglang(args: ScriptArgs) -> None:
    """Serve the policy on one engine card in the foreground."""
    cards = _cards(args.engine_device)
    if len(cards) != 1:
        raise typer.BadParameter("sglang serves one card; start one process per engine card")
    os.environ[VISIBLE_DEVICES_ENV] = cards[0]
    os.environ.update(EXTRA_PROCESS_ENV)
    os.environ["SGLANG_ADMIN_API_KEY"] = ADMIN_API_KEY
    argv = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--device",
        DEVICE_TYPE,
        "--attention-backend",
        SGLANG_ATTENTION_BACKEND,
    ]
    argv += [
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
        str(SEQ_LEN + ROLLOUT_MAX_RESPONSE_LEN),
        "--max-running-requests",
        str(ROLLOUT_BATCH_SIZE),
        "--cuda-graph-backend-decode",
        "disabled",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.sglang_port),
    ]
    command = " ".join(shlex.quote(part) for part in argv)
    args.create_backend().exec_command_cpu(command)


def _execute(args: ScriptArgs, train_args: str) -> None:
    backend = args.create_backend()
    backend.execute_train(
        train_args=train_args,
        num_gpus_per_node=len(_cards(args.trainer_device)),
        megatron_model_type=None,
        train_script=str(Path(__file__).resolve().parents[2] / "train.py"),
        extra_env_vars=_trainer_env(args),
    )


app = typer.Typer()


@app.command()
@command_utils.dataclass_cli
def prepare(args: ScriptArgs):
    """Check the local checkpoint and dataset; nothing is downloaded when both exist."""
    _require(args.model_path, "model")
    _require(args.data_path, "dataset")
    typer.echo(f"model:   {args.model_path}")
    typer.echo(f"dataset: {args.data_path}")


@app.command()
@command_utils.dataclass_cli
def sglang(args: ScriptArgs):
    """Serve the policy on --engine-device in the foreground."""
    _serve_sglang(args)


@app.command()
@command_utils.dataclass_cli
def train(args: ScriptArgs):
    """Train on --trainer-device against the SGLang server on --engine-device."""
    _require(args.model_path, "model")
    _require(args.data_path, "dataset")
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    engine_cards = _cards(args.engine_device)
    engine_addrs = " ".join(f"http://127.0.0.1:{args.sglang_port + index}" for index in range(len(engine_cards)))
    train_args = " ".join(
        [
            f"--hf-checkpoint {args.model_path}",
            _dataset_args(args),
            "--rm-type math",
            f"--num-rollout {args.num_rollout}",
            _recipe_args(),
            _train_backend_args(),
            f"{command_utils.get_default_wandb_args(__file__, run_id=args.run_id)}",
            f"--rollout-external-engine-addrs {engine_addrs}",
            f"--rollout-num-gpus {len(engine_cards)}",
            "--rollout-num-gpus-per-engine 1",
            f"--update-weight-transfer-mode {WEIGHT_TRANSFER_MODE}",
            _misc_args(args),
            args.extra_args,
        ]
    )
    _execute(args, train_args)


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
