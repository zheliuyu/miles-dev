"""Qwen3.5-4B GRPO on one Ascend 950DT host: FSDP trainer on one card, SGLang on another.

The two jobs are separate processes that the launcher pins with ASCEND_RT_VISIBLE_DEVICES: the trainer sees only
`--trainer-device` and drives an external SGLang engine on `--engine-device`. Weights return through the disk-delta
protocol (NpuPlatform.weight_transfer_modes), the same way the TPU launcher returns them over ICI.

Run `prepare` once (it only checks the local model and dataset, no download when they already exist), then `sglang`
in its own shell (it stays in the foreground), then `train`. `debug-data` + `debug-train` run the FSDP trainer alone
on a fixed token batch, with no SGLang server, for a first training check.

Args:
  --model-path: local Qwen3.5-4B checkpoint (defaults to the weights already on this host).
  --data-path: dapo-math-17k jsonl.
  --trainer-device / --engine-device: physical Ascend card ids, written into ASCEND_RT_VISIBLE_DEVICES.
  --sglang-port: port of the engine `train` connects to.
  --num-rollout / --rollout-batch-size / --n-samples-per-prompt: the closed-loop workload.
  --seq-length: trainer sequence length; 2048 keeps the first closed loop inside one card.
  --debug-data-dir: where `debug-data` writes `{rollout_id}.pt` and `debug-train` reads it.
  --extra-args: more train.py flags, appended last.

Examples:
  python scripts/npu/run_qwen3_5_4b_fsdp.py debug-data
  python scripts/npu/run_qwen3_5_4b_fsdp.py debug-train --trainer-device 7
  python scripts/npu/run_qwen3_5_4b_fsdp.py sglang --engine-device 1
  python scripts/npu/run_qwen3_5_4b_fsdp.py train --trainer-device 0
"""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import typer

from miles.utils.external_utils import command_utils

SEQ_LEN = 2048
# NpuPlatform.weight_transfer_modes; the launcher does not import torch_npu to read it
WEIGHT_TRANSFER_MODE = "disk-delta"


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    run_id: str = command_utils.create_run_id()
    model_path: str = "/mnt/weight/Qwen3.5-4B"
    data_path: str = "/mnt/share/dataset/dapo-math-17k/dapo-math-17k.jsonl"
    # physical cards; the launcher writes them into ASCEND_RT_VISIBLE_DEVICES for each job
    trainer_device: str = "0"
    engine_device: str = "1"
    sglang_port: int = 31001
    seq_length: int = SEQ_LEN
    num_rollout: int = 3
    rollout_batch_size: int = 4
    n_samples_per_prompt: int = 4
    rollout_max_response_len: int = 1024
    save_dir: str = "/home/weichunyu/miles-950/checkpoints"
    # one 4B checkpoint costs ~51 GB (20 GB model + 32 GB Adam fp32 state), and the root disk is tight
    save_interval: int = 3
    debug_data_dir: str = "/home/weichunyu/miles-950/debug-rollout"
    extra_args: str = ""
    # the engine runs in its own shell; a run must not SIGKILL it, which the backend's default cleanup does
    graceful_ray_stop: bool = True


def _require(path: str, what: str) -> None:
    if not Path(path).exists():
        raise typer.BadParameter(f"{what} not found: {path}")


def _trainer_env(args: ScriptArgs) -> dict[str, str]:
    return {
        "MILES_ACCELERATOR": "npu",
        # the trainer must see exactly one card; Ray schedules the worker onto a synthetic GPU slot, the mask is ours
        "ASCEND_RT_VISIBLE_DEVICES": args.trainer_device,
        # 950 multimodal attention backend selection, same as the working probe D launch
        "ASCEND_USE_FIA": "1",
        # skips the launcher's nvidia-smi probe, exactly as the TPU launcher does
        "NCCL_NVLS_ENABLE": "0",
    }


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


def _recipe_args(args: ScriptArgs) -> str:
    return (
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        f"--rollout-temperature 1 "
        f"--global-batch-size {args.rollout_batch_size * args.n_samples_per_prompt} "
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


def _train_backend_args(args: ScriptArgs) -> str:
    return (
        "--train-backend fsdp "
        "--micro-batch-size 1 "
        "--gradient-checkpointing "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {args.seq_length} "
        "--log-probs-backend torch "
        "--distributed-backend hccl "
    )


def _misc_args(args: ScriptArgs) -> str:
    return (
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 1 "
        "--num-gpus-per-node 1 "
        f"--save {args.save_dir} "
        f"--save-interval {args.save_interval} "
    )


def serve_sglang(args: ScriptArgs) -> None:
    """Serve the policy on the engine card in the foreground."""
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = args.engine_device
    os.environ["ASCEND_USE_FIA"] = "1"
    argv = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
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
        str(args.seq_length + args.rollout_max_response_len),
        "--max-running-requests",
        str(args.rollout_batch_size),
        "--cuda-graph-backend-decode",
        "disabled",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.sglang_port),
    ]
    os.execv(sys.executable, argv)


def make_debug_data(args: ScriptArgs) -> None:
    """Write a rollout-data file the trainer can consume without an SGLang server."""
    from transformers import AutoTokenizer

    from miles.utils.types import Sample

    _require(args.model_path, "model")
    _require(args.data_path, "dataset")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    prompts = []
    with open(args.data_path) as handle:
        for line in handle:
            if len(prompts) == args.rollout_batch_size:
                break
            row = json.loads(line)
            if row.get("prompt"):
                prompts.append(row["prompt"])
    if not prompts:
        raise typer.BadParameter(f"no prompts in {args.data_path}")

    # a short deterministic response: enough tokens to make a loss, no reward signal to interpret yet
    response_text = "答案是 2。"
    response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]

    samples = []
    for prompt_index, raw_prompt in enumerate(prompts):
        # flatten chat messages to plain text: the debug file only needs integer token ids
        text = (
            "\n".join(str(message.get("content", "")) for message in raw_prompt)
            if isinstance(raw_prompt, list)
            else raw_prompt
        )
        prompt_ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
        prompt_ids = prompt_ids[: max(1, args.seq_length - len(response_ids) - 2)]
        for _ in range(args.n_samples_per_prompt):
            tokens = [int(token) for token in (*prompt_ids, *response_ids)]
            samples.append(
                Sample(
                    index=prompt_index,
                    group_index=prompt_index,
                    prompt=text,
                    tokens=tokens,
                    response=response_text,
                    response_length=len(response_ids),
                    loss_mask=[1] * len(response_ids),
                    rollout_log_probs=[0.0] * len(response_ids),
                    # GRPO normalizes within one rollout, which rejects mixed rewards: a uniform reward keeps the
                    # first step about the training loop, not about the reward signal
                    reward=1.0,
                    status=Sample.Status.COMPLETED,
                )
            )

    out_dir = Path(args.debug_data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "0.pt"
    # imported here so `--help` works without the CANN environment that importing torch needs
    import torch

    torch.save(dict(rollout_id=0, metadata={}, samples=[sample.to_dict() for sample in samples]), path)
    typer.echo(f"wrote {path} with {len(samples)} samples")


def _execute(args: ScriptArgs, train_args: str) -> None:
    backend = args.create_backend()
    backend.execute_train(
        train_args=train_args,
        num_gpus_per_node=1,
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
    serve_sglang(args)


@app.command()
@command_utils.dataclass_cli
def debug_data(args: ScriptArgs):
    """Write a fixed rollout-data file for `debug-train`."""
    make_debug_data(args)


@app.command()
@command_utils.dataclass_cli
def debug_train(args: ScriptArgs):
    """Train one rollout on the fixed debug batch, with no SGLang server."""
    args.rollout_batch_size = 2
    args.n_samples_per_prompt = 2
    # one rollout with the default interval would never save; the debug run exists to show the checkpoint lands
    args.save_interval = 1
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    train_args = " ".join(
        [
            f"--hf-checkpoint {args.model_path}",
            f"--load-debug-rollout-data {Path(args.debug_data_dir) / '{rollout_id}.pt'}",
            "--num-rollout 1",
            _dataset_args(args),
            _recipe_args(args),
            _train_backend_args(args),
            _misc_args(args),
            args.extra_args,
        ]
    )
    _execute(args, train_args)


@app.command()
@command_utils.dataclass_cli
def train(args: ScriptArgs):
    """Train on --trainer-device against the SGLang server on --engine-device."""
    _require(args.model_path, "model")
    _require(args.data_path, "dataset")
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    train_args = " ".join(
        [
            f"--hf-checkpoint {args.model_path}",
            _dataset_args(args),
            "--rm-type math",
            f"--num-rollout {args.num_rollout}",
            _recipe_args(args),
            f"--rollout-external-engine-addrs http://127.0.0.1:{args.sglang_port}",
            "--rollout-num-gpus 1",
            "--rollout-num-gpus-per-engine 1",
            f"--update-weight-transfer-mode {WEIGHT_TRANSFER_MODE}",
            _train_backend_args(args),
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
