"""探针 C：Qwen3.5-4B 单卡 bf16 前向 + 一次反向（HF transformers，NPU 0）。"""

import os
import time

import torch
import torch_npu
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("PROBE_C_MODEL", "/mnt/weight/Qwen3.5-4B")
DTYPE = torch.bfloat16


def main():
    torch.npu.set_device(0)
    print("device", torch.npu.get_device_name(0), flush=True)

    cfg = AutoConfig.from_pretrained(MODEL)
    print("config class", type(cfg).__name__, "| model_type", getattr(cfg, "model_type", "?"), flush=True)
    for attr in ("layer_types", "linear_attention", "full_attention_interval"):
        if hasattr(cfg, attr):
            value = getattr(cfg, attr)
            print(f"config.{attr} = {str(value)[:160]}", flush=True)

    t0 = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=DTYPE, attn_implementation="eager", low_cpu_mem_usage=True
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE, attn_implementation="eager")
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)
    print("class", type(model).__name__, flush=True)
    print("params", f"{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B", flush=True)

    model = model.to("npu:0")
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    text = "Solve: what is 2 + 3? Answer with the number only."
    enc = tok(text, return_tensors="pt")
    ids = enc["input_ids"].to("npu:0")
    if ids.shape[1] > 32:
        ids = ids[:, :32]
    print("input shape", tuple(ids.shape), flush=True)

    torch.npu.reset_peak_memory_stats()
    out = model(input_ids=ids, labels=ids)
    loss = out.loss
    print("loss", float(loss.detach().cpu()), flush=True)
    print("finite loss", bool(torch.isfinite(loss).cpu()), flush=True)

    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    print("params with grad", f"{len(grads)}/{sum(1 for _ in model.parameters())}", flush=True)
    if grads:
        total = torch.sqrt(sum(g.float().pow(2).sum() for g in grads))
        print("grad norm", float(total.cpu()), flush=True)
    peak = torch.npu.max_memory_allocated() / 1024**3
    print(f"HBM peak {peak:.1f} GB", flush=True)
    print("PROBE_C_OK", flush=True)


if __name__ == "__main__":
    main()
