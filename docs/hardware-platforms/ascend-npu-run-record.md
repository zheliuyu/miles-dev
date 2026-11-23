# miles-dev 在 Ascend 950DT 上的支持度与接入记录

> 本地执行记录（探针与阶段 4 写于 2026-11-20，阶段 5 闭环写于 2026-11-23）。**提交 PR 前删除本文件。** 执行脚本见 `scripts/npu/probes/`，构建与探针日志见 `scripts/npu/evidence/`。

分析对象：`/home/weichunyu/miles-dev`。记录开始时的基线是 `fa13e2ec`。平台与启动器提交到 `eaa238ab`，这一点相对 `radixark/miles` 的 `main` 超前 7、落后 32。fork 是 `zheliuyu/miles-dev`。

本文是单份执行记录：硬件事实、为什么不用仓库自带补丁、选定的接入方式、环境与验证、已落地的改动、剩余阶段。

## 0. 现状一页

| 阶段 | 状态 | 证据 |
| --- | --- | --- |
| 环境 | ✅ | `miles-950-py312` 重建并验证，31 项版本与目标一致 |
| 探针 A（单卡算子） | ✅ | bf16 4096² 前向+反向，loss 4095.75，HBM 峰值 432 MB |
| 探针 B（两卡 HCCL） | ✅ | all-reduce `[3,5,7]`、rooted reduce `30/20`、16M 元素 2.59 s |
| 探针 C（Qwen3.5-4B 前向） | ✅ | loss 2.9333，426/426 参数有梯度，HBM 20.8 GB |
| 探针 D（SGLang 生成） | ✅ | NPU 1 起服务，返回「1+1 等于 2。」，14 tokens |
| 阶段 3 平台对象 | ✅ 已提交 | 5 个 CPU 单测通过；`MILES_ACCELERATOR=npu` → npu/eager/disk-delta |
| 阶段 4 纯训练 | ✅ | debug 数据 4 步，20.5 s，checkpoint 落盘（51 GB） |
| 阶段 5 闭环 | ✅ | 3 个 rollout 退出码 0，weight version 1→2→3；从 `iter_0000003` 续跑 rollout 3，退出码 0，写出 `iter_0000004`。见 §9 |
| 阶段 6 扩 4 卡 / 阶段 7 提 PR | ⬜ | 阶段 5 已不再挡住；4 卡启动器和 HCCL `broadcast` 都还没做 |

## 1. 机器事实（2026-11-20 实测）

| 项目 | 实测 |
| --- | --- |
| 主机 | `177-5-25-1.user3p.v-tal.net.br`，openEuler 24.03 SP3，aarch64 |
| CPU / 内存 | 鲲鹏 950 7592C，2 路 × 96 核 = 192 核；内存 2.2 TiB（可用约 2.0 TiB） |
| NPU | 8 × Ascend950DT，板卡名 9582，芯片 V120，每卡 HBM 96 GB（合计约 768 GB） |
| 驱动 / 固件 | HDK `25.1.rc3`，固件 `9.2.0.101.235`，`npu-smi 25.1.rc3` |
| CANN | 系统自带 `/usr/local/Ascend/cann-9.2.0`（另有 `cann-9.2.0-dsv-20261030`）；训练环境统一 source `/home/pkg/cann920/cann-9.2.0` |
| base Python | 3.12.14 |
| 磁盘 | 一个 checkpoint 约 51 GB。工作目录 `/home/weichunyu/miles-950` 是指向 `/mnt/share/weichunyu/miles-950` 的符号链接（NFS，百 TB 级空闲）。checkpoint 与 `weight-delta` 必须留在这条路径上，不能写回根分区 |
| 选卡 | 卡是动态的：跑之前先 `npu-smi info`，再用 `ASCEND_RT_VISIBLE_DEVICES` 钉住，**不杀别人的进程** |

## 2. 为什么不用仓库自带的 NPU 补丁

上游声明的硬件是 NVIDIA GB300/GB200/B300/B200/H200/H100/A100 与 AMD MI355X/MI350X/MI325X/MI300X，
`setup.py` 分类器是 `NVIDIA CUDA`。昇腾只出现在 `docker/npu_patch/`，且这份补丁对不上本机：

| 组件 | 补丁要求 | 本机 |
| --- | --- | --- |
| Miles | `551d159` | `fa13e2ec`（更后面的主线） |
| Python | 3.11 | 3.12 |
| CANN | 8.5.0 + NNAL/ATB | 9.2.0 |
| HDK | 25.3.RC1（910 代） | 25.1.rc3（950DT） |
| torch / 训练栈 | torch-npu 2.8.0 + Megatron + MindSpeed | 不适用 |

`git apply --check docker/npu_patch/miles.patch` 不能应用（`command_utils.py` 已被拆包、多处补丁块对不上），
补丁里的 `execute_train_npu` 还把 `ASCEND_HOME_PATH` 写死成开发者目录，并注释掉了 P2P 权重更新。
结论：**硬件够，软件路线要重做；不改系统 CANN，也不碰这份补丁。**

结论：**硬件支持度高，软件支持度要靠这次接入补上**。换栈之后本机的能力边界如下：

| 能力 | 判断 |
| --- | --- |
| 直接 `pip install` 后按 CUDA 配方训练 | 不可用（无 NPU 设备后端）；本次已补上 `MILES_ACCELERATOR=npu` |
| 套用 `docker/npu_patch` | 不可用：补丁锁在 CANN 8.5 / Python 3.11 / torch-npu 2.8 / 910 代 |
| 中小模型 RL（4B、2 卡） | 可用：探针 A–D、阶段 4，以及阶段 5 的 GRPO 闭环与 checkpoint 续跑已验证。reward 接近 0，回答大多顶到 1024 截断 |
| 8 卡满配、千亿级、README 里的前沿模型 | 没有对应实现，也不在本轮范围 |
| 修改系统 CANN / 系统 Python | 不做；训练环境自带 CANN 9.2.0 与 conda 环境 |

`scripts/run_qwen3_4b_npu.py` 依赖未合入的 `execute_train_npu`，当前 HEAD 上导入即失败，同样不进入这条路径。

## 3. 选定方案：平台对象 + FSDP + 外部 SGLang

照仓库对 TPU 的接法：

- 新增 `miles.platforms` 的 `npu` 后端（`MILES_ACCELERATOR=npu`），设备事实声明在平台类里，共享代码不写死 `"cuda"`；
- 训练用 **FSDP** 读 HuggingFace 权重（单卡不需要张量并行）；
- rollout 用**外部 SGLang 进程**，训练与引擎分卡，不用 `--colocate`（其实现依赖 CUDA IPC）；
- 权重回传先用现成的 `--update-weight-transfer-mode disk-delta`；`broadcast` 等 HCCL 稳定后再试，作为去掉磁盘中转的优化；
- 分布式后端 `hccl`，CPU offload 关闭（避免拼出未测过的 `cpu:gloo,npu:hccl`）。

刻意不做：Megatron、MindSpeed 接入、MXFP8/NVFP4、P2P RDMA、8 卡、Qwen3.5-27B/35B-A3B、修改系统 CANN、
`docker/npu_patch/`。`flash-linear-attention` 留到正确性验证之后，仅用于给 GDN 提速。

## 4. 目标与验收

目标：**在 Ascend950DT 上跑通一次 Miles 的 GRPO 闭环，并按上游多硬件规则提交回 `radixark/miles`。**

闭环的六件事：SGLang 采样 dapo-math-17k → 数学 reward 打分 → FSDP 对 Qwen3.5-4B（bf16）做优化器更新 →
新权重送回 SGLang → 下一轮 rollout 的 weight version 递增 → 写出 checkpoint，新进程能从 checkpoint 继续跑一轮。

验收规模：**2 张卡**（1 张训练、1 张引擎）。4B 权重约 8 GB，一张 96 GB 卡足够；4 卡数据并行是闭环之后的事。

完成定义（四条同时成立）：① 3 个 rollout 跑完并能从 checkpoint 续跑；② 改动拆成 PR 1–4 提交上游；
③ `docs/hardware-platforms/ascend-npu.md` 能让另一台同栈机器从 `prepare` 走到 `train`；④ 默认 CUDA 路径 CPU 测试仍通过。

## 5. 软件栈

环境：`miles-950-py312`，`/root/miniconda3/envs/miles-950-py312`，Python 3.12.16。
`conda activate miles-950-py312` 会经 `etc/conda/activate.d/ascend.sh` 自动 source
`/home/pkg/cann920/ascend-toolkit/set_env.sh`（`ASCEND_HOME_PATH=/home/pkg/cann920/cann-9.2.0`）。
旧的 `miles-950`（Python 3.11）**没有重建**：950 内核轮子按 3.12 构建。

| 组件 | 版本 | 说明 |
| --- | --- | --- |
| torch / torchvision / torchaudio | 2.10.0+cpu / 0.25.0 / 2.10.0 | 与 `torch_npu 2.10.0.post7` 配套 |
| torch_npu | 2.10.0.post7 | 官方配套 CANN 9.2.0 |
| triton-ascend | 3.6.0（同时装 `triton 3.6.0`、新增 `triton_dist`） | `pip install triton-ascend==3.6.0 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple`；`triton.backends = ['amd','ascend','nvidia']` |
| numpy / psutil / protobuf | 1.26.4 / 6.0.0 / 6.33.6 | numpy 由 `triton-ascend` 的元数据钉住 |
| SGLang | 0.5.23 | `--no-deps` 装（元数据要 torch 2.14，会换坏 torch） |
| sgl-kernel-npu | 2026.11.0.post4 | release `2026.11.0.post10` 的 zip，`compute_units=ascend950`，`--no-deps` |
| transformers | 5.19.0 | 与 Miles `requirements.txt` 一致 |
| datasets / pyarrow | 4.0.0 / 20.0.0 | 见下方"版本冲突" |
| miles | 0.2.0（editable） | `pip install -e /home/weichunyu/miles-dev --no-deps` + requirements |
| ray | 2.58.2 | `ray[default]>=2.56,<2.59` |
| 其他 | blake3 1.0.9 / xxhash 3.7.1 / zstandard 0.25.0 / sglang-router 0.3.2 / accelerate 1.16.0 / wandb 0.31.0 / memray 1.20.0 / polars 1.42.1 / omegaconf 2.4.0 | 均与目标一致 |

不装：`nvidia-resiliency-ext`、`ring_flash_attn`（CUDA 专用）、`torchft-nightly`（仅 x86_64）、`flash-linear-attention`（暂缓）。

构建方式（三条，取代逐个下载）：

1. **torch 栈本机复用**：`qwen3_5` 环境里有同批轮子，直接复制，跳过 146 MB 的 torch 下载；
   漏掉的 `functorch`、`torchgen`、`torchdata`、`typing_extensions`、`pyyaml`、`pybind11`、`torchaudio` 后来补齐。
2. **安装器用 `uv`**：本机 Python 的 TLS 栈异常慢——同一个 wheel，`curl` 4.7 MB/s、`urllib` 33 KB/s、
   `pip` 27–98 KB/s，且豆瓣/阿里云/华为云/清华/USTC 五个源一致（瓶颈在客户端，不在镜像）。`uv` 走 rustls，
   整套 SGLang 依赖 47 秒装完。
3. **`sgl-kernel-npu` 经代理取**：本机 `curl https://github.com` 返回 000，改走
   `https://gh-proxy.com/https://github.com/...` 下载 release。

版本冲突（已解决）：`datasets 5.2.0` 要求 `pyarrow>=24`，pyarrow 24+ 要求 numpy 2，而 `triton-ascend` 钉
`numpy==1.26.4`；三者互斥会导致 Ray job 起不来（`pyarrow requires NumPy 2.0 or newer`）。
解法是降 `datasets` 与 `pyarrow`：`uv pip install 'pyarrow==20.0.0' 'datasets<5'`。

## 6. 验证：探针 A–D（全部通过）

| 探针 | 做法 | 结果 |
| --- | --- | --- |
| A 单卡算子 | `ASCEND_RT_VISIBLE_DEVICES=0`，4096² bf16 矩阵乘 + 反向 | loss 4095.75，输入梯度范数 2.83/2.83，输出全有限；HBM 峰值 432 MB → 释放回 0 |
| B 两卡 HCCL | `0,1`，`torch.distributed` + `hccl`，all-reduce / rooted reduce / 16M 元素 | 两 rank 都得 `[3.0,5.0,7.0]`；`reduce(dst=0)` = `30.0`（rank1 保持 `20.0`）；校验和 `50331648`，含初始化 2.59 s |
| C 单卡 HF 前向 | `0`，`transformers` 加载 Qwen3.5-4B，17 token 前向 + 反向 | `Qwen3_5Config` 4.21B，426 张量 0.6 s 加载；loss 2.9333、426/426 参数有梯度、范数 128.02、HBM 20.8 GB |
| D SGLang 生成 | `1`，`--device npu --attention-backend ascend --mamba-ssm-dtype bfloat16` | 首次 105 s 起服务、升级 triton 后 50 s；返回「1+1 等于 2。」，14 tokens，`stop`；HBM 约 47 GB |

结论：**950DT 的软硬件栈能满足 FSDP 训练 + SGLang rollout**；GDN、causal conv、Gemma RMSNorm 在 `torch_npu`
上按 `transformers` 的 PyTorch 参考实现即可训练（未装 `causal_conv1d` / `flash-linear-attention`）。
HCCL 的 all-reduce 与 rooted reduce 都可用，`broadcast` 权重回传具备前提。

探针 D 通过的启动参数（也是引擎部分的依据）：

```bash
conda activate miles-950-py312
export ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_USE_FIA=1
python -m sglang.launch_server \
  --model-path /mnt/weight/Qwen3.5-4B --device npu --attention-backend ascend \
  --language-only --dtype bfloat16 --mamba-ssm-dtype bfloat16 \
  --tp-size 1 --mem-fraction-static 0.5 --context-length 2048 --max-running-requests 1 \
  --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled \
  --host 127.0.0.1 --port 31001
```

## 7. 已落地的代码改动

分支 `feat/ascend-950-support`。到 `eaa238ab` 为止是平台、启动器和记录本身（相对 `radixark/miles:main` 超前 7、落后 32）：

| 提交 | 内容 |
| --- | --- |
| `085d3d8a` | `feat(npu)`：新增 `miles/platforms/npu.py` 与 `MILES_ACCELERATOR=npu` 选择，含 CPU 单测 |
| `483d20e8` | `fix(weight-update)`：disk-delta 里 `torch.cuda.current_stream / current_device` 改 `torch.accelerator` |
| `dd7140ef` | `fix(fsdp)`：actor/checkpoint 的 `current_device`、`empty_cache`、`synchronize`、RNG 改 `torch.accelerator` |
| `158f554c` | `fix(fsdp)`：`init_device_mesh` 的 `"cuda"` 改 `current_platform.device_type`（否则 `torch._C._cuda_setDevice` 崩） |
| `4aa155c9` | `docs(npu)`：新增 `scripts/npu/run_qwen3_5_4b_fsdp.py` 与 `scripts/npu/README.md` |
| `fa8409a3` | 本记录、探针与阶段 4 证据 |
| `eaa238ab` | 文档里的提交哈希改写成上面这一列 |

阶段 5 为了把闭环跑通，又改了下面这些共享路径。没有这些修改，§9 的两次任务都会在中途退出：

| 文件 | 原因 |
| --- | --- |
| `miles/utils/memory_utils.py`、`miles/backends/fsdp_utils/actor.py` | 本机 `torch.accelerator` 的显存接口报 `Allocator for npu is not a DeviceAllocator`，改走 `torch.npu` |
| `miles/backends/fsdp_utils/adaptations/precision.py`、`actor.py` | `from_pretrained` 按 `config.dtype=bfloat16` 把 48 个文件里的 fp32 张量（GDN 的 `A_log`、`norm.weight`）转成 bf16，disk-delta 基线校验失败 |
| `miles/backends/fsdp_utils/hf_weight_iterator.py` | `tie_word_embeddings` 使 `state_dict` 带上 `lm_head.weight`，safetensors 里没有这个名字 |
| `miles/utils/arguments.py` | disk-delta 把带 `latest_checkpointed_iteration.txt` 的 `--load` 一律拒绝；FSDP 也写这个文件。拒绝只保留给 Megatron |
| `miles/backends/fsdp_utils/checkpoint.py` | 纯文本步不会给视觉塔留下 Adam 状态，空优化器加载时却为每个可训练参数建状态。优化器加载改为 `allow_partial_load` |

另外，库存 SGLang 0.5.23 没有 `/pull_weights`。接收端是环境里的本地补丁（`sglang_npu_pull_weights.py` 以及 `http_server.py` 上的路由），不在本仓库。重装 SGLang 会丢掉它。

`NpuPlatform` 的取值：`device_type=npu`、`compile_backend=eager`、`triton_kernels=False`、`flex_attention=False`、
`dynamic_shapes=False`、`nvml_numa_affinity=False`、`rooted_reduce=False`、`premul_sum_reduce=False`、
`weight_transfer_modes={'disk-delta'}`、`prefill_weight_versions=False`、`rollout_kv_cache_dtype=bfloat16`；
设备绑定发生在建进程组**之前**（HCCL 在 `init_process_group` 时读当前设备，和 TPU 相反）。

启动器 `scripts/npu/run_qwen3_5_4b_fsdp.py`（结构对齐 `scripts/tpu/`）：子命令 `prepare` / `sglang` / `train` /
`debug-data` / `debug-train`；`--trainer-device`、`--engine-device` 写进各自的 `ASCEND_RT_VISIBLE_DEVICES`；
默认指向本机已有权重与数据集；`graceful_ray_stop=True`（否则后端的清理会 `pkill -9 sglang`，把外部引擎杀掉）。
Ray 资源不需要改 `miles/ray/*`：`execute_train` 会执行 `ray start --num-gpus N`，Ray 随即报 `GPU: N`，
放置组正常调度；代价是 Ray 不再替 worker 设 `ASCEND_RT_VISIBLE_DEVICES`，由启动器自己写。

## 8. 阶段 4 结果（纯训练，不接生成）

`debug-train --trainer-device 1`（`ASCEND_RT_VISIBLE_DEVICES=1`）通过 `--load-debug-rollout-data` 读一份固定的
`Sample` 列表（格式 `torch.save(dict(rollout_id=…, metadata={}, samples=[s.to_dict()…]))`，参数解析时自动置
`debug_train_only=True`、跳过 SGLang）。结果：4 个优化器 step，每步 4–6 s（`actor_train` 20.5 s），退出码 0。

- `train/loss` 与 `grad_norm` 为 0.0：GRPO 要求同一 rollout 内 reward 一致，debug 数据只能给 uniform reward，
  优势恒为 0。这一步证明的是训练循环（前向、反向、优化器、checkpoint），不是 reward 信号。
- checkpoint 写到 `checkpoints/iter_0000001`：model 20 GB + optimizer 32 GB（Adam fp32）+ `rng.pt` / `lr_scheduler`，
  **合计 51 GB**，`meta.json` 记录 `iteration=1, next_rollout_id=1, world_size=1`。

## 9. 阶段 5 结果与剩余工作

### 阶段 5：闭环（2026-11-23，已通过）

引擎在 5 卡，端口 31001。第一次训练在 6 卡，任务 `raysubmit_t48Pmq7NdnZ1bTNa`，15:46 退出码 0，写出 `iter_0000003`（51 GB，`next_rollout_id=3`）。配方就是上一节的 3 个 rollout、batch 4、每题 4 条、回答上限 1024、`disk-delta`、评估关闭。`--update-weight-disk-dir` 与 `--update-weight-local-checkpoint-dir` 由 `--extra-args` 传入，分别指向 `/home/weichunyu/miles-950/weight-delta` 和 `rollout-checkpoint`。

| 轮次 | 采样 | 训练 | 送回引擎 |
| --- | --- | --- | --- |
| 训练前 | — | — | version 1，变更 0% |
| rollout 0 | 16/16，回答长度 1005，截断 0.875，reward 0 | step 0，`grad_norm` 1.26，`loss` 0 | version 2，1.58%，0.34 GB |
| rollout 1 | 16/16，回答长度 1016，截断 0.938，reward 约 0 | step 1，`grad_norm` 1.19 | version 3，1.25%，0.28 GB |
| rollout 2 | 16/16，回答长度 928，截断 0.75，reward 0 | step 2，`grad_norm` 0，`loss` 0 | 没有再发一版；权重留在 checkpoint |

第 2、3 次送回的 version（2 和 3）都大于第 1 次（1）。rollout 2 更新后的权重在 `iter_0000003` 里，进程退出前没有打成 version 4。

续跑在 4 卡，任务 `raysubmit_V1rAisPxAQdVumd3`，`--load /home/weichunyu/miles-950/checkpoints --num-rollout 4`，17:26 退出码 0。模型、优化器、学习率调度器从 `iter_0000003` 载入，rollout 进度从 `checkpoints/rollout/2` 接上。新进程的 disk-delta 版本从 1 重新计，这一版相对原始 HF 权重变更 2.22%、0.45 GB，引擎回报 200。rollout 3：16/16，回答长度 1024，截断 1.0，reward 0，`weight_version` 1；step 3 的 `loss` 与 `grad_norm` 都是 0。随后写出 `iter_0000004`（`rollout_id=3`，`next_rollout_id=4`）。

四条通过标准都有结果：3 个 rollout 退出码 0；后续送回的 weight version 高于第一版；reward 日志有数值；同一 checkpoint 再完成 1 个 rollout。reward 没有上升，回答大多顶到 1024。这证明的是采样、打分、更新、回传、续跑这条链，不是策略变好了。

### 阶段 6：扩到 4 张空闲卡

原计划：训练用两张卡（FSDP DP=2），rollout 用另外两张卡（两个 SGLang，各 1 卡），仍用 `disk-delta`；通过标准同阶段 5。卡号按开跑时的 `npu-smi` 现选，不预留 0、1、2、3。

这一步还不能直接开。当前启动器把训练和引擎都钉成一张卡（`--actor-num-gpus-per-node 1`，一个 `--sglang-port`）。`broadcast` 也还不是把传输模式改成 HCCL 就行：`miles/backends/training_utils/weight_update/protocols/broadcast.py` 把进程组建在 `backend="nccl"` 上，`NpuPlatform.weight_transfer_modes` 仍只有 `disk-delta`。4 卡闭环先继续用 `disk-delta`；`broadcast` 单独做，测完再决定要不要写进平台类。

阶段 5 还说明两件和规模无关的事，4 卡跑不掉：上面那些共享路径修复要留在这棵树上；两个引擎都依赖环境内的 `/pull_weights` 补丁。reward 仍不作为通过条件。

### 阶段 7：PR 拆分（目标 `radixark/miles` 的 `main`）

| PR | 内容 |
| --- | --- |
| 1 | `miles/platforms/npu.py`、`_select_platform` 的 `npu` 分支、CPU 单测 |
| 2 | `scripts/npu/run_qwen3_5_4b_fsdp.py`、`scripts/npu/README.md`（无 950 的 CI 上只做导入与参数快照） |
| 3 | 阶段 4–6 里为避开 `torch.cuda` / `nccl` 而改的共享调用点（每个都落到 `current_platform` 或 `torch.accelerator`） |
| 4 | `docs/hardware-platforms/ascend-npu.md` + `overview.md` 链接，含版本、2 卡复现命令、4 卡扩展 |
| 5 | 仅当阶段 6 的 `broadcast` 已测过：HCCL 权重更新 |

PR 说明附本机日志（探针 A/B/C/D、rollout 的 loss/reward、`npu-smi` 设备号）。上游 CI 没有 950 runner，
第一批 PR 不注册 e2e CI；`pre-commit` 本地过 ruff、black（行宽 119）、isort，提交信息 `feat(npu): …`。
`README.md` 的硬件名单随文档 PR 一起改，写明范围是单机 Ascend950DT + Qwen3.5-4B + FSDP + SGLang，
不要写成"支持昇腾全系列"。`scripts/run_qwen3_4b_npu.py` 与 `docker/npu_patch/` 不动，PR 里注明 950 的入口是
`scripts/npu/`。

## 10. 风险与停机条件

| 情况 | 处理 |
| --- | --- |
| `torch_npu` 与 CANN 9.2.0 不匹配 | 已用 `2.10.0.post7` 通过单卡矩阵乘、探针 C/D。报 CANN 版本错误就停在当前环境换轮子，不改系统 CANN |
| 9.1 的 `sgl-kernel-npu` 在 9.2.0 上加载失败 | 导入与 Gemma RMSNorm 已通过；某算子失败则用 `/home/pkg/cann920/cann-9.2.0` 按 `Ascend950PR_9599` 重编，编不过就停 |
| 训练出现未实现算子 | 记算子名/shape/CANN 报错；能用 PyTorch 组合替代就替代并加数值对照，不能就缩短序列，不改模型结构 |
| HCCL 超时或挂起 | 端到端继续用 `disk-delta` + 单卡训练，集合通信不作为闭环前提 |
| checkpoint 写进根分区 | 停。只写 `/home/weichunyu/miles-950`（NFS 符号链接）；一个 `iter_*` 约 51 GB |
| 目标卡被别人占用 | 换仍空闲的卡并改 `ASCEND_RT_VISIBLE_DEVICES`，不杀已有进程 |
| 共享代码里出现新的 CUDA 调用 | 按多硬件规则改到 `current_platform` / `torch.accelerator`；一处调用要扩成大段 `if npu` 时，把差异收回 `NpuPlatform` |

## 11. 复现索引

- 环境：`conda activate miles-950-py312`（CANN 9.2.0 自动注入）
- 代码：`/home/weichunyu/miles-dev`，分支 `feat/ascend-950-support`
- 启动器：`scripts/npu/run_qwen3_5_4b_fsdp.py`，说明：`scripts/npu/README.md`
- 权重：`/mnt/weight/Qwen3.5-4B`；数据：`/mnt/share/dataset/dapo-math-17k/dapo-math-17k.jsonl`
- 工作目录：`/home/weichunyu/miles-950/`（`checkpoints/`、`weight-delta/`、`debug-rollout/`、日志）
- 平台入口：`miles/platforms/npu.py`；安装记录见本文 §5
