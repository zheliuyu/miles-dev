"""探针 B：两卡 HCCL all-reduce + rooted reduce。用 fork 启动，避免重导入 __main__。"""

import datetime
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, world_size):
    torch.npu.set_device(rank)
    dist.init_process_group(
        backend="hccl",
        init_method="tcp://127.0.0.1:29511",
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(seconds=300),
    )
    t0 = time.time()
    x = torch.tensor([1.0 + rank, 2.0 + rank, 3.0 + rank], dtype=torch.float32, device="npu")
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    got_all = x.cpu().tolist()

    y = torch.tensor([10.0 * (rank + 1)], dtype=torch.float32, device="npu")
    dist.reduce(y, dst=0, op=dist.ReduceOp.SUM)
    got_root = y.cpu().tolist()

    big = torch.ones(16 * 1024 * 1024, dtype=torch.float32, device="npu") * (rank + 1)
    dist.all_reduce(big, op=dist.ReduceOp.SUM)
    checksum = float(big.sum().cpu())
    dt = time.time() - t0

    print(
        f"rank{rank}: all_reduce={got_all} rooted_reduce={got_root} "
        f"16M-sum={checksum:.0f} wall={dt:.2f}s device={torch.npu.current_device()}",
        flush=True,
    )
    dist.barrier()
    if rank == 0:
        print("PROBE_B_OK", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.start_processes(worker, args=(2,), nprocs=2, join=True, start_method="fork")
