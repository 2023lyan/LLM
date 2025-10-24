import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import timeit
import pandas as pd

def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if backend == "nccl":
        torch.cuda.set_device(rank)

def run_all_reduce(rank, backend, tensor_size):
    device = torch.device(f"cuda:{rank}" if backend == "nccl" else "cpu")
    x = torch.ones(tensor_size // 4, dtype=torch.float32).to(device)

    # Warm-up
    for _ in range(5):
        dist.all_reduce(x, async_op=False)
        if backend == "nccl":
            torch.cuda.synchronize()

    dist.barrier()
    # Timing
    times = []
    for _ in range(10):
        dist.barrier()
        start = timeit.default_timer()
        dist.all_reduce(x, async_op=False)
        if backend == "nccl":
            torch.cuda.synchronize()
        end = timeit.default_timer()
        times.append(end - start)

    return sum(times) / len(times)

def worker(rank, world_size, backend, tensor_size, results):
    setup(rank, world_size, backend)
    avg_time = run_all_reduce(rank, backend, tensor_size)
    gathered_times = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_times, avg_time)
    if rank == 0:
        avg_time_overall = sum(gathered_times) / world_size
        results.append({
            "backend": backend,
            "world_size": world_size,
            "tensor_size_MB": tensor_size / (1024 * 1024),
            "avg_time_sec": avg_time_overall
        })
    dist.destroy_process_group()


if __name__ == "__main__":
    backends = ["gloo", "nccl"] if torch.cuda.is_available() else ["gloo"]
    world_sizes = [2, 4, 6]
    sizes_MB = [1, 10, 100, 1024]  # 1MB, 10MB, 100MB, 1GB = 1024MB
    results = mp.Manager().list()

    for backend in backends:
        for ws in world_sizes:
            if backend == "nccl" and torch.cuda.device_count() < ws:
                print(f"Skipping NCCL world_size={ws}, GPUs insufficient.")
                continue
            for size_MB in sizes_MB:
                size_bytes = int(size_MB * 1024 * 1024)
                mp.spawn(worker, nprocs=ws, args=(ws, backend, size_bytes, results), join=True)

    df = pd.DataFrame(list(results))
    print(df.to_markdown())