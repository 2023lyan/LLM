from cs336_basics.model import BasicsTransformerLM as Transformer
import argparse
import timeit
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import torch.optim as optim
from einops import rearrange
import torch.nn as nn
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ddp_overlap_individual_parameters import DDP

MODEL_CONFIGS = {
    "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl":     dict(d_model=1600, d_ff=6400,  num_layers=48, num_heads=25),
    "2.7B":   dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}

def parse_args():
    parser = argparse.ArgumentParser(description="Naive DDP Benchmarking script for CS336 systems assignment.")
    parser.add_argument("--model_size", type=str, choices=MODEL_CONFIGS.keys(), default="medium", help="Model size to benchmark.")
    parser.add_argument("--vocab_size", type=int, default = 10000, help="Vocabulary size.")
    parser.add_argument("--batch_size", type=int, default = 2, help="Batch size.")
    parser.add_argument("--context_length", type=int, default = 128, help="Context length.")
    parser.add_argument("--rope_theta", type=float, default = 10000.0, help="RoPE theta parameter.")
    return parser.parse_args()

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, device_id=torch.device(f"cuda:{rank}"))

def cleanup():
    dist.destroy_process_group()

def naive_ddp(rank, world_size, args, data, target, lr, loss_fn, results):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    torch.manual_seed(42)

    per_rank = len(data) // world_size
    x = data[rank * per_rank : (rank + 1) * per_rank].to(device)
    y = target[rank * per_rank : (rank + 1) * per_rank].to(device)
    model = Transformer(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        **MODEL_CONFIGS[args.model_size],
        rope_theta=args.rope_theta,
    ).to(device)
    model = DDP(model)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for _ in range(5): # Warm-up
        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        loss = loss_fn(rearrange(output, "batch_size sequence_length ... -> (batch_size sequence_length) ..."),
               rearrange(y, "batch_size sequence_length -> (batch_size sequence_length)"))
        loss.backward()
        model.finish_gradient_synchronization()
        optimizer.step()

    torch.cuda.synchronize()
    dist.barrier()
    start_total = timeit.default_timer()

    optimizer.zero_grad(set_to_none=True)
    output = model(x)
    loss = loss_fn(rearrange(output, "batch_size sequence_length ... -> (batch_size sequence_length) ..."), 
                   rearrange(y, "batch_size sequence_length -> (batch_size sequence_length)")) 
    loss.backward()

    model.finish_gradient_synchronization()

    optimizer.step()
    torch.cuda.synchronize()
    end_total = timeit.default_timer()
    total_time = end_total - start_total
    gathered_times = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_times, total_time)
    if rank == 0:
        avg_total_time = sum(t for t in gathered_times) / world_size
        results.append(avg_total_time)

    cleanup()

if __name__ == "__main__":
    args = parse_args()
    world_size = 2
    
    batch_size = args.batch_size * world_size
    context_length = args.context_length
    vocab_size = args.vocab_size
    
    data = torch.randint(0, vocab_size, (batch_size, context_length))
    target = torch.randint(0, vocab_size, (batch_size, context_length))
    loss_fn = nn.CrossEntropyLoss()
    lr = 1e-3

    results = mp.Manager().list()
    mp.spawn(
        naive_ddp,
        nprocs=world_size,
        args=(world_size, args, data, target, lr, loss_fn, results),
        join=True
    )
    total_time = results[0]
    print(f"total_time: {total_time:.4f}s")