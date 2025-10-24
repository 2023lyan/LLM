import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def train_single_process(model, data, target, loss_fn):
    model = ToyModel(data.shape[1], target.max().item() + 1)
    torch.manual_seed(42)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def naive_ddp(rank, world_size, in_features, out_features, data, target, lr, loss_fn, results):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    torch.manual_seed(42)
    model = ToyModel(in_features, out_features).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    per_rank = len(data) // world_size
    x = data[rank * per_rank : (rank + 1) * per_rank].to(device)
    y = target[rank * per_rank : (rank + 1) * per_rank].to(device)

    optimizer.zero_grad(set_to_none=True)
    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()

    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world_size)

    optimizer.step()

    gathered_state = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_state, {k: v.cpu() for k, v in model.state_dict().items()})
    if rank == 0:
        state = {k: v.cpu() for k, v in model.state_dict().items()}
        for i in range(1, world_size):
            for k in state.keys():
                if not torch.allclose(state[k], gathered_state[i][k], rtol=1e-6, atol=1e-6):
                    print(f"[Mismatch] rank0 vs rank{i}: {k}")
        print("All ranks synced successfully.")
        results.append(state)

    cleanup()


def main():
    world_size = torch.cuda.device_count()

    in_features = 20
    out_features = 5
    batch_size = 64
    lr = 1e-3
    seed = 42

    torch.manual_seed(seed)

    data = torch.randn(batch_size, in_features)
    target = torch.randint(0, out_features, (batch_size,))
    loss_fn = nn.CrossEntropyLoss()

    ref_model = ToyModel(in_features, out_features)
    torch.manual_seed(seed)
    ref_state = train_single_process(ref_model, data, target, loss_fn)

    manager = mp.Manager()
    results = manager.list()

    mp.spawn(
        naive_ddp,
        nprocs=world_size,
        args=(world_size, in_features, out_features, data, target, lr, loss_fn, results),
        join=True
    )

    ddp_state = results[0]
    for k in ref_state.keys():
        assert torch.allclose(ref_state[k], ddp_state[k], rtol=1e-6, atol=1e-6)
    print("DDP training matches single-process training!")


if __name__ == "__main__":
    main()
