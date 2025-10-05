# Lecture 8: Parallelism 2

## Collective operations: conceptual primitives used for distributed programming
World Size: total number of devices
Rank: unique ID for each device, from 0 to World Size - 1
- Broadcast: Rank 0: t0 -> Rank 1: t0, Rank 2: t0, Rank 3: t0, Rank 4: t0
- Scatter: Rank 0: (t0, t1, t2, t3) -> Rank 1: t0, Rank 2: t1, Rank 3: t2, Rank 4: t3
- Gather: Rank 1: t0, Rank 2: t1, Rank 3: t2, Rank 4: t3 -> Rank 0: (t0, t1, t2, t3)
- Reduce: Rank 1: t0, Rank 2: t1, Rank 3: t2, Rank 4: t3 -> Rank 0: t0+t1+t2+t3
- All Gather: Rank 1: t0, Rank 2: t1, Rank 3: t2, Rank 4: t3 -> Rank 1: (t0, t1, t2, t3), Rank 2: (t0, t1, t2, t3), Rank 3: (t0, t1, t2, t3), Rank 4: (t0, t1, t2, t3)
- All Reduce: Rank 1: t0, Rank 2: t1, Rank 3: t2, Rank 4: t3 -> Rank 1: t0+t1+t2+t3, Rank 2: t0+t1+t2+t3, Rank 3: t0+t1+t2+t3, Rank 4: t0+t1+t2+t3
- Reduce Scatter: Rank 1: (a0, a1, a2, a3), Rank 2: (b0, b1, b2, b3), Rank 3: (c0, c1, c2, c3), Rank 4: (d0, d1, d2, d3) -> Rank 1: (a0+b0+c0+d0, _, _, _), Rank 2: ( _, a1+b1+c1+d1, _, _), Rank 3: ( _, _, a2+b2+c2+d2, _), Rank 4: ( _, _, _, a3+b3+c3+d3)
All Reduce = Reduce Scatter + All Gather

## Torch distributed
Classic:
Within a node: PCIe
Across nodes: Ethernet

Modern:
Within a node: NVLink connects GPUs directly, bypass CPU
Across nodes: NVSwitch connects nodes directly, bypass Ethernet

`dist.barrier()` is used to synchronize all processes in
`dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)`: All Reduce operation.
`dist.reduce_scatter_tensor(tensor, output_tensor, op=dist.ReduceOp.SUM, async_op=False)`: Reduce Scatter operation.
`dist.all_gather_into_tensor(tensor_list, tensor, async_op=False)`: All Gather operation.

## Sample code for data parallelism
```python
def data_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_steps: int):
    setup(rank, world_size)

    # Get the slice of data for this rank (in practice, each rank should load only its own data)
    batch_size = data.size(0)  # @inspect batch_size
    num_dim = data.size(1)  # @inspect num_dim
    local_batch_size = int_divide(batch_size, world_size)  # @inspect local_batch_size
    start_index = rank * local_batch_size  # @inspect start_index
    end_index = start_index + local_batch_size  # @inspect end_index
    data = data[start_index:end_index].to(get_device(rank))

    # Create MLP parameters params[0], ..., params[num_layers - 1] (each rank has all parameters)
    params = [get_init_params(num_dim, num_dim, rank) for i in range(num_layers)]
    optimizer = torch.optim.AdamW(params, lr=1e-3)  # Each rank has own optimizer state

    for step in range(num_steps):
        # Forward pass
        x = data
        for param in params:
            x = x @ param
            x = F.gelu(x)
        loss = x.square().mean()  # Loss function is average squared magnitude

        # Backward pass
        loss.backward()

        # Sync gradients across workers (only difference between standard training and DDP)
        for param in params:
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)

        # Update parameters
        optimizer.step()

        print(f"[data_parallelism] Rank {rank}: step = {step}, loss = {loss.item()}, params = {[summarize_tensor(params[i]) for i in range(num_layers)]}", flush=True)

    cleanup()
```
## Sample code for tensor parallelism
```python
def tensor_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int):
    setup(rank, world_size)

    data = data.to(get_device(rank))
    batch_size = data.size(0)  # @inspect batch_size
    num_dim = data.size(1)  # @inspect num_dim
    local_num_dim = int_divide(num_dim, world_size)  # Shard `num_dim`  @inspect local_num_dim

    # Create model (each rank gets 1/world_size of the parameters)
    params = [get_init_params(num_dim, local_num_dim, rank) for i in range(num_layers)]

    # Forward pass
    x = data
    for i in range(num_layers):
        # Compute activations (batch_size x local_num_dim)
        x = x @ params[i]  # Note: this is only on a slice of the parameters
        x = F.gelu(x)

        # Allocate memory for activations (world_size x batch_size x local_num_dim)
        activations = [torch.empty(batch_size, local_num_dim, device=get_device(rank)) for _ in range(world_size)]

        # Send activations via all gather
        dist.all_gather(tensor_list=activations, tensor=x, async_op=False)

        # Concatenate them to get batch_size x num_dim
        x = torch.cat(activations, dim=1)

    print(f"[tensor_parallelism] Rank {rank}: forward pass produced activations {summarize_tensor(x)}", flush=True)

    # Backward pass: homework exercise

    cleanup()
```
## Sample code for pipeline parallelism
```python
def pipeline_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_micro_batches: int):
    setup(rank, world_size)

    # Use all the data
    data = data.to(get_device(rank))
    batch_size = data.size(0)  # @inspect batch_size
    num_dim = data.size(1)  # @inspect num_dim

    # Split up layers
    local_num_layers = int_divide(num_layers, world_size)  # @inspect local_num_layers

    # Each rank gets a subset of layers
    local_params = [get_init_params(num_dim, num_dim, rank) for i in range(local_num_layers)]

    # Forward pass

    # Break up into micro batches to minimize the bubble
    micro_batch_size = int_divide(batch_size, num_micro_batches)  # @inspect micro_batch_size
    if rank == 0:
        # The data
        micro_batches = data.chunk(chunks=num_micro_batches, dim=0)
    else:
        # Allocate memory for activations
        micro_batches = [torch.empty(micro_batch_size, num_dim, device=get_device(rank)) for _ in range(num_micro_batches)]

    for x in micro_batches:
        # Get activations from previous rank
        if rank - 1 >= 0:
            dist.recv(tensor=x, src=rank - 1)

        # Compute layers assigned to this rank
        for param in local_params:
            x = x @ param
            x = F.gelu(x)

        # Send to the next rank
        if rank + 1 < world_size:
            print(f"[pipeline_parallelism] Rank {rank}: sending {summarize_tensor(x)} to rank {rank + 1}", flush=True)
            dist.send(tensor=x, dst=rank + 1)

    text("Not handled: overlapping communication/computation to eliminate pipeline bubbles")

    # Backward pass: homework exercise

    cleanup()
```

