# Lecture 8: Parallelism 2

## Collective Operations

Conceptual primitives used for **distributed programming**.

---

### Key Terms

- **World Size** — total number of devices  
  $$
  \text{World Size} = N
  $$

- **Rank** — unique ID for each device  
  $$
  \text{Rank} \in \{0, 1, 2, \dots, N-1\}
  $$

---

### Collective Communication Primitives

#### 1. **Broadcast**

Send one tensor from the root (usually rank 0) to all other devices:

$$
\text{Rank }0:\; t_0 \quad \Rightarrow \quad
\text{Rank }1:t_0,\;
\text{Rank }2:t_0,\;
\text{Rank }3:t_0,\;
\text{Rank }4:t_0
$$

---

#### 2. **Scatter**

Distribute parts of a tensor from rank 0 to all ranks:

$$
\text{Rank }0:\; (t_0, t_1, t_2, t_3)
\quad \Rightarrow \quad
\begin{cases}
\text{Rank }1: t_0 \\
\text{Rank }2: t_1 \\
\text{Rank }3: t_2 \\
\text{Rank }4: t_3
\end{cases}
$$

---

#### 3. **Gather**

Collect tensors from all ranks to rank 0:

$$
\text{Rank }1:t_0,\;
\text{Rank }2:t_1,\;
\text{Rank }3:t_2,\;
\text{Rank }4:t_3
\quad \Rightarrow \quad
\text{Rank }0:(t_0, t_1, t_2, t_3)
$$

---

#### 4. **Reduce**

Aggregate tensors from all ranks into one (e.g., by summation):

$$
\text{Rank }1:t_0,\;
\text{Rank }2:t_1,\;
\text{Rank }3:t_2,\;
\text{Rank }4:t_3
\quad \Rightarrow \quad
\text{Rank }0:(t_0 + t_1 + t_2 + t_3)
$$

---

#### 5. **All-Gather**

Each rank gets the concatenated results from all ranks:

$$
\begin{aligned}
\text{Input:} &\quad \text{Rank }1:t_0,\; \text{Rank }2:t_1,\; \text{Rank }3:t_2,\; \text{Rank }4:t_3 \\
\text{Output:} &\quad \text{Each rank: } (t_0, t_1, t_2, t_3)
\end{aligned}
$$

---

#### 6. **All-Reduce**

Each rank computes the reduced (e.g., summed) result of all tensors:

$$
\begin{aligned}
\text{Input:} &\quad \text{Rank }1:t_0,\; \text{Rank }2:t_1,\; \text{Rank }3:t_2,\; \text{Rank }4:t_3 \\
\text{Output:} &\quad \text{Each rank: } (t_0 + t_1 + t_2 + t_3)
\end{aligned}
$$

---

#### 7. **Reduce-Scatter**

Each rank contributes partial tensors and keeps a shard of the reduced result:

$$
\begin{aligned}
&\text{Rank }1:(a_0, a_1, a_2, a_3) \\
&\text{Rank }2:(b_0, b_1, b_2, b_3) \\
&\text{Rank }3:(c_0, c_1, c_2, c_3) \\
&\text{Rank }4:(d_0, d_1, d_2, d_3)
\end{aligned}
$$

After reduce-scatter:

$$
\begin{cases}
\text{Rank }1: (a_0+b_0+c_0+d_0,\_,\_,\_) \\
\text{Rank }2: (\_,a_1+b_1+c_1+d_1,\_,\_) \\
\text{Rank }3: (\_,\_,a_2+b_2+c_2+d_2,\_) \\
\text{Rank }4: (\_,\_,\_,a_3+b_3+c_3+d_3)
\end{cases}
$$

---

### Relationship Between Operations

The **All-Reduce** operation can be expressed as:

$$
\text{All-Reduce} = \text{Reduce-Scatter} + \text{All-Gather}
$$

---

**Summary Table**

| Operation | Direction | Result at Each Rank | Purpose |
|------------|------------|---------------------|----------|
| **Broadcast** | One → All | Copy of root tensor | Synchronize data |
| **Scatter** | One → All | Partitioned tensors | Distribute workload |
| **Gather** | All → One | Concatenated tensor | Collect results |
| **Reduce** | All → One | Aggregated tensor | Summation or averaging |
| **All-Gather** | All ↔ All | Concatenated tensors everywhere | Share all data |
| **All-Reduce** | All ↔ All | Aggregated tensor everywhere | Sync gradients |
| **Reduce-Scatter** | All ↔ All | Partitioned reduced result | Efficient parallel reduction |

---


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

