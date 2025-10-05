# Lecture 7: Parallelism 1
All reduce = reduce-scatter + all-gather

- All Reduce: A, B, C, D -> A+B+C+D, A+B+C+D, A+B+C+D, A+B+C+D
- Reduce Scatter: (A0, A1, A2, A3), (B0, B1, B2, B3), (C0, C1, C2, C3), (D0, D1, D2, D3) -> (A0+B0+C0+D0, _, _, _), ( _, A1+B1+C1+D1, _, _), ( _, _, A2+B2+C2+D2, _), ( _, _, _, A3+B3+C3+D3)
- All Gather: (A0+B0+C0+D0, _, _, _), ( _, A1+B1+C1+D1, _, _), ( _, _, A2+B2+C2+D2, _), ( _, _, _, A3+B3+C3+D3)

## Parallelism primitives
- Data parallelism: Naive data parallelism, ZeRO(Zero Redundancy Optimizer) level 1-3
- Model parallelism: Tensor parallelism, Pipeline parallelism
- Activation parallelism: Sequence parallelism,

### Data parallelism
- Naive data parallelism: divide batch(B data) into M parts, each GPU gets B/M data, each GPU has a full model replica, after backward do all-reduce to sync gradients. Only improves throughput(speed), not memory usage.

- ZeRO:
most of tne memory is used to store optimizer states, the core idea is to split up the expensivs parts(state) and use the reduce-scatter method.

P_{os}: 2\phi + 2\phi + K\phi / N_d(to divide the optimizer states)
P_{os+g}: 2\phi + 2\phi / N_d + K\phi / N_d (to divide the optimizer states and gradients)
P_{os+g+p}: 2\phi / N_d + 2\phi / N_d + K\phi / N_d (to divide the optimizer states, gradients and parameters)

Stage 1: only partition optimizer states
- Step 1: each GPU computes gradients on its mini-batch
- Step 2: Reduce-scatter gradients across GPUs
- Step 3: Each GPU updates its partition of optimizer states and model parameters
- Step 4: All-gather model parameters across GPUs

Stage 2: partition optimizer states and gradients
- Step 1: each GPU incrementally goes backward on the computation graph. After computing a layer's gradients, immediately reduce it to the right worker. Once gradients are not needed, free them.
- Step 2: Each GPU updates its partition of optimizer states and model parameters
- Step 3: All-gather model parameters across GPUs

Stage 3: partition optimizer states, gradients and parameters (aka Fully Sharded Data Parallel, FSDP)
- Parameters / gradients are requested / sent and then immediately freed
- The all-gathers happen all at once while forward happens, masking the comm cost.

### Model parallelism

- Layer-wise model parallelism: each layer is on a different GPU, only works for very large models, but low GPU utilization.

- pipeline parallelism: Process 'micro-batches' to keep all GPUs busy. Ratio of bubble time to compute time = (number of stages - 1) / (number of micro-batches)

- Tensor parallelism: split the parameters of each matrix across multiple GPUs. 

Tensor parallelism vs Pipeline parallelism:
Pros:
- no bubble time
Cons:
- more communication cost

### Activation parallelism
Activation memory needed per layer: sbh(34 + 5as/h)
- a: number of attention heads
- b: batch size
- s: sequence length
- h: hidden dimension
- The 5as/h term comes from the attention term incl dropout
- As with FlashAttention, we can omit this term via recomputation

Activation memory needed per layer with tensor parallelism: sbh(10 + 24 / t + 5as/ht)
- t: number of tensor parallelism GPUs
- The remaining 10 term comes from LayerNorm(4sbh), Dropout(2sbh), and input to the mlp and attention (4sbh)

Sequence parallelism:
- Split the sequence length across multiple GPUs
- For tensor parallelism, we need to cut the hidden dimension, for sequence parallelism, we cut the sequence length
- Split up the layer norm/dropout layers along the sequence axis.

Activation memory needed per layer with sequence parallelism, tensor parallelism and recomputation: sbh(34 / t)

Simple rules of thumb:
1. Until your model fits in memory.
2. Then until you run out of GPUs.
3. Tensor parallel = 8 is often optimal