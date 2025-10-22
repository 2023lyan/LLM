# Lecture 7: Parallelism 1

---

## All-Reduce, Reduce-Scatter, All-Gather

**All-Reduce** can be decomposed as:

$$
\text{All-Reduce} = \text{Reduce-Scatter} + \text{All-Gather}
$$

- **All-Reduce:**  
  $(A, B, C, D) \to (A+B+C+D,\; A+B+C+D,\; A+B+C+D,\; A+B+C+D)$

- **Reduce-Scatter:**  
  $$
  (A_0, A_1, A_2, A_3),\; (B_0, B_1, B_2, B_3),\; (C_0, C_1, C_2, C_3),\; (D_0, D_1, D_2, D_3)
  $$
  $\Downarrow$
  $$
  (A_0+B_0+C_0+D_0,\_,\_,\_),\;
  (\_,A_1+B_1+C_1+D_1,\_,\_),\;
  (\_,\_,A_2+B_2+C_2+D_2,\_),\;
  (\_,\_,\_,A_3+B_3+C_3+D_3)
  $$

- **All-Gather:**  
  $$
  (A_0+B_0+C_0+D_0,\_,\_,\_),\;
  (\_,A_1+B_1+C_1+D_1,\_,\_),\;
  (\_,\_,A_2+B_2+C_2+D_2,\_),\;
  (\_,\_,\_,A_3+B_3+C_3+D_3)
  $$

---

## Parallelism Primitives

- **Data Parallelism:**  
  Naive Data Parallelism, ZeRO (Zero Redundancy Optimizer, Levels 1–3)

- **Model Parallelism:**  
  Tensor Parallelism, Pipeline Parallelism

- **Activation Parallelism:**  
  Sequence Parallelism

---

## Data Parallelism

### Naive Data Parallelism

- Divide a batch of size $B$ into $M$ parts.  
- Each GPU processes $\frac{B}{M}$ samples.  
- Each GPU keeps a **full model replica**.  
- After backward pass, perform **All-Reduce** to synchronize gradients.

✅ Improves **throughput** (speed)  
❌ Does **not** reduce **memory usage**.

---

### ZeRO (Zero Redundancy Optimizer)

Most of the memory is used for **optimizer states**.  
The idea is to **partition expensive components** (states, gradients, parameters)  
and use **Reduce-Scatter / All-Gather** for synchronization.

#### Memory Partition Formulas

Let:
- $\phi$ = model parameter size
- $K$ = optimizer state factor (e.g., 2 for Adam)
- $N_d$ = number of devices (GPUs)

Then:

| Stage | Partitioned Items | Memory per Device |
|--------|------------------|-------------------|
| ZeRO-1 | Optimizer states | $P_{os} = 2\phi + 2\phi + \dfrac{K\phi}{N_d}$ |
| ZeRO-2 | Optimizer states + Gradients | $P_{os+g} = 2\phi + \dfrac{2\phi}{N_d} + \dfrac{K\phi}{N_d}$ |
| ZeRO-3 | Optimizer states + Gradients + Parameters | $P_{os+g+p} = \dfrac{2\phi}{N_d} + \dfrac{2\phi}{N_d} + \dfrac{K\phi}{N_d}$ |

---

### ZeRO Stage 1: Partition Optimizer States

1. Each GPU computes gradients on its local mini-batch.  
2. **Reduce-Scatter** gradients across GPUs.  
3. Each GPU updates its partition of optimizer states and model parameters.  
4. **All-Gather** model parameters across GPUs.

---

### ZeRO Stage 2: Partition Optimizer States and Gradients

1. Each GPU performs backward incrementally.  
   - After computing a layer’s gradient, immediately reduce it to the target GPU.  
   - Free gradients once not needed.  
2. Each GPU updates its partition of optimizer states and model parameters.  
3. **All-Gather** parameters before forward.

---

### ZeRO Stage 3: Partition All (Optimizer States, Gradients, Parameters)

Also known as **Fully Sharded Data Parallel (FSDP).**

- Parameters and gradients are fetched and freed *on demand*.  
- **All-Gathers** overlap with forward computation, hiding communication cost.

---

## Model Parallelism

### Layer-wise Model Parallelism

- Each layer resides on a different GPU.  
- Works only for **very large models**, but results in **low GPU utilization**.

---

### Pipeline Parallelism

Split the model into multiple *stages* and process *micro-batches* to keep all GPUs busy.

Bubble (idle) time ratio:

$$
\frac{T_{\text{bubble}}}{T_{\text{compute}}} = \frac{N_{\text{stages}} - 1}{N_{\text{micro-batches}}}
$$

---

### Tensor Parallelism

Split **parameter tensors** (matrices) across multiple GPUs.

**Comparison:**

| Type | Pros | Cons |
|------|------|------|
| **Tensor Parallelism** | No bubble time | More communication |
| **Pipeline Parallelism** | Less communication | Bubble time |

---

## Activation Parallelism

### Activation Memory per Layer

$$
M_{\text{act}} = s b h (34 + \frac{5 a s}{h})
$$

where:  
- $a$ = number of attention heads  
- $b$ = batch size  
- $s$ = sequence length  
- $h$ = hidden dimension  

The term $\frac{5 a s}{h}$ comes from the attention term (including dropout).  
Like **FlashAttention**, we can eliminate this term via recomputation.

---

### With Tensor Parallelism

$$
M_{\text{act-tp}} = s b h \left( 10 + \frac{24}{t} + \frac{5 a s}{h t} \right)
$$

where $t$ = number of tensor-parallel GPUs.

- The base `10` term comes from:
  - LayerNorm: $4 s b h$
  - Dropout: $2 s b h$
  - Inputs to MLP + attention: $4 s b h$

---

### Sequence Parallelism

- Split the **sequence length** across multiple GPUs (instead of hidden dimension).  
- For Tensor Parallelism → split hidden dimension.  
- For Sequence Parallelism → split sequence length.  
- LayerNorm and Dropout layers are also split along the sequence axis.

With sequence parallelism + tensor parallelism + recomputation:

$$
M_{\text{act-tp-sp}} = \frac{s b h (34)}{t}
$$

---

## Rules of Thumb

1. **Scale** until your model fits in memory.  
2. Then **scale out** until you run out of GPUs.  
3. Tensor Parallel degree $t = 8$ is often near-optimal.

---
