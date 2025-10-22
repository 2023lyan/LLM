# Lecture 2: PyTorch, Resource Accounting

## Memory Accounting

**Tensors Memory:**  
Parameters, gradients, and optimizer states.

---

### Floating-Point Formats

**float32:** 32 bits  
- Bit layout:  
  - Bit 31: sign  
  - Bits 30–23: exponent  
  - Bits 22–0: fraction  

Each value occupies 32 bits = 4 bytes.  
Memory is determined by:
1. Number of values $N$
2. Data type size (e.g., 4 bytes for float32)

---

**float16:** 16 bits  
- Bit 15: sign  
- Bits 14–10: exponent  
- Bits 9–0: fraction  

→ Easy to overflow or underflow due to smaller exponent range.

---

**bfloat16:** 16 bits  
- Bit 15: sign  
- Bits 14–7: exponent  
- Bits 6–0: fraction  

Larger exponent range than float16, but less precision.

---

To get tensor data type info:
```python
float32_info = torch.finfo(torch.float32)
float16_info = torch.finfo(torch.float16)
bfloat16_info = torch.finfo(torch.bfloat16)
````

---

**fp8:** 8 bits

* Bit 7: sign
* Bits 6–4: exponent
* Bits 3–0: fraction

Very rough and lossy.

---

Training with **float32** requires a lot of memory.
Using **float16** or **bfloat16** can save memory and speed up training,
but may be unstable for some models.

---

## Compute Accounting

By default, tensors are stored in **CPU memory**.
To move a tensor to GPU, use:

```python
tensor = tensor.to("cuda")
```

---

## Tensor Storage

Tensors are **pointers** into allocated memory.

To get a slice of a tensor **without copying**:

```python
tensor_slice = tensor[start:end]
```

To create a new tensor **with new memory allocation**:

```python
new_tensor = tensor.contiguous()
new_tensor[0] = 1.0
assert tensor[0] != 1.0  
```

---

## Matrix Multiplication

```python
x = torch.ones(4, 8, 16, 32)
w = torch.ones(32, 2)
y = x @ w
assert y.shape == (4, 8, 16, 2)
```

For tensors with more than 2 dimensions,
the **last two** dimensions are used for matrix multiplication,
and the rest are treated as batch dimensions.

---

## Tensor Einops

**Einops** is a library for manipulating tensors with named dimensions.

For **jaxtyping**:

```python
x: Float[torch.Tensor, "batch seq dim"] = torch.ones(2, 2, 1, 3)
```

This is documentation-only — no runtime enforcement.

---

## Einops Einsum

```python
x: Float[torch.Tensor, "batch seq1 hidden"] = torch.ones(2, 3, 4)
y: Float[torch.Tensor, "batch seq2 hidden"] = torch.ones(2, 3, 4)
# Old way
z = x @ y.transpose(-1, -2)
# New way
z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")
```

The new form is more readable and explicit.

---

## Einops Reduce

```python
x: Float[torch.Tensor, "batch seq hidden"] = torch.ones(2, 3, 4)
# Old way
z = x.sum(dim=-1)
# New way
z = reduce(x, "... hidden -> ...", "sum")
```

Readable and clear — indicates which dimension is reduced.

---

## Einops Rearrange

```python
x: Float[torch.Tensor, "batch seq hidden"] = torch.ones(2, 3, 4)
w: Float[torch.Tensor, "hidden1 hidden2"] = torch.ones(4, 4)

x = rearrange(x, "... (heads hidden1) -> ... heads hidden1", heads=2)
x = einsum(x, w, "... hidden1, hidden1 hidden2 -> ... hidden2")
x = rearrange(x, "... heads hidden1 -> ... (heads hidden1)")
```

* `rearrange`: splits or merges tensor dimensions flexibly.

---

## Tensor Operations FLOPs

**FLOPs:** Floating-point operations
**FLOP/s:** Floating-point operations per second

They are *different* concepts.

---

### Example

**A100 GPU:** 312 teraFLOP/s

For a matrix multiplication $(B, D) @ (D, K)$:

$$
\text{FLOPs} = 2 \times B \times D \times K
$$

Each output element does one multiplication and one addition.

Matrix multiplication is the most expensive operation in deep learning.

---

### Forward Pass Estimate

$$
\text{FLOPs}_{\text{forward}} = 2 \times (\text{tokens}) \times (\text{params})
$$

---

### Model FLOPs Utilization (MFU)

$$
\text{MFU} = \frac{\text{actual FLOP/s}}{\text{theoretical FLOP/s}}
$$

Higher is better.
$\text{FLOP/s}$ depends on hardware and data type (e.g., H100 $\gg$ A100, bfloat16 $\gg$ float32).

Tensor Cores are specialized GPU hardware for matrix multiplication.

---

## Gradient FLOPs

For one token:

$$
\text{FLOPs}_{\text{forward}} = 2 \times (\text{params})
$$

$$
\text{FLOPs}_{\text{backward}} = 4 \times (\text{params})
$$

Total:

$$
\text{FLOPs}_{\text{total}} = 6 \times (\text{params}) \times (\text{data points})
$$

---

## Model Parameters

```python
w = nn.Parameter(torch.randn(input_dim, hidden_dim))
assert isinstance(w, torch.Tensor)
assert type(w.data) == torch.Tensor
```

---

## Parameter Initialization

We want an initialization **invariant to hidden dimension**, so divide by its square root:

$$
w \sim \mathcal{N}!\left(0, \frac{1}{\sqrt{d_{\text{hidden}}}}\right)
$$

---

## Randomness & Reproducibility

Set random seeds to make runs reproducible:

```python
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
```

---

## Data Loading

Avoid loading all data into memory. Use **memory mapping**:

```python
data = np.memmap("data.npy", dtype=np.float32, mode="r", shape=(num_samples, num_features))
```

Modes:

* `"r"` → read-only
* `"r+"` → read/write
* `"w+"` → write (creates new file)
* `"c"` → copy-on-write

---

## Optimizers

| Optimizer    | Concept                                                 |
| ------------ | ------------------------------------------------------- |
| **SGD**      | gradient descent                                        |
| **Momentum** | SGD + exponential moving average of gradients           |
| **AdaGrad**  | SGD + divide by $\text{grad}^2$ (per-parameter scaling) |
| **RMSProp**  | AdaGrad + exponential moving average of $\text{grad}^2$ |
| **Adam**     | RMSProp + momentum                                      |

---

## Memory Estimation

$$
\text{num\_params} = D^2 \cdot n_{\text{layers}} + D
$$

$$
\text{num\_activations} = B \cdot D \cdot n_{\text{layers}}
$$

$$
\text{num\_gradients} = \text{num\_params}
$$

$$
\text{num\_optimizer\_states} = \text{num\_params}
$$

Total memory (in bytes, assuming 4 bytes per value):

$$
\text{total\_memory} = (\text{num\_params} + \text{num\_activations} + \text{num\_gradients} + \text{num\_optimizer\_states}) \times 4
$$

---

## Checkpointing

During training, save model and optimizer state to recover after interruption.

```python
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
}
torch.save(checkpoint, "checkpoint.pth")
```

---

## Mixed Precision Training

**Idea:** Use `float32` by default, but `bfloat16` or `fp8` when possible.

Plan:

1. Use {`bfloat16`, `fp8`} for **forward pass (activations)**
2. Use `float32` for **other computations**



