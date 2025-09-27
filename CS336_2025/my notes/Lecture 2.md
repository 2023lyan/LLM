# Lecture 2: PyTorch, Resource Accounting

## Memory Accounting
Tensors Memory:
parameters, gradients, and optimizer states.

float32: 32 bits (The default type in PyTorch)
31: sign
30-23: exponent
22-0: fraction

Memory is determined by:
1. number of values
2. data type of each value

float16: 16 bits
15: sign
14-10: exponent
9-0: fraction

Easy to be overflowed or underflowed.

bfloat16: 16 bits
15: sign
14-7: exponent
6-0: fraction
Larger exponent range than float16, but less resolution.

To get the information of a tensor:
```python
float32_info = torch.finfo(torch.float32)
float16_info = torch.finfo(torch.float16)
bfloat16_info = torch.finfo(torch.bfloat16)
```

fp8: 8 bits
7: sign
6-4: exponent
3-0: fraction
very rough.

Training with float32, required lots of memory.
Using float16 or bfloat16 can save memory and speed up training, but requires careful handling. It's instable for some models.

## Compute Accounting
By default, tensors are stored in CPU memory.
To move a tensor to GPU:

# Tensor Storage
tonsors are pointers into allocated memory.

If we want to get a slice of a tensor without copying the data, we can use:
```python
tensor_slice = tensor[start:end]
```

But if we want to create a new tensor with new memory allocation, we can use the contiguous method:
```python
new_tensor = tensor.contiguous()
new_tensor[0] = 1.0
assert tensor[0] != 1.0  
```

# Matrix Multiplication
```python
x = torch.ones(4, 8, 16, 32)
w = torch.ones(32, 2)
y = x @ w
assert y.shape == (4, 8, 16, 2)
```

For the tensor with more than 2 dimensions, the *last two* dimensions are used for matrix multiplication, and the rest are treated as batch dimensions.

# Tensor Einops

Einops is a library for manipulating tensors where dimensions are named.

for jaxtyping:
```python
x: Float[torch.Tensor, "batch seq dim"] = torch.ones(2, 2, 1, 3)
```
this is just documentation, not enforcement.

# Einops Einsum
```python
x: Float[torch.Tensor, "batch seq1 hidden"] = torch.ones(2, 3, 4)
y: Float[torch.Tensor, "batch seq2 hidden"] = torch.ones(2, 3, 4)
# Old way
z = x @ y.transpose(-1, -2) # Low readability because we don't know the meaning of dimensions.

# New way
z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")
# This is more readable and clear.
```

# Eisnop Reduce
```python
x: Float[torch.Tensor, "batch seq hidden"] = torch.ones(2, 3, 4)
# Old way
z = x.sum(dim=-1)  # Less readable, we don't know the meaning of
# dimensions.
# New way
z = reduce(x, "... hidden -> ...", "sum")
# This is more readable and clear.
```

# Einops Rearrange
```python
x: Float[torch.Tensor, "batch seq hidden"] = torch.ones(2, 3, 4)

w = Float[torch.Tensor, "hidden1 hidden2"] = torch.ones(4, 4)

x = rearrange(x, "... (heads hidden1) -> ... heads hidden1", heads=2) # break the last dimension into two dimensions.
x = einsum(x, w, "... hidden1, hidden1 hidden2 -> ... hidden2") 

# We can also combine 2 dimensions into one:
x = rearrange(x, "... heads hidden1 -> ... (heads hidden1)")
```

# Tensor Operations Flops

FLOPs: the number of floating-point operations per second.

FlOP/s: the number of floating-point operations per second.

They are 2 different concepts.

A100: 312 teraFLOP/s

(B, D) @ (D, K), flops = 2 * B * D * K, x[i][j], w[j][k], one multiplication and one addition to the total.

Matrix multiplication is the most expensive operation in deep learning, in general.

Forwoard pass: FlOPs = 2 (# tokens) * (# params)

MFU(Model FLOPs Utilization): mfu = actual_flop/s / promised_flop/s

FLOP/s depends on the hardware and data type. H100 >> A100, bfloat16 >> float32.

Tensor Core of GPU: specialized hardware for matrix multiplication, can speed up training.

# Gradient FLOPs
For one token:

num_forward_flops = 2 * (# params)

num_backward_flops = 4 * (# params)

Total FLOPs = 6 * (# params) * (# data points)

# Model Parameters
w = nn.Parameter(torch.randn(input_dim, hidden_dim))

assert isinstance(w, torch.Tensor)

assert type(w.data) == torch.Tensor

# Parameter Initialization
We want an initialization that is invariant to the hidden dimension. So we devide by the square root of the hidden dimension.

# Note about Randomness
For reproducibility, we can set the random seed differently for each time we use it.
```python
# For PyTorch
SEED = 42
torch.manual_seed(SEED)

# For NumPy
np.random.seed(SEED)

# For Python's random module
random.seed(SEED)
```

# Data Loading
Don't load all data into memory at once, use memmap to lazily load only the accessed parts into memory.
```python
data = np.memmap("data.npy", dtype=np.float32, mode="r", shape=(num_samples, num_features))
# mode: r is read-only, r+ is read/write, w+ is write (creates a new file if it doesn't exist), c is copy-on-write.
```

# Optimizer
AdaGrad:
momentum: SGD + exponential averaging of gradients.
AdaGrad: SGD + averaging by grad^2.
RMSProp: AdaGrad + exponential averaging of grad^2
Adam: RMSProp + momentum.

# Memory
num_params = (D * D * num_layers) + D

num_activations = B * D * num_layers

num_gradients = num_params

num_optimizer_states = num_params

total_memory = (num_params + num_activations + num_gradients + num_optimizer_states) * 4

# Checkpoint
During training, the process can be interrupted, so we need to save the model state.
```python
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
}
torch.save(checkpoint, "checkpoint.pth")
```

# Mixed Precision Training
Solution: Use float32 by default, but use {bfloat16, fp8} when possible.

A concrete plan:
1. Use {bfloat16, fp8} for the forward pass (activation).
2. Use float32 for the rest.

