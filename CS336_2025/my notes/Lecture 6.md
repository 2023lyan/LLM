# Lecture 6: Kernel, Triton

## Benchmarking and Profiling

Benchmarking: How long does it take?
Profiling: Where time is being spent?

Warmup runs: First few runs may be slower due to caching, compilation, etc.
`torch.cuda.synchronize()`: Wait for all threads to finish, which is mendatory for benchmarking GPU code.
`time.time()`: Wall-clock time, affected by other processes.
`torch.cuda.nvtx` and Nsight Systems: More detailed profiling tools for GPU code.
- `with nvtx.range("label"):`: Mark a section of code for profiling.
- `nvtx.range_push("label")` and `nvtx.range_pop()`: Manually mark start and end of a section.

## CUDA Kernel (Written in CUDA/C++)
- fusion: Combining multiple operations into a single kernel to reduce overhead.

- CUDA kernel: CUDA is an extension of C/C++ with APIs for managing GPUs.
Simplified picture: write f(i), CUDA kernel computes f(i) for all i.

- Grid: collection of thread blocks: numBlocks = (2, 4), blockDim = (1, 8)

- Thread block: collection of threads: blockIdx = (0, 1)
Thread: single unit of operation: threadIdx = (0, 3)

- The code uses (blockIdx, blockDim, threadIdx) to determine what to do.

`os.environ["CUDA_LAUNCH_BLOCKING"] = "1"`: Make CUDA tell me what went wrong.

```c++
#include <math.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
__global__ void gelu_kernel(float* in, float* out, int num_elements){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements){
    out[i] = 0.5 * in[i] * (1.0 + tanh(sqrt(2.0 / M_PI) * (in[i] + 0.044715 * pow(in[i], 3))));
    }
}
inline unsigned int cdiv(unsigned int x, unsigned int y) {
    // compute ceil(x / y) 
    return (x + y - 1) / y;
}
torch::Tensor gelu(torch::Tensor x){
    TORCH_CHECK(x.device().is_cuda());
    TORCH_CHECK(x.is_contiguous());
    torch::Tensor y = torch::empty_like(x);
    int num_elements = x.numel();
    int block_size = 1024;
    int num_blocks = cdiv(num_elements, block_size);
    gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), num_elements);
    C10_CUDA_CHECK();
    return y;
}
```
## Triton (Written in Python)
Compiler does more work, can actually outperform PyTorch implementations!
PTX(Parallel Thread Execution) low-level assembly language for NVIDIA GPUs.

```python
def manual_softmax(x: torch.Tensor):
    # M: number of rows, N: number of columns
    M, N = x.shape

    # Compute the max of each row (MN reads, M writes)
    x_max = x.max(dim=1)[0]

    # Subtract off the max (MN + M reads, MN writes)
    x = x - x_max[:, None]

    # Exponentiate (MN reads, MN writes)
    numerator = torch.exp(x)

    # Compute normalization constant (MN reads, M writes)
    denominator = numerator.sum(dim=1)

    # Normalize (MN reads, MN writes)
    y = numerator / denominator[:, None]

    # Total: 5MN + M reads, 3MN + 2M writes
    # In principle, should have MN reads, MN writes (speedup of 4x!)
    return y


def triton_softmax(x: torch.Tensor):
    # Allocate output tensor
    y = torch.empty_like(x)

    # Determine grid
    M, N = x.shape                          # Number of rows x number of columns
    block_size = triton.next_power_of_2(N)  # Each block contains all the columns
    num_blocks = M                          # Each block is a row

    # Launch kernel
    triton_softmax_kernel[(M,)](
        x_ptr=x, y_ptr=y,
        x_row_stride=x.stride(0), y_row_stride=y.stride(0),
        num_cols=N, BLOCK_SIZE=block_size
    )

    return y


@triton.jit
def triton_softmax_kernel(x_ptr, y_ptr, x_row_stride, y_row_stride, num_cols, BLOCK_SIZE: tl.constexpr):
    assert num_cols <= BLOCK_SIZE

    # Process each row independently
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # Read from global memory
    x_start_ptr = x_ptr + row_idx * x_row_stride
    x_ptrs = x_start_ptr + col_offsets
    x_row = tl.load(x_ptrs, mask=col_offsets < num_cols, other=float("-inf"))

    # Compute
    x_row = x_row - tl.max(x_row, axis=0)
    numerator = tl.exp(x_row)
    denominator = tl.sum(numerator, axis=0)
    y_row = numerator / denominator

    # Write back to global memory
    y_start_ptr = y_ptr + row_idx * y_row_stride
    y_ptrs = y_start_ptr + col_offsets
    tl.store(y_ptrs, y_row, mask=col_offsets < num_cols)
```

## Compliation
`torch.compile` can compile PyTorch code to optimized kernels.
