# Lecture 5: GPU

Anatomy of a GPU: 
GPU has many SM (streaming multiprocessors)
Each SM has many SP (streaming processors)
The closer the memory to the SM, the faster it is.
L1 and shared memory is inside the SM. L2 cache is on die, and global memory are the memory chips next to the GPU.

Execution model of a GPU:
Threads: Threads "do the work" in parallel, SIMT (same instruction multiple threads)
Blocks: Blocks are groups of threads. Each block runs on a SM and can share memory.
Warps: Threads always execute in a "warp" of 32 consecutively numbered threads each.

Memory model of a GPU:
Shared memory: shared among threads in a block, very fast.
Register: private to each thread, very fast.
Local memory: private to each thread, but stored in global memory, very slow.
Global memory: accessible by all threads, very slow.
Constant memory: read-only memory, cached, faster than global memory.

Compute scaling is faster than memory scaling, so the bottleneck is often memory bandwidth.

What makes ML workloads fast?
1. Control divergence: Avoid if-else statements that cause threads in a warp to take different paths.
2. Low precision: Low precision improves arithmetic intensity.
3. Operator fusion to minimize memory access
4. recomputation: compute more to reduce memory access, for example, recompute activations during backpropagation instead of storing them.

5. Memory coalescing and DRAM:
DRAM is the global memory, and it read data of the "burst mode", which will read a chunk of data from the aligned address, but not the only one data you want. 
Memory accesses are *coalesced* if all the threads in a warp fall within the same burst section.
Example: matrix multiplication C = A * B, A, B and C are all stored in row-major order. If we comput each C[i][j] row by row, then each thread in a warp will access the same row of A, but different columns of B, which are coalesced. But if we compute C column by column, then each thread in a warp will access the same column of B, which are not coalesced.
6. Tiling(most important): group and order threads to minimize global memory access.
For example, matrix multiplication C = A * B, we tile the matrices into smaller blocks, and put the blocks to different SMs to get the partial sum of the result. This will reduce the number of accesses to global memory, and increase the use of shared memory.

Summary of methods:
1. Reduce memory accesses
- Coalescing
- Fusion

2. Move memory to shared memory
- Tiling

3. Trade memory for computation
- Quantization
- Recomputation

FlashAttention:
First, use tiling method to comput QK^T in tiles, and store the intermediate results in shared memory.
Second, use the online softmax method to compute the softmax in the tiles which used in computing the multiplication of matrix Q and K.
Online softmax:
m_i = max(m_{i-1}, x_i)
d_i = d_{i-1} * exp(m_{i-1} - m_i) + exp(x_i - m_i)

y_i = exp(x_i - m_V) / d_V
where m_i is the maximum value of the first i elements, d_i is the sum of the exponentials of the first i elements, and y_i is the softmax output of the i-th element.