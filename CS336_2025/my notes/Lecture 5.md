# Lecture 5: GPU

## Anatomy of a GPU

* A GPU has many **SMs (Streaming Multiprocessors)**.
* Each SM has many **SPs (Streaming Processors / CUDA cores)**.
* **Memory hierarchy**: the closer to the SM, the faster.

  * **Registers, shared memory, L1 cache**: inside the SM, very fast.
  * **L2 cache**: on die, shared across SMs.
  * **Global memory (DRAM, e.g. GDDR/HBM)**: off-chip, very large but very slow.

---

## Execution model

* **Threads**: smallest execution unit, run in SIMT (same instruction, multiple threads).
* **Block**: a group of threads. Each block runs on one SM and has access to shared memory.
* **Warp**: 32 consecutive threads that always execute in lockstep.

---

## Memory model

* **Registers**: private to each thread, fastest storage.
* **Shared memory**: shared among threads in a block, very fast (on-chip SRAM).
* **Local memory**: private to each thread but allocated in global memory if registers spill → as slow as global memory.
* **Global memory (DRAM)**: accessible by all threads, very high latency.
* **Constant memory**: read-only, cached, faster than global memory for broadcast.

⚠️ Local memory is not a separate physical memory—it’s just global memory used for thread-private data when registers are insufficient.

---

## Compute vs Memory

* Compute throughput has been scaling faster than memory bandwidth.
* Result: memory bandwidth is often the bottleneck for GPU workloads.

---

## Making ML workloads fast

1. **Control divergence**

   * Avoid if/else that cause different threads in a warp to take different execution paths → warps serialize, wasting parallelism.

2. **Low precision**

   * Using FP16 / INT8 reduces memory traffic and increases arithmetic intensity (FLOPs per byte).
   * Modern GPUs have tensor cores optimized for low-precision matmuls.

3. **Operator fusion**

   * Fuse multiple elementwise ops into a single kernel to reduce round-trips to global memory.

4. **Recomputation**

   * Trade computation for memory.
   * Example: in backpropagation, don’t store all activations—recompute them when needed.

5. **Memory coalescing**

   * DRAM is read in bursts (cache-line sized chunks).
   * Memory accesses are *coalesced* if all 32 threads in a warp access addresses within the same burst.
   * Example (row-major layout):

     * Warp accessing **same row, different columns** → addresses contiguous → coalesced.
     * Warp accessing **same column, different rows** → addresses strided by row width → not coalesced.

6. **Tiling (the most important)**

   * Split the output matrix **C** into tiles. Assign each tile to one block running on an SM.
   * Each block loads corresponding tiles of A and B into shared memory.
   * Perform partial sums inside shared memory (fast reuse) before moving on to the next tile.
   * Reduces redundant global memory accesses and improves throughput.

---

## Summary of methods

1. **Reduce memory accesses**

   * Coalescing
   * Fusion

2. **Move memory to shared memory**

   * Tiling

3. **Trade memory for computation**

   * Quantization (low precision)
   * Recomputation

---

## FlashAttention

### Key idea

* Standard attention requires forming (QK^T), which is huge and cannot fit in memory.
* FlashAttention avoids materializing the full matrix by **tiling + online softmax**.

### Steps

1. **Tiling**

   * Load small tiles of (K, V) into shared memory.
   * Compute partial products (S^{(t)} = QK^{(t)T}) tile by tile.

2. **Online softmax**

   * Maintain running maximum (m) and denominator (d).
   * Update incrementally as new tiles are processed:
     [
     m_j = \max(m_{j-1}, x_j), \quad
     d_j = d_{j-1} \cdot e^{m_{j-1} - m_j} + e^{x_j - m_j}
     ]
   * This allows computing the softmax tile-by-tile while staying numerically stable.

3. **Accumulate outputs**

   * For each tile, compute partial results (\text{softmax}(S^{(t)}) V^{(t)}).
   * Combine them with the online softmax rescaling trick.
