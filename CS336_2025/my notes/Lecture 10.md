# Lecture 10: Inference

---

## 1. Concept

**Inference:**  
Given a **fixed model**, generate responses given prompts.

- **Training:** the model can see all tokens, so it can be parallelized.  
- **Inference:** tokens must be generated sequentially, hence cannot be parallelized across time.

---

## 2. Arithmetic Intensity

**Arithmetic Intensity (AI):** the ratio of computation to memory transfer,  
i.e. how much compute we perform per byte transferred.  
Higher AI indicates better utilization of compute units.

Two phases of inference:

1. **Prefill:** encode the prompt into vectors (parallelizable)  
2. **Generation:** produce new tokens one by one (sequential)

Let  

$$
S = \text{number of conditioned tokens}, \qquad 
T = \text{number of generated tokens}.
$$

| Stage | Condition | Limiting factor |
|:--|:--|:--|
| Prefill | $T = S$ | Compute-limited |
| Generation | $T = 1$ | Memory-limited |

---

### 2.1 MLP Layers

Approximate arithmetic intensity:

$$
I_{\text{MLP}} \propto B \times T
$$

where $B$ is the batch size.

- In the prefill stage, $B T$ can be large, so it is compute-limited.  
- In the generation stage, $T = 1$; it is difficult to increase $B$ enough, so it becomes memory-limited.

---

### 2.2 Attention Layers

$$
I_{\text{Attn}} = \frac{S T}{S + T}
$$

- For prefill: $T = S \Rightarrow I_{\text{Attn}} = S / 2$ (compute-limited)  
- For generation: $T = 1 \Rightarrow I_{\text{Attn}} < 1$ (memory-limited)

Therefore,

$$
\text{Prefill is compute-limited, while generation is memory-limited.}
$$

For MLP layers, $I_{\text{MLP}} \propto B$.  
For attention layers, $I_{\text{Attn}} \approx 1$; batching does not help.

---

## 3. Reducing the KV Cache

Inference is memory-limited, so we aim to **reduce the size of the KV cache**
while maintaining accuracy.

---

### 3.1 Grouped-Query Attention (GQA)

Idea: $N$ query heads but only $K$ key/value heads,  
each key/value head serves $N / K$ query heads.

The reduction factor in KV cache is

$$
\text{Reduction Factor} = \frac{N}{K}.
$$

Special cases:

- Multi-Head Attention (MHA): $K = N$  
- Multi-Query Attention (MQA): $K = 1$  
- Grouped-Query Attention (GQA): $1 < K < N$

---

### 3.2 Multi-Head Latent Attention (MLA)

Each key/value vector of dimension $N H$ is projected down to a latent space of dimension $C$:

$$
K, V: \mathbb{R}^{N H} \rightarrow \mathbb{R}^{C}.
$$

Example: in DeepSeek V2, $N H = 16384$ and $C = 512$.

MLA is incompatible with RoPE, so an additional 64 dimensions are added for positional encoding:

$$
512 + 64 = 576 \text{ total dimensions.}
$$

---

### 3.3 Cross-Layer Attention (CLA)

Share the key/value pairs **across layers**,  
similar to how GQA shares them across heads.

This approach improves the Pareto frontier between accuracy and latency/memory usage.

---

### 3.4 Local (Sliding-Window) Attention

Attend only to a local window of previous tokens.

Let the window size be $w$ and the number of layers $L$.  
Then the effective receptive field scales approximately as

$$
O(L \times w).
$$

This makes the KV cache independent of the total sequence length $S$.

To maintain accuracy, interleave local layers with global layers,  
for example, one global layer every six local layers.

---

## 4. Speculative Sampling

A method to accelerate generation **without approximation error**.

Use a lightweight **draft model** $p$ to propose tokens,  
and a heavier **target model** $q$ to verify them.

Algorithm outline:

1. Model $p$ proposes $k$ candidate tokens.  
2. Model $q$ evaluates them in parallel.  
3. Accept if $q$ agrees; otherwise revert to sequential generation.

Mathematically, this yields an **exact sample** from $q$:

$$
P_{\text{speculative}}(x) = q(x).
$$

Extensions:

- **Medusa:** the draft model generates multiple tokens simultaneously.  
- **EAGLE:** the draft model uses hidden representations from $q$.

---

## 5. Paged Attention (vLLM)

### Problem

Traditional KV allocation uses contiguous blocks of memory for each request,
which leads to both **internal** and **external fragmentation**:

- Different sequences have different lengths.  
- Some allocated memory remains unused.

### Solution

PagedAttention divides the KV cache into fixed-size **non-contiguous blocks**,  
allowing shared prefixes and copy-on-write updates.

Key properties:

- Prefix blocks can be shared across requests.  
- Supports multiple responses per prompt.  
- Enables efficient memory reuse under dynamic batching.

PagedAttention combines ideas from **operating systems paging** with GPU inference.

---

## 6. Throughput–Latency Trade-off

Let  

$$
\Phi = \text{number of model parameters}, \qquad 
B = \text{batch size}.
$$

Then the total memory requirement is approximately

$$
M_{\text{total}} \approx \Phi + B \times M_{\text{KV}}.
$$

Latency is dominated by memory I/O:

$$
\text{Latency} \propto \frac{M_{\text{total}}}{\text{Bandwidth}},
$$

and throughput is

$$
\text{Throughput} \propto \frac{B}{\text{Latency}}.
$$

| Batch Size | Latency | Throughput | Memory Feasibility |
|:--|:--|:--|:--|
| $B = 1$ | Low | Low | Fits in memory |
| $B = 64$ | Moderate | High | Fits in memory |
| $B = 256$ | High | Saturated | Exceeds memory |

Smaller $B$ gives better latency (useful for time-to-first-token).  
Larger $B$ gives better throughput (useful for batch processing).

---

## 7. Key Takeaways

- Inference is **memory-bound** and highly dynamic.  
- Arithmetic intensity during generation is low.  
- Reducing KV cache (GQA, MLA, CLA, local attention) helps.  
- Speculative sampling uses asymmetry between checking and generation.  
- PagedAttention applies OS paging ideas to GPU KV memory.  
- Real deployment requires balancing throughput and latency.  

---

### Summary

- Training is a one-time cost; inference is repeated many times.  
- Optimize inference efficiency via memory reduction, quantization, and smart batching.  
- Architectural changes (e.g., Mamba, diffusion models) can further improve inference efficiency.
