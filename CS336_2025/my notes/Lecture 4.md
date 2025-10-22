# Lecture 4: Mixture of Experts (MoE)

## Dense Models vs Sparse Models

- **Dense Models:**  
  All parameters are active for each input.

- **Sparse Models:**  
  Only a subset of parameters (experts) are active for each input.

With the same FLOPs, sparse models can have *more parameters* than dense models — hence potentially better performance.

> **Intuition:**  
> Same FLOPs → more parameters → better accuracy.

---

MoE architectures can be **parallelized** across multiple devices, allowing for efficient distributed training and inference.

Many state-of-the-art open-source models (e.g., **Qwen**) use MoE.

---

### Drawbacks

- Training objectives are somewhat heuristic and unstable.  
- MoEs show advantages mainly when you *must* split the model across devices.  
- Implementation is more complex ("messy").

---

## Routing Function

The **routing function** decides which experts to use for each input.

Routing can be implemented in multiple ways:

- **Token chooses experts**  
- **Expert chooses tokens**  
- **Global routing** via optimization  

Common routing strategies:

- **Top-$k$ routing**
- **Hash-based routing** (baseline)
- **Reinforcement Learning (RL)** to learn routes
- **Optimization-based matching** between tokens and experts

---

### Top-$k$ Routing

Let:
- $u$ = token representation (input)
- $e_i$ = expert embeddings

Compute the routing logits:

$$
s = \mathrm{Softmax}(u \cdot e)
$$

Then select the top-$k$ experts (commonly $k = 2$):

$$
g_i =
\begin{cases}
s_i, & \text{if expert } i \text{ is in top-}k \\
0, & \text{otherwise}
\end{cases}
$$

The output is the mixture of top experts plus a residual connection:

$$
h = \sum_i g_i \cdot \mathrm{FFN}_i(u) + u
$$

---

### FLOPs Efficiency

To keep **FLOPs constant** while increasing parameter count:
- Use **more experts**, each **smaller** (i.e., fewer parameters per expert).
- Include a few **shared experts** to improve generalization.

Thus, you can scale model size *without* increasing compute cost.

---

## Load Balancing Loss

Without constraints, some experts may be overloaded while others are underutilized.

A common **load balancing loss** encourages uniform usage:

$$
\mathcal{L}_{\text{balance}} = \alpha N \sum_i f_i P_i
$$

where:
- $f_i$: fraction of tokens assigned to expert $i$
- $P_i$: probability of expert $i$ being chosen
- $N$: total number of tokens
- $\alpha$: weighting coefficient

This regularization prevents collapse into a few dominant experts.

---

## DeepSeek V3 Variation — Per-Expert Biases

To further stabilize routing, DeepSeek v3 adds **per-expert learnable biases**.

Routing scores are computed as:

$$
s'_i = s_i + b_i
$$

where $b_i$ is a learnable bias for each expert.

Then select top-$k$ experts based on $s'_i$:

$$
g_i =
\begin{cases}
s_i, & \text{if } i \in \text{Top-}k(s'_i) \\
0, & \text{otherwise}
\end{cases}
$$

They call this approach **auxiliary-loss-free balancing**,  
because it implicitly achieves better expert load balance *without* adding an explicit balancing loss term.

---

## Summary

| Concept | Formula / Idea | Notes |
|----------|----------------|-------|
| Routing logits | $s = \mathrm{Softmax}(u \cdot e)$ | Token–expert affinity |
| Gating | Top-$k$ selection of $s$ | Determines active experts |
| Output | $h = \sum_i g_i \cdot \mathrm{FFN}_i(u) + u$ | Combines selected experts |
| Load balancing loss | $\mathcal{L} = \alpha N \sum_i f_i P_i$ | Encourages uniform expert usage |
| DeepSeek v3 bias | $s'_i = s_i + b_i$ | Implicit balancing |

---

**Key takeaways:**
- MoE allows scaling parameters *without increasing FLOPs.*
- Routing is critical for performance and stability.
- Load balancing ensures experts are used evenly.
- Modern variants (like DeepSeek v3) simplify balancing via per-expert biases.
