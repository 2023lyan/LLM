# Lecture 3: Architectures, Hyperparameters

## Architectures

### Pre-Norm vs Post-Norm

Pre-Norm is a more stable architecture than Post-Norm. It can avoid the exploding gradient problem even without warm-up. But the post-norm architecture is easier to be in the situation of exploding gradients.

**New things: Double Norm:**  
Add the LayerNorm before and after the FFN. It's used in the Grok, Gamma2.

An idea is that Post-Norm will break the path of gradients backpropagation of the Residual Connection, which has very good properties. Pre-Norm will not break the path.

---

### LayerNorm vs RMSNorm

**LayerNorm:**

$$
y = \frac{x - E(x)}{\sqrt{\mathrm{Var}(x) + \varepsilon}} \cdot \gamma + \beta
$$

It is like the standardization of the input.

**RMSNorm:**

$$
y = \frac{x}{\sqrt{\|x\|^2 + \varepsilon}} \cdot \gamma
$$

It is like the linear scaling of the input.

RMSNorm is faster because it does not require the mean calculation, and it does not have the bias term $\beta$.

Normalization contains very few FLOPs, but it will cause a lot of memory movement, which increases the runtime of it.

---

### Bias Term

Most modern architectures do not use the bias term in the linear layer. It makes the training more stable.

---

### Activation Function

- **GLU:**

  $$
  \text{GLU}(x) = \max(0, xW_1) \to \max(0, xW_1) \odot xV
  $$

  where $\odot$ is the element-wise multiplication.

- **FF.ReGLU:**

  $$
  y = (\max(0, xW_1) \odot xV)W_2
  $$

- **GeGLU:**

  $$
  y = (\mathrm{GELU}(xW_1) \odot xV)W_2
  $$

- **Swish:**

  $$
  y = x \cdot \sigma(x)
  $$

- **SwiGLU:**

  $$
  y = (\mathrm{Swish}(xW_1) \odot xV)W_2
  $$

---

### Serial vs Parallel

**Parallel Layers:**

$$
\text{Standard: } y = x + \mathrm{MLP}(\mathrm{LayerNorm}(x + \mathrm{Attention}(\mathrm{LayerNorm}(x))))
$$

$$
\text{Parallel: } y = x + \mathrm{Attention}(\mathrm{LayerNorm}(x)) + \mathrm{MLP}(\mathrm{LayerNorm}(x))
$$

A few models do parallel layers.

---

### Position Embedding

Sine, Absolute, Relative, Rotary  
**RoPE:** Rotary Position Embedding

Rotation is determined by the position.

For a $d$-dimensional vector, we just cut the vector into pairs, then each pair is a 2-dimensional vector, and we rotate it by the position.

$$
f_{q,k}(x_m, m) = R W_{q,k} x_m
$$

where $R$ is the rotation matrix.

It will take place at the *attention layer*.

---

## Hyperparameters

### Dimension of FFN

$$
d_{\text{ff}} = 4 d_{\text{model}}
$$

Exception 1: **GLU variants**

$$
d_{\text{ff}} = \frac{8}{3} d_{\text{model}}
$$

Exception 2: **T5**

Empirically, the ratio should be between 1 and 10.

---

### Dimension of Attention

$$
XQ \in \mathbb{R}^{n \times d} \rightarrow \mathbb{R}^{n \times h \times \frac{d}{h}} \rightarrow \mathbb{R}^{h \times n \times \frac{d}{h}}
$$

(The head axis is like a batch axis.)

$$
\text{head-dim} > \frac{d_{\text{model}}}{\text{num-heads}}
$$

---

### Aspect Ratio

$$
\frac{d_{\text{model}}}{n_{\text{layers}}} \approx 128
$$

for most models.

---

## Regularization

Dropout and Weight Decay. Weight decay is more popular.

Weight decay interacts with the learning rate.

---

## Stability Tricks

For the output softmax, we can use the *z-loss* method:

$$
L_z = \sum_i \big(\log P(x_i) - \alpha \log^2 z(x_i)\big)
$$

which adds a penalty term $10^{-4} \log^2(Z)$ to make the model more stable.

For the Attention Softmax, we can add two normalization layers before $K$ and $Q$ do the dot product.  
This is called the **KQ Normalization** trick.

---

## Attention Heads

- **GQA / MQA:** Saving inference costs by reducing the number of heads.

- **Sparse or Sliding Window Attention:** restricting the attention pattern to reduce compute cost.

- **Exotic SSM stuff.**

---

### Arithmetic Intensity

$$
\text{AI} = \frac{\text{Total arithmetic operations}}{\text{Total memory accesses}}
$$

We want it to be high, because memory is expensive on GPU, but arithmetic operations are cheap.

---

### KV Cache

We can cache the $K$ and $V$ vectors in the attention layer to reduce the computation cost.  
What we really need to compute is the down-triangular part of $QK^\top$.  
If we do not cache $K$ and $V$, we need to compute the full matrix.

---

### MQA

**Multi-Query Attention**, where we use the same $K$ and $V$ for all heads, but different $Q$ s.  
It reduces memory accesses because when we do the KV cache, we only need to store one $K$ and $V$ for all heads.  
And luckily, we still gain multi-head behavior because the $Q$ s are different.

---

### GQA

**Grouped Query Attention**, where we group the queries into several groups, and each group has its own $K$ and $V$.  
MHA and MQA are special cases of GQA.

---

### Sliding Window Attention

We restrict the attention pattern to a sliding window, which reduces the computation cost.
