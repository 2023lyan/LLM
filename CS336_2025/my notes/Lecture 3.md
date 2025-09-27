# Lecture 3: Architectures, Hyperparameters

## Architectures

- Pre-Norm vs Post-Norm

Pre-Norm is a more stable architecture than Post-Norm. It can avoid the exploding gradient problem even without warm-up. But the post-norm architecture is easier to be in the situation of exploding gradients.

New things: "Double" Norm:
Add the LayerNorm before and after the FFN. It's used in the Grok, Gamma2.

An idea is that Post-Norm will break the path of gradients backpropagation of the Residual Connection, which has very good properties. Pre-Norm will not break the path.

- LayerNorm vs RMSNorm

LayerNorm:
y  = (x - E(x)) / sqrt(Var(x + eps))* gamma + beta
It is like the standardization of the input.

RMSNorm:
y = x / sqrt(norm(x)^2 + eps) * gamma
It is like the linear scaling of the input.

RMSNorm is faster because it does not require the mean calculation, and it does not have the bias term beta.

Normalization contains very few FLOPs, but it will cause a lot of memory movement, which increases the runtime of it.

- Bias Term
Most modern architectures do not use the bias term in the linear layer. It makes the training more stable.

- Activation Function
GLU: max(0, xW1) -> max(0, xW1) ⊙ xV, where ⊙ is the element-wise multiplication.
FF.ReGLU:(max(0, xW1) ⊙ xV) W2

GeGLU: (GELU(xW1) ⊙ xV) W2

Swish: x * sigmoid(x)
SwiGLU: (Swish(xW1) ⊙ xV) W2

- Serial vs Parallel
Parallel Layers:
standard: y = x + MLP(LayerNorm(x+ Attention(LayerNorm(x))))
parallel: y = x + Attention(LayerNorm(x)) + MLP(LayerNorm(x))
A few models do parallel layers.

- Position Embedding
Sine, Absolute, Relative, Rotary
RoPE: Rotary Position Embedding

Rotation is determined by the position.

For a d-dimensional vector, we just cut the vector into pairs, then each pair is a 2-dimensional vector, and we rotate it by the position.

f_{q, k}(x_m, m) = R W{q, k}x_m

where R is the rotation matrix.

It will take at the *attention layer*.

## Hyperparameters

### Dimension of FFN
d_{ff} = 4d_{model} for the FFN.

Exception 1: GLU variants:
d_{ff} = 8 / 3 d_{model}
Exception 2: T5

Empirically, the ratio should be between 1 and 10.

### Dimension of Attention
XQ is R^{n * d} -> R^{n * h * d/h} -> R^{h * n * d/h}(The head axis is like a batch axis)

head-dim > model-dim / num-heads

### Aspect Ratio
d_{model} / n_{layers} = 128 for most models.

## Regularization
Dropout and Weight Decay. Weight decay is more popular.

Weight decay interacts with the learning rate. 

## Stability Tricks
For the output softmax, we can use the *z-loss* method, which is sum_i(log(P(xi)) - \alpha * log^2(z(xi))), which adds a penalty term 1e-4 log^2(Z) to make the model more stable.

For the Attention Softmax, we can add 2 Normalization layers before K and Q do the dot product. This is called the *KQ Normalization* trick.

## Attention Heads
- GQA/MQA: Saving inference costs by reducing the number of heads.

- Sparse or sliding window attention: restricting the attention pattern to reduce compute cost

- Exotic SSM stuff.

Arithmetic intensity: Total arithmetic operations / total memory accesses. We want it to be high, bucause memory is expensive in GPU, but the arithmetic operations are cheap.

KV cache: We can cache the K and V vectors in the attention layer to reduce the computation cost. What we really need to compute is the down triangular matrix of the Q dot K^T, if we do not cache the K and V vectors, we need to compute the full matrix.

MQA: Multi-Query Attention, where we use the same K and V for all heads, but different Qs. It reduces the memory accesses because when we do the KV cache, we only need to store one K and V for all heads. And luckily, we can also gained multi-head attention because the Qs are different.

GQA: Grouped Query Attention, where we group the queries into several groups, and each group has its own K and V. MHA and MQA are special cases of GQA.

Sliding Window Attention: We restrict the attention pattern to a sliding window, which reduces the computation cost.