# Lecture 4: Mixture of Experts (MoE)

- Dense Models vs Sparse Models
Dense models have all parameters active for each input, while sparse models activate only a subset of parameters for each input. With the same FLOPs, sparse models can have more parameters than dense models.

Intuition: Same FLOP,  more parameters, does better.

MOE can be parallelized across multiple devices, allowing for efficient training and inference.

MOEs are most of the highest-performance open-source models, like Qwen.

Drawback: Messy, only have advantages when you have to split the model. Training objectives are somewhat heuristic, unstable.

Routing function: decides which experts to use for each input, token choise top-k (commonly k = 2), or hashing routing(baseline), RL to learn routes, solve a matching problem.
- Token chooses experts
- Expert chooses tokens
- Global routing via optimization

Top-k routing:
s = Softmax(u\dot e)
g = s(s is top k) or 0(otherwise)
h = \sum(g * FFN_i(u)) + u

More experts but not more FLOPs: cut the experts into smaller but larger number of experts, and a few shared experts.

Load balancing losses: to avoid some experts being overloaded while others are underutilized.
loss = alpha * N * sum(fi*Pi), where fi is the fraction of tokens assigned to expert i, Pi is the probability of expert i being chosen.
DeepSeek v3 variation – per-expert biases: g = s or 0, s is selected by top-k for s + bias_i, bias_i is a learnable parameter for each expert. They call this auxiliary loss free balancing.

