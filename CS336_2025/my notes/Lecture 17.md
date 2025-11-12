# Lecture 17: Alignment - RL 2

_Last lecture_: overview of reinforcement learning from verifiable rewards (RLVR).  
_This lecture_: deeper look at the **mechanics of policy gradient** and how GRPO is derived and implemented.

---

## 1. RL Setup for Language Models

### 1.1 Core Components

- **State** $s$: prompt + generated response so far  
- **Action** $a$: generate the next token  
- **Transition** $T(s'|s,a)$: deterministic, since $s' = s + a$

In language models:
- Each token generation is a deterministic transition (unlike robotics).
- Planning or test-time computation can use the same LM autoregressive process.

**Policy:** $\pi(a|s)$ — the language model itself (parameterized by $\theta$).

**Rollout (trajectory):**  
$s_0 \to a_0 \to s_1 \to a_1 \to \dots \to s_T \to R$

**Objective:** maximize expected reward  
$$
J(\theta) = \mathbb{E}_{s,a\sim\pi_\theta}[R(s,a)]
$$
where the expectation is taken over prompts $s$ and responses $a$.

### 1.2 Reward Types

Focus on **outcome rewards** that depend on the entire response, and **verifiable rewards**, which are deterministic to compute (e.g., correctness in math or code).

Examples:
- “Therefore, the answer is 3 miles.” → correctness reward = 1  
- Non-verifiable reward (e.g., helpfulness) → learned from human preference (RLHF)

**Discounting and bootstrapping** are less meaningful for text-based outcomes.

---

## 2. Policy Gradient Foundations

Let $a$ denote the entire generated response for simplicity.

We aim to maximize:
$$
\mathbb{E}[R] = \int p(s) \pi(a|s) R(s,a)
$$

Taking the gradient:
$$
\nabla_\theta \mathbb{E}[R]
= \int p(s)\nabla_\theta \pi(a|s)R(s,a)
= \int p(s)\pi(a|s)\nabla_\theta \log \pi(a|s)R(s,a)
$$

Thus:
$$
\nabla_\theta J(\theta)
= \mathbb{E}_{s,a\sim\pi_\theta}\big[\nabla_\theta \log \pi_\theta(a|s)\, R(s,a)\big]
$$

### 2.1 Naive Policy Gradient

Algorithm:
1. Sample prompt $s$ and response $a \sim \pi(a|s)$  
2. Update:
   $$
   \theta \leftarrow \theta + \eta \nabla_\theta \log \pi(a|s) R(s,a)
   $$

If $R(s,a) \in \{0,1\}$ (e.g., “correct or incorrect”):
- Only correct responses contribute positive updates.
- The dataset effectively shifts as the policy changes.

**Problem:** extremely high variance and sparse signal.

In RLHF, reward models produce smoother (continuous) reward distributions.

---

## 3. Variance Reduction: Baselines and Advantages

We know:
$$
\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi(a|s) R(s,a)]
$$

But this estimator has high variance. We can subtract a **baseline** $b(s)$:
$$
\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi(a|s) (R(s,a) - b(s))]
$$

since:
$$
\mathbb{E}_{a\sim\pi}[\nabla_\theta \log \pi(a|s)] = 0
$$

### 3.1 Example: Two States

| State | Action | Reward |
|:--|:--|:--|
| s₁ | a₁ | 11 |
| s₁ | a₂ | 9 |
| s₂ | a₁ | 0 |
| s₂ | a₂ | 2 |

We’d like to prefer $(s₁,a₁)$ and $(s₂,a₂)$,  
but naive gradients treat raw rewards 11 > 2, ignoring state context.

Add baselines:
- $b(s_1)=10$
- $b(s_2)=1$

Variance reduces dramatically:
- Naive variance ≈ 4.743  
- With baseline ≈ 0.957

Thus $b(s)$ can dramatically reduce gradient noise.

### 3.2 Optimal Baseline

For one-parameter models:
$$
b^*(s) = \frac{\mathbb{E}[(\nabla_\theta \pi(a|s))^2 R]}{\mathbb{E}[(\nabla_\theta \pi(a|s))^2]}
$$

Hard to compute → use heuristic:
$$
b(s) \approx \mathbb{E}[R|s]
$$

---

### 3.3 Advantage Function

Define:
- $V(s) = \mathbb{E}[R|s]$ (expected reward from $s$)
- $Q(s,a) = \mathbb{E}[R|s,a]$

Then:
$$
A(s,a) = Q(s,a) - V(s)
$$

If $b(s)=V(s)$, the baseline-adjusted reward equals the **advantage**:
$$
R - b(s) = A(s,a)
$$

Hence policy gradient with baselines is equivalent to **advantage-weighted updates**.

---

## 4. GRPO (Group Relative Policy Optimization)

A simplified PPO that removes the critic and uses **group-level normalization** as a built-in baseline.

- Each prompt (state) has multiple sampled responses (actions).
- Within a group of responses, normalize their rewards.

Algorithm steps:

1. **Generate responses** for each prompt.  
2. **Compute rewards** per response (verifiable or outcome-based).  
3. **Normalize** within the group:
   $$
   A_i = \frac{r_i - \bar r}{\text{std}(r) + \epsilon}
   $$
4. **Update** with policy gradient:
   $$
   \mathcal{L} = -\mathbb{E}[A_i \log \pi(a_i|s)] + \beta D_{\mathrm{KL}}(\pi_\theta\|\pi_{\text{ref}})
   $$

No critic, no GAE → simpler and more scalable.

Implementation:

```python
adv = (reward - reward.mean()) / (reward.std() + 1e-4)
loss = -(adv * logprobs).mean() + beta * kl_penalty
```

#### Algorithm — Iterative Group Relative Policy Optimization (GRPO)

**Input:**  
- Initial policy model $ \pi_{\theta_{\text{init}}} $  
- Reward model $ r_{\varphi} $  
- Task prompts $ \mathcal{D} $  
- Hyperparameters $ \epsilon, \beta, \mu $

---

1. Initialize policy model  
   $ \pi_\theta \leftarrow \pi_{\theta_{\text{init}}} $

2. **For** iteration = 1 … L **do**  
   1. Set reference model  
      $ \pi_{\text{ref}} \leftarrow \pi_\theta $  
   2. **For** step = 1 … M **do**  
      1. Sample a batch $ \mathcal{D}_b $ from $ \mathcal{D} $  
      2. Update old policy model  
         $ \pi_{\theta_{\text{old}}} \leftarrow \pi_\theta $  
      3. For each question $ q \in \mathcal{D}_b $, sample $ G $ outputs  
         $ \{ o_i^G \}_{i=1}^{G} \sim \pi_{\theta_{\text{old}}}(\cdot | q) $  
      4. Compute rewards $ \{ r_i^G \}_{i=1}^{G} $ for each output $ o_i $  
         by running reward model $ r_{\varphi} $  
      5. Compute group-relative advantage  
         $ A_{i,t} $ for the $t$-th token of $ o_i $  
      6. **For** GRPO iteration = 1 … μ **do**  
         1. Update policy model $ \pi_\theta $  
            by maximizing the GRPO objective
   3. Update reward model $ r_{\varphi} $  
      through continuous training using a replay mechanism 

---

**Output:** Final policy $ \pi_\theta $


---

## 5. Code Walkthrough Highlights

### 5.1 Task Example: Sorting Numbers

* **Prompt:** `[3, 1, 0, 2]`
* **Response:** `[0, 3, 1, 2]`
* **Reward:** based on how close the response is to the sorted sequence `[0,1,2,3]`

Two reward functions:

1. `sort_distance_reward`: +1 per position match
2. `sort_inclusion_ordering_reward`: partial credit for inclusion and ordering

---

### 5.2 Simple LM Model

Non-autoregressive model with separate encoder/decoder matrices:

$$
\text{encoded} = \text{einsum}(\text{embeddings}, W_\text{enc})
$$
$$
\text{decoded} = \text{einsum}(\text{encoded}, W_\text{dec})
$$
$$
\text{logits} = \text{einsum}(\text{decoded}, W_\text{embed})
$$

Responses sampled via:
$$
a \sim \text{softmax}(\text{logits})
$$

---

### 5.3 Reward and Delta Computation

`compute_deltas()` modes:

| Mode                   | Formula                                   | Description                  |
| :--------------------- | :---------------------------------------- | :--------------------------- |
| `"rewards"`            | $\delta = R$                              | raw rewards                  |
| `"centered_rewards"`   | $\delta = R - \bar R$                     | subtract mean per prompt     |
| `"normalized_rewards"` | $\delta = (R - \bar R)/(\text{std}+1e-5)$ | normalize within prompt      |
| `"max_rewards"`        | zero out non-max                          | only update on best response |

---

### 5.4 Loss Functions

| Mode          | Formula                                                                        | Note              |
| :------------ | :----------------------------------------------------------------------------- | :---------------- |
| `"naive"`     | $-\mathbb{E}[\delta \log \pi]$                                                 | REINFORCE         |
| `"unclipped"` | $-\mathbb{E}[\frac{\pi}{\pi_{\text{old}}}\delta]$                              | raw PPO           |
| `"clipped"`   | $-\mathbb{E}[\min(r_t\delta, \mathrm{clip}(r_t,1-\epsilon,1+\epsilon)\delta)]$ | PPO with clipping |

Also possible to add **KL penalty**:
$$
\text{KL}(p|q) = \mathbb{E}_p[q/p - \log(q/p) - 1]
$$

---

### 5.5 Freezing Parameters

When computing ratios $\frac{p(a|s)}{p_{\text{old}}(a|s)}$,
the denominator (`p_old`) must be treated as **frozen** (no gradient).

```python
with torch.no_grad():
    p_old = torch.sigmoid(w)
ratio = p / p_old
ratio.backward()
```

---

## 6. Experiments and Observations

### 6.1 Raw Rewards

* Unstable learning, poor convergence.

### 6.2 Centered Rewards

* Helps by assigning **negative gradients** to below-average samples.
* No update when all responses have equal reward.
* Converges better but can still get stuck in local optima.

### 6.3 Normalized Rewards

* Similar to centered, but divides by std.
* In real GRPO (e.g., DeepSeek R1), full normalization often avoided (biases long outputs).

Conclusion:

> RL optimization is non-trivial.
> Without careful baselines or normalization, training easily collapses or saturates.

---

## 7. Summary and Takeaways

* Reinforcement learning is **key to surpassing human capabilities** in reasoning models.
* “If you can measure it, you can optimize it.”
* Policy gradient is conceptually simple but variance reduction is crucial.
* RL training systems are much more complex than pretraining — multiple models, inference-heavy workloads.