# Lecture 16: Alignment - RL 1

## 1. From RLHF to RLVR

### 1.1 Motivation
Reinforcement Learning from Human Feedback (RLHF) aligns models using human preferences.  
However:
- Human preference data is **expensive** and **subjective**.
- Reinforcement Learning from Verifiable Rewards (**RLVR**) replaces human evaluation with **objective, automatically checkable signals** (e.g., correctness in math, code, or logic).

Goal: maintain alignment and reasoning improvement while reducing cost and bias.

---

## 2. Review: DPO and PPO Foundations

### 2.1 DPO: Direct Preference Optimization
DPO reformulates RLHF as a supervised classification problem.

Start from the standard RLHF objective:
$$
\max_\pi \ \mathbb{E}_{x \sim D, y \sim \pi(\cdot|x)} [r(x, y)]
$$

With a nonparametric assumption, the implied reward satisfies:
$$
r(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}
$$

Thus DPO directly optimizes pairwise preferences without on-policy rollouts:
- Positive gradient on preferred responses.
- Negative gradient on dispreferred responses.
- Stable and easy to train.

This leads to performance comparable to PPO but with much lower complexity.

---

### 2.2 PPO Refresher
Proximal Policy Optimization (PPO) updates policies while constraining divergence.

Objective:
$$
L^{\text{CLIP}}(\theta) = 
\mathbb{E}\Big[
\min \big(
r_t(\theta) \hat{A}_t,\ 
\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t
\big)
\Big]
$$
where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$ and $\hat{A}_t$ is the advantage estimate.

In language modeling:
- Each action is a token.
- Reward applies to the full output.
- A KL penalty regularizes deviation from the reference:
$$
r'(x, y) = r(x, y) - \beta D_{\text{KL}}(\pi_\theta(\cdot|x) || \pi_{\text{ref}}(\cdot|x))
$$

---

## 3. GRPO: Group Relative Policy Optimization

### 3.1 Motivation
PPO requires a critic and complex tuning.  
DPO removes the critic but depends on pairwise preference data.

**GRPO** (Group Relative Policy Optimization) removes both:
- No value model.
- No preference data.
- Works directly on **verifiable reward signals**.

---

### 3.2 Algorithm

Key idea: normalize reward *within a batch* (group) and use it as a relative advantage.

Given rollout rewards $\{r_i\}$ in a group:
$$
A_i = \frac{r_i - \bar{r}_{\text{group}}}{\text{std}_{\text{group}} + \epsilon}
$$

Then apply the policy gradient:
$$
\nabla_\theta J(\theta) = 
\mathbb{E}_{\pi_\theta} [A_i \nabla_\theta \log \pi_\theta(y_i | x)]
$$

Steps:
1. Compute reward per rollout.
2. Normalize mean and variance within group.
3. Add KL penalty for stability.
4. Update with gradient ascent.

---

### 3.3 Implementation Simplicity

A minimal PyTorch form:
```python
adv = (reward - reward.mean()) / (reward.std() + 1e-4)
loss = -(adv * logprobs).mean() + beta * kl_penalty
```

No critic, no GAE — simpler, faster, more scalable.

---

### 3.4 Theoretical Considerations

* The variance normalization is **not** a valid baseline → introduces bias.
* Recent work (Liu et al., 2025) proposes an unbiased version using **leave-one-out** baselines.
* Length normalization term helps mitigate excessive chain-of-thought (CoT) verbosity.

---

## 4. RLVR in the Wild: Case Studies

### 4.1 DeepSeek R1

#### Setup

* Base model: DeepSeek-V3.
* RL algorithm: GRPO.
* Rewards:

  * **Accuracy reward**: correctness on reasoning tasks.
  * **Format reward**: correct chain-of-thought (CoT) format tags.

#### Results

* Produces long CoTs with “aha moments”.
* Verifiable rewards make RL efficient.
* GRPO avoids overfitting and is stable even with large-scale reasoning data.

#### Comparison: R1 vs R1-Zero

| Aspect         | R1-Zero        | R1                         |
| :------------- | :------------- | :------------------------- |
| Initialization | None           | SFT from reasoning data    |
| Rewards        | Verifiable     | Verifiable + consistency   |
| Tasks          | Reasoning only | Reasoning + non-verifiable |
| RL Algorithm   | GRPO           | GRPO                       |
| Post-step      | SFT + RLHF     | SFT + RLHF                 |

#### Pipeline

1. SFT on reasoning and general data (≈800k examples).
2. RL with GRPO (verifiable rewards).
3. Distill generated CoT traces into smaller student models.

---

### 4.2 Kimi K1.5

#### Setup

* Released concurrently with R1; comparable to OpenAI o1.
* Uses a mixture of verifiable and preference-based rewards.

#### Data Filtering

* Remove trivial or low-difficulty problems.
* Use “best-of-8” sampling to create harder datasets.
* Maintain coverage of reasoning, math, and code tasks.

#### Training

1. SFT on filtered long-CoT data.

2. RL stage with length control:
   $$
   r_{\text{len}} = \lambda f(\text{length}), \quad \lambda \in [-0.5, 0.5]
   $$

   * Penalizes excessively long chains of thought.
   * Applied late in training to maintain accuracy.

3. Curriculum learning: train easy → hard.

4. Sampling proportional to $(1 - \text{success rate})$.

#### Infrastructure

* Long-CoT rollouts are computationally expensive.
* Kimi uses dynamic rollout batching to improve GPU utilization.

---

### 4.3 Qwen 3

#### Overview

Qwen 3 (by Alibaba) builds on the RLVR recipe and surpasses both R1 and Kimi.

#### Pipeline

1. SFT on filtered, diverse, hard reasoning data.
2. GRPO on small verifiable subsets (≈4k examples).
3. Distillation and post-RLHF alignment.

#### Innovations

* **Thinking-mode fusion:** combine reasoning and non-reasoning data using special “thinking” tokens.
* **Adaptive stopping:** model learns when to end CoT early.
* Empirical finding: general RLHF can slightly degrade reasoning, so reasoning RLVR is trained separately.

---

## 5. Analysis and Discussion

### 5.1 Why RLVR?

Compared to RLHF:

* **Human feedback** is costly, slow, and subjective.
* **Verifiable rewards** (e.g., code passes tests, math answers correct) are objective, scalable, and automatable.

Thus, RLVR allows large reasoning models to self-improve via automated evaluation.

---

### 5.2 Common Pitfalls

1. **Overoptimization** — reward hacking and reduced diversity.
2. **Mode collapse** — model outputs become deterministic.
3. **Length bias** — CoT grows excessively long (controlled by normalization or penalties).

---

### 5.3 Lessons from Modern Models

| Model       | Algorithm           | Reward Type                   | Highlights                           |
| :---------- | :------------------ | :---------------------------- | :----------------------------------- |
| DeepSeek R1 | GRPO                | Verifiable (accuracy, format) | Simple, scalable, state-of-the-art   |
| Kimi K1.5   | PG + length control | Verifiable + regularized      | Curriculum + brevity tuning          |
| Qwen 3      | GRPO                | Verifiable + fusion           | Efficient reasoning and adaptive CoT |

---

## 6. Summary

1. **DPO** simplifies RLHF by removing rollouts and explicit rewards.
2. **PPO** provides stability but at higher cost.
3. **GRPO** removes critics and value models, enabling simple, efficient training.
4. **RLVR** generalizes RLHF to tasks with verifiable rewards, supporting automatic correctness checks.
5. **Recent models (R1, Kimi, Qwen)** confirm that verifiable-reward RL can reach or surpass human-feedback-trained models.

---

**Key Takeaway:**
RLHF depends on subjective preference signals; RLVR leverages *objective correctness signals*.
This shift enables scalable, low-cost, and highly effective reasoning alignment in large language models.



# Appendix — RLVR Mathematical Notes (GRPO vs PPO) and R1 Training Pipeline

## A. Policy Gradient Foundation

For a policy $\pi_\theta(y\mid x)$ and a sequence-level reward $r(x,y)$,  
the REINFORCE gradient is:

$$
\nabla_\theta J(\theta)
= \nabla_\theta \mathbb{E}_{x\sim D,\ y\sim \pi_\theta(\cdot\mid x)}[r(x,y)]
= \mathbb{E}_{x,y}\big[r(x,y)\,\nabla_\theta \log \pi_\theta(y\mid x)\big].
$$

We can subtract a **baseline** $b(x)$ (independent of $y$) without changing the expectation:

$$
\mathbb{E}_{x,y}\big[(r(x,y)-b(x))\,\nabla_\theta \log \pi_\theta(y\mid x)\big]
= \nabla_\theta J(\theta),
$$

because $\mathbb{E}_{y\sim\pi_\theta}[\nabla_\theta \log \pi_\theta(y\mid x)] = 0$.

When rewards are token-level, we use $A_t = Q_t - V_t$ (advantage).  
For sequence-level verifiable rewards, we usually take $A = r - b$.

---

## B. From KL Constraint to PPO

### B.1 KL-Constrained Optimization (TRPO Formulation)

We aim to **constrain policy divergence** from the previous policy $\pi_{\text{old}}$:

$$
\max_\theta\ \mathbb{E}_{y\sim \pi_{\text{old}}(\cdot\mid x)}\!
\Big[\tfrac{\pi_\theta(y\mid x)}{\pi_{\text{old}}(y\mid x)}\,\hat A(y,x)\Big]
\quad \text{s.t.}\quad
\mathbb{E}_x\big[D_{\mathrm{KL}}(\pi_{\text{old}}(\cdot\mid x)\,\|\,\pi_\theta(\cdot\mid x))\big]\le \delta.
$$

Relaxing this constraint gives the **KL-penalized objective** (common in LM PPO):

$$
\max_\theta\ \mathbb{E}_{y\sim \pi_{\text{old}}}\![r_t(\theta)\hat A]
\ -\ \beta\ \mathbb{E}_x\!\big[D_{\mathrm{KL}}(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x))\big],
$$

where $r_t(\theta)=\frac{\pi_\theta}{\pi_{\text{old}}}$.

### B.2 PPO: Clipped Surrogate Objective

PPO simplifies TRPO by clipping the ratio $r_t$ to limit step size:

$$
L^{\mathrm{CLIP}}(\theta)
= \mathbb{E}\big[\min\big(r_t(\theta)\hat A,\ \mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A\big)\big].
$$

An additional KL term can be used for further stability:

$$
L^{\mathrm{PPO}}(\theta)
= L^{\mathrm{CLIP}}(\theta)
-\beta\,\mathbb{E}_x\!\big[D_{\mathrm{KL}}(\pi_\theta\,\|\,\pi_{\text{ref}})\big].
$$

PPO thus balances **trust-region-like stability** with **simple implementation**.

---

## C. GRPO (Group Relative Policy Optimization)

### C.1 From REINFORCE to Group-Normalized Advantage

In RLVR settings (e.g., math/code verification), each input $x$ may have multiple sampled completions $\{y_i\}_{i=1}^m$ with rewards $\{r_i\}$.

Define **group-normalized advantage**:

$$
A_i^{\mathrm{grp}} = \frac{r_i - \bar r}{\mathrm{std}(r) + \epsilon},
\qquad
\bar r = \frac{1}{m}\sum_{j=1}^m r_j.
$$

Then the gradient update is:

$$
\nabla_\theta J(\theta)
\approx
\mathbb{E}\big[A_i^{\mathrm{grp}}\ \nabla_\theta \log \pi_\theta(y_i\mid x)\big]
\ -\ \beta\,\nabla_\theta D_{\mathrm{KL}}(\pi_\theta\,\|\,\pi_{\text{ref}}).
$$

And the scalar loss form:

$$
\mathcal{L}(\theta)
= -\,\mathbb{E}[A_i^{\mathrm{grp}}\ \log \pi_\theta(y_i\mid x)]
\ +\ \beta\,D_{\mathrm{KL}}(\pi_\theta\,\|\,\pi_{\text{ref}}).
$$

Implementation (PyTorch):

```python
adv = (reward - reward.mean()) / (reward.std() + 1e-4)
loss = -(adv * logprobs).mean() + beta * kl_penalty
```

No critic, no GAE — extremely simple and scalable.

---

### C.2 Bias–Variance Analysis

* **Unbiased condition:** requires $b(x)$ independent of sampled $y$.
* **GRPO bias:** normalization depends on all $r_i$, hence indirectly on policy samples — introducing bias.
* **Why it works:**

  * Greatly reduces variance (relative ranking signal).
  * Small bias tolerated due to small steps and KL regularization.
  * Works exceptionally well in verifiable-reward domains.

---

### C.3 Unbiased Improvement (Leave-One-Out Normalization)

To reduce bias, define leave-one-out normalization:

$$
\bar r^{(-i)} = \frac{1}{m-1}\sum_{j\ne i} r_j,
\qquad
\mathrm{std}^{(-i)} = \mathrm{std}({r_j}_{j\ne i}),
$$

then compute:

$$
A_i^{\text{unbiased}} = \frac{r_i - \bar r^{(-i)}}{\mathrm{std}^{(-i)} + \epsilon}.
$$

This removes sample–baseline covariance, yielding an unbiased yet low-variance estimator (Liu et al., 2025).

---

### C.4 Comparison: GRPO vs PPO

| Aspect               | PPO                        | GRPO                         |
| :------------------- | :------------------------- | :--------------------------- |
| Value function       | Requires critic $V_\theta$ | None                         |
| Advantage estimation | GAE or Monte Carlo         | Group normalization          |
| Regularization       | Trust region or KL         | KL penalty only              |
| Data type            | On-policy rollouts         | On-policy verifiable rewards |
| Bias                 | Depends on critic accuracy | Mild due to normalization    |
| Complexity           | Moderate (multi-model)     | Extremely simple (1–2 lines) |

---

## D. Length Normalization and Reward Shaping

### D.1 Empirical Normalization (Length Regularization)

To prevent long, verbose CoTs, apply a penalty to reward:

$$
A_i^{\mathrm{len}} =
\frac{r_i - \lambda,g(\mathrm{len}(y_i)) - \bar r}{\mathrm{std} + \epsilon},
$$

where $g(\cdot)$ can be linear or logarithmic and $\lambda \in \mathbb{R}$ controls strength.
This heuristic is **not potential-based**, so it modifies the optimal policy but improves training stability.

### D.2 Potential-Based Shaping (Theoretical)

If a potential function $\Phi(s)$ exists, we can modify rewards by:

$$
r'*t = r_t + \gamma \Phi(s*{t+1}) - \Phi(s_t),
$$

which keeps the optimal policy invariant.
However, in language modeling (non-Markov, long context), this is impractical,
so empirical normalization (D.1) is preferred.

---

## E. DPO and “Implicit Reward” Connection

In RLHF, with a reference policy $\pi_{\text{ref}}$, the **implied reward** is:

$$
r(x,y) = \beta \log \frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}.
$$

Substituting this into pairwise preference data $(y^+,y^-)$ leads to DPO’s logistic objective —
essentially a contrastive classification problem.
DPO uses *soft preference signals*; GRPO uses *explicit verifiable rewards*.

---

## F. Training Stability Essentials

1. **KL Regularization**
   $$
   \mathcal{L}*{\mathrm{KL}} = \beta, D*{\mathrm{KL}}(\pi_\theta(\cdot\mid x),|,\pi_{\text{ref}}(\cdot\mid x))
   $$
   – Prevents divergence and mode collapse.

2. **Small step sizes and gradient clipping**
   – Keeps updates stable even with biased normalization.

3. **Curriculum and sampling reweighting**
   – Sample proportionally to $(1 - \text{success rate})$
   to maintain a steady learning signal density.

---

## G. DeepSeek R1 Training Pipeline (Text Diagram)

```text
Pretrained Base (DeepSeek-V3)
  ↓
Supervised Fine-Tuning (SFT)
  - Reasoning data + general data (~800K samples)
  ↓
GRPO (Verifiable Reward Reinforcement)
  - Accuracy reward + Format reward
  - Group-normalized advantages
  - KL regularization vs reference model
  - Optional length normalization
  ↓
Distillation
  - Extract CoT reasoning traces
  - Train smaller student models
  ↓
Optional RLHF / Safety Alignment
  - Improves conversational quality and safety
```

Key insights:

* **Verifiable rewards** enable scalable, automatic alignment.
* **GRPO** offers simplicity and throughput benefits.
* **Distillation** transfers reasoning capability to smaller models efficiently.

---

## H. Summary (Mathematical and Practical)

* From KL constraints we derive TRPO → PPO (clipped + KL-regularized).
* GRPO replaces value estimation with **group normalization**, introducing controlled bias but large variance reduction.
* Length normalization or penalties are essential for reasoning tasks.
* In verifiable-reward settings (math, code, logic), **GRPO + KL** forms a minimal yet powerful recipe.
* Empirically proven in DeepSeek R1, Kimi K1.5, and Qwen 3 — achieving o1-level reasoning with simplicity and efficiency.
