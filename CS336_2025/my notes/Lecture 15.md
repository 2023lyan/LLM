# Lecture 15: Alignment - SFT/RLHF

## 1. Overview

Pretraining enables general capabilities (GPT-3 level), but lacks control.  
Instruction following and safety tuning aim to align models with desired behaviors.

Goals:
1. Collect data that reflects desired behaviors.
2. Train LMs to follow instructions and behave safely.
3. Understand scaling requirements for alignment.

---

## 2. Instruction Tuning (SFT)

### 2.1 Standard Approach
Two-stage pipeline:
1. Supervised Fine-Tuning (SFT): imitation of reference behaviors.
2. Reinforcement Learning from Human Feedback (RLHF): optimization by preference.

### 2.2 Datasets
Representative instruction-tuning datasets:
- **FLAN**: mixture of tasks (classification, summarization, question answering).
- **Alpaca**: simple instruction-response pairs (instruction synthesis).
- **OpenAssistant**: longer, multi-turn, knowledge-rich, often with references.

Differences:
- Variation in style (length, formality, bullet vs. paragraph).
- Presence of references and factual content.
- Scale and safety coverage.

---

## 3. Dataset Characteristics and Effects

### 3.1 Style and Length Effects
- Response length strongly affects human and model preference scores.  
- Long responses often rated higher regardless of factual correctness.

### 3.2 Knowledge and Factuality
Fine-tuning on rare or unknown facts can lead to hallucination.  
Empirical findings:
1. Avoid tuning on tail knowledge.
2. RL-style correctness feedback can stabilize factuality.
3. Knowledge extraction remains unstable and non-localized in LMs.

### 3.3 Safety Tuning
- Goal: reduce harmful, biased, or unsafe outputs.  
- Even small datasets (≈500 safety examples) can substantially improve alignment.
- Challenge: balancing safety vs. over-refusal.

---

## 4. Integrating Instruction Tuning with Pretraining

### 4.1 Motivation
Full SFT is costly; instruction data can be mixed into pretraining to improve efficiency.

### 4.2 Midtraining / Two-Phase Training
1. Phase 1: web-scale pretraining.
2. Phase 2: mix instruction data during pretraining.
3. Short SFT round to refine behavior.

This mitigates catastrophic forgetting and allows scaling of instruction learning.

---

## 5. From Imitation to Optimization

### 5.1 Imitation (SFT)
Fit model distribution:
$$
\hat{p}_\theta(y|x) \approx p^*(y|x)
$$
- Purely generative modeling.
- Relies on labeled examples.

### 5.2 Optimization (RLHF)
Formulate as reward maximization:
$$
\max_p \mathbb{E}_p[R(y, x)]
$$
- Reward $R(y, x)$ quantifies quality or preference.
- The model acts as a policy to be optimized.

### 5.3 Motivation for RLHF
- Scalar feedback (preferences) is cheaper than full reference outputs.
- Experts can more easily *rank* than *author* outputs.
- Post-training data costs dominate total alignment expense.

### 5.4 The G–V Gap
People often prefer outputs that differ from the “ground truth.”  
Preference data captures human intent more directly than supervised data.

---

## 6. RLHF Data Collection

### 6.1 Standard Setup
- Collect pairwise comparisons $(y_{good}, y_{bad})$ for the same prompt $x$.
- Feedback sources: crowdworkers, experts, or LMs.

### 6.2 Human Feedback
- InstructGPT: 40 annotators from ScaleAI / Upwork.
- Complexities: annotation quality, correctness verification, GPT-assisted labeling.

### 6.3 Demographic and Ethical Concerns
- Annotator demographics significantly influence model behavior.
- Crowdsourcing at scale raises fairness, bias, and labor issues.

### 6.4 AI-Generated Feedback
- GPT-4-level models achieve near-human agreement in pairwise ranking.
- Used in systems like Zephyr, Ultrafeedback, and Tulu3.
- Enables low-cost, scalable RLHF pipelines.

### 6.5 Self-Training and Constitutional AI
Models can critique and improve their own outputs based on predefined constitutions (Bai et al., Anthropic).

---

## 7. RLHF Algorithms

### 7.1 PPO (Proximal Policy Optimization)

Goal:
Maximize expected reward while preventing large policy updates.

Policy gradient:
$$
\nabla_\theta \mathbb{E}_{p_\theta}[R(z)] = \mathbb{E}_{p_\theta}[R(z)\nabla_\theta \log p_\theta(z)]
$$

Variants:
- TRPO: linearized trust-region optimization.
- PPO: clips policy ratio to constrain update magnitude.

Used in InstructGPT and “Learning to Summarize from Human Feedback” (Stiennon et al.).

---

### 7.2 DPO (Direct Preference Optimization)

Simplified alternative to PPO:
- Removes explicit reward model and on-policy rollouts.
- Optimizes directly on pairwise preferences using supervised objectives.

DPO objective (conceptually):
- Positive gradient on preferred samples.
- Negative gradient on dispreferred samples, weighted by prediction error.

Advantages:
- Stable and easy to train.
- Comparable performance to PPO without reinforcement overhead.

Widely used in modern open-source models (e.g., Zephyr, Tulu, Mistral).

#### 1. Start from RLHF Objective

$$
\max_{\pi_\theta}
\; \mathbb{E}_{x,y\sim\pi_\theta}
[r(x,y)] 
- \beta D_{KL}\!\big[\pi_\theta(y|x)\|\pi_{\text{ref}}(y|x)\big]
$$

- $r(x,y)$: reward model  
- $ \pi_{\text{ref}} $: reference (SFT) model  
- $ \beta $: KL regularization strength  

---

#### 2. Solve for Optimal Policy

Write as a Lagrangian with constraint $ \sum_y \pi(y|x)=1 $:

$$
\mathcal{L}(\pi,\lambda)
=\sum_y\pi(y|x)\!\left[
r(x,y)
-\beta\log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}
\right]+\lambda(\sum_y\pi(y|x)-1)
$$

Set derivative to zero:

$$
r(x,y)-\beta(\log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}+1)+\lambda=0
$$

Solve for $ \pi(y|x) $:

$$
\boxed{
\pi_r(y|x)=\frac{1}{Z(x)}\,\pi_{\text{ref}}(y|x)
\exp\!\left(\frac{1}{\beta}r(x,y)\right)
}
$$

This is a **Boltzmann distribution** over rewards.

---

#### 3. Implied Reward

Rearranging gives:

$$
\boxed{
r(x,y)=\beta\log\frac{\pi_r(y|x)}{\pi_{\text{ref}}(y|x)}+\beta\log Z(x)
}
$$

→ Reward equals log-ratio between policy and reference model.

---

#### 4. Preference Loss

Given human preference pairs $(x,y^+,y^-)$:

$$
\text{loss}=-\mathbb{E}[\log\sigma(r(x,y^+)-r(x,y^-))]
$$

Substitute $r(x,y)$ from above (constants cancel):

$$
\boxed{
\mathcal{L}_{\text{DPO}}
=-\mathbb{E}_{(x,y^+,y^-)}
\!\left[
\log\sigma\!\Big(
\beta\log\frac{\pi_\theta(y^+|x)}{\pi_{\text{ref}}(y^+|x)}
-\beta\log\frac{\pi_\theta(y^-|x)}{\pi_{\text{ref}}(y^-|x)}
\Big)
\right]
}
$$

---

#### 5. Gradient Form (Update Rule)

$$
\nabla_\theta \mathcal{L}_{\text{DPO}}
=-\beta\,\mathbb{E}
\!\left[
(\sigma(\hat r_\theta(x,y^-))-\sigma(\hat r_\theta(x,y^+)))
(\nabla_\theta\log\pi_\theta(y^+|x)
-\nabla_\theta\log\pi_\theta(y^-|x))
\right]
$$

- Increase prob. of good completions $y^+$
- Decrease prob. of bad completions $y^-$
- Weighted by reward prediction error

---

#### 6. Summary

| Term | RLHF (PPO) | DPO |
|------|-------------|------|
| Reward | Explicit $r_\phi$ | Implied via log-ratio |
| KL | Explicit term | Built-in |
| Training | Reinforcement Learning | Supervised |
| Critic | Needed | None |
| Data | Online rollouts | Offline preference pairs |

---

##### Key Idea

> **DPO $\approx$ RLHF without RL:**  
> Replace policy gradient + reward model  
> with a closed-form, supervised objective directly on human preferences.

---

## 8. Practical Issues in RLHF

### 8.1 Overoptimization
- Excessive reward optimization leads to overfitting.
- Model may exploit reward model biases (reward hacking).

### 8.2 Mode Collapse
- Diversity reduction in outputs.
- Loss of calibrated probabilistic behavior (model becomes deterministic).

### 8.3 Length Effects
- Reinforcement for “longer = better” pattern amplifies verbosity biases.

---

## 9. Summary

**SFT**
1. Works best when extracting latent behaviors, not adding new knowledge.
2. Adding factual data may hurt generalization.
3. Small, well-curated safety and instruction data have large gains.

**RLHF**
1. Enables preference-based optimization beyond imitation.
2. DPO offers simple and scalable implementation.
3. Must guard against overoptimization and loss of diversity.

---

**Key Takeaway:**  
Instruction tuning and RLHF form a two-stage pipeline—  
SFT teaches *what* to do, RLHF optimizes *how well* to do it—  
but both depend critically on data quality, annotator diversity, and controlled optimization.

| Concept         | Definition                        | Analogy in LMs                                 | Pros                                          | Cons                  |                 |
| :-------------- | :-------------------------------- | :--------------------------------------------- | :-------------------------------------------- | :-------------------- | --------------- |
| **On-policy**   | Train on data from current policy | PPO / GRPO sampling new completions            | Stable                                        | Slow, expensive       |                 |
| **Off-policy**  | Train on old / external data      | Replay old completions with importance weights | Efficient                                     | Distribution mismatch |                 |
| **Model-free**  | No environment model              | GRPO / PPO / REINFORCE                         | Simple                                        | High variance         |                 |
| **Model-based** | Learn env model $T(s'             \| s,a)$                                          | “World Model” LM predicting next state/reward | Sample efficient      | Modeling errors |
| **Online**      | Train while generating            | PPO training in RLHF                           | Up-to-date                                    | Expensive             |                 |
| **Offline**     | Train from static dataset         | Supervised fine-tuning (SFT) or offline RL     | Cheap                                         | Data mismatch         |                 |
