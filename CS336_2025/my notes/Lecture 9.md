# Lecture 9: Scaling Laws Basics

## Motivation

> **Question:**
> Given massive compute (e.g., 10,000 H100s), how to design the best LM?

Scaling laws give us **predictive, quantitative rules** for how performance (loss) changes with:

* **Data size** (n)
* **Model size** (m)
* **Compute** (C)
* **Hyperparameters** (batch size, LR, etc.)

Goal:
Use **small models** to infer optimal **large-model** configurations — instead of tuning big models directly.

---

## Historical Background

| Period    | Key Work                     | Insight                                                |
| --------- | ---------------------------- | ------------------------------------------------------ |
| 1993–2001 | Banko & Brill (2001)         | Test loss vs dataset size → **log-linear** scaling     |
| 2012      | Kolachina et al.             | Power-law relation between data and performance        |
| 2017      | Hestness et al.              | First large-scale neural scaling laws (MT, LM, Speech) |
| 2020      | Kaplan et al.                | Unified scaling across model, data, compute            |
| 2022      | Hoffmann et al. (Chinchilla) | Refined scaling laws accounting for LR schedule        |

---

## Data Scaling Laws

### Empirical Law

$$
Loss \propto n^{-\alpha}, \quad \text{typically } \alpha \in [0.05, 0.1]$$

* Linear in log–log space ("power law" or "scale-free").
* Observed across domains and architectures.

### Intuition

* Estimation error (mean, regression, etc.) decays polynomially with ( $n$ ).
* Neural networks show **slower** scaling (smaller $\alpha$) than classical theory predicts.

### Theoretical Example

**Mean estimation** with samples ($x_i \sim N(\mu, \sigma^2)$):
$$
\mathbb{E}[(\hat{\mu} - \mu)^2] = \frac{\sigma^2}{n}
\Rightarrow \log(\text{error}) = -\log n + 2\log \sigma
$$
$\Rightarrow$ **Scaling law** with slope −1 on log–log plot.

### Nonparametric Example

In (d)-D space:
$$
\text{Error} \propto n^{-1/d}
\Rightarrow \text{slope} = -\frac{1}{d}$$
→ More flexible models (higher intrinsic dimension) scale slower.

### Intrinsic Dimensionality Hypothesis

$$
\alpha \approx \frac{1}{d_{\text{intrinsic}}}
$$
(rough heuristic: slope depends on the data’s intrinsic dimension.)

---

## Beyond Data Quantity

### (1) **Data Composition**

* Composition affects *offset*, not *slope* of the scaling curve.
  (Kaplan et al. 2021; Hashimoto 2021)

### (2) **Data Repetition**

* Effective data size ( D' < D ) when samples are repeated.
* Repetition reduces useful scaling gain.
  → Data selection should be **adaptive to scale**.

---

## Model Scaling Laws

$$
Loss \propto m^{-\beta}, \quad \beta \in [0.05, 0.1]
$$

* Predictable, log-linear loss decrease as model size increases.
* Diminishing returns beyond a certain scale.

### Combined Model + Data Law

$$
Loss = n^{-\alpha} + m^{-\beta} + C
$$
→ captures tradeoff between dataset and parameter count.

---

## Compute Scaling Laws

Given compute ( $C = n \times m$ ):

$$
Loss \propto C^{-\gamma}
$$

* Enables **compute-optimal training**:
  $$
  N_{\text{opt}} \propto C^{0.73}, \quad D_{\text{opt}} \propto C^{0.27}
  $$
  (Chinchilla, 2022)

* Implication: better to **train smaller models on more data**.

---

## Hyperparameter Scaling

### (1) Architecture

* **Transformers** scale much better than **LSTMs**.
  (Kaplan 2020; Tay 2022)

### (2) Optimizer

* ADAM > SGD for scaling stability.
  (Hestness 2017)

### (3) Depth vs Width

* 1→2 layers: huge improvement; deeper models yield diminishing returns (<10⁷ params).
* Not all parameters equal (e.g., embeddings scale differently).
* Mixture-of-Experts show special scaling patterns.

### (4) Batch Size Scaling

#### Empirical Relation

* There exists a **critical batch size** ( $B_{\text{crit}}$ ):

  * ( $B < B_{\text{crit}}$ ): noise-dominated, increasing batch improves convergence.
  * ( $B > B_{\text{crit}}$ ): diminishing returns.

#### Noise Scale

$$
\mathcal{B}*{\text{noise}} = \frac{\text{Tr}(\Sigma)}{|\mathbb{E}[g]|^2}
\Rightarrow B*{\text{crit}} \sim \mathcal{B}_{\text{noise}}
$$

* Determines when gradient noise stops being dominant.
* As loss decreases, noise scale increases ⇒ larger optimal batch.

#### Scaling Trend

$$
B^* \propto \text{Noise Scale} \propto \frac{1}{L_{\text{target}}}
$$
→ Smaller target loss ⇒ larger optimal batch.

---

### (5) Learning Rate Scaling

* **Naive scaling**: LR needs to decrease with model size.
* **Scale-aware strategies**:

  * **μP (Maximum Update Parametrization)** (Yang et al. 2022; Yao et al. 2024):
    ensures activations and gradient updates remain scale-invariant.
  * Enables predictable LR tuning across scales.

---

## Downstream & Architecture Dependence

* Scaling laws hold cleanly for **pretraining**, but **downstream tasks** often deviate.
* Fine-tuning and data domain mismatch can distort scaling.
* (Tay et al. 2023: downstream scaling ≠ pretraining scaling)

---

## Practical Uses of Scaling Laws

| Use Case                            | Description                                                  |
| ----------------------------------- | ------------------------------------------------------------ |
| **Predict large-model performance** | Fit on small models, extrapolate to large                    |
| **Hyperparameter search**           | Find optimizer, LR, batch trends before large-scale training |
| **Compute allocation**              | Balance model vs data for fixed compute                      |
| **Data curation**                   | Estimate value of more or better data                        |
| **Architecture search**             | Compare scaling exponents across designs                     |

---

## Data–Model–Compute Joint Scaling

From **Rosenfeld (2020)**:
$$
Loss = n^{-\alpha} + m^{-\beta} + C
$$
→ Fit on small runs → predict large-model accuracy.

**Kaplan (2020):**
$$
Loss = m^{-\alpha} + n^{-1/\beta}
$$

**Applications:**

* Decide whether to “buy more GPUs” or “collect more data”.
* Compute-optimal tradeoff between data, model, and runtime.

---

## Chinchilla Revisited

Chinchilla (Hoffmann et al., 2022) improved on Kaplan by:

* Correcting for **LR schedules** (cosine vs linear decay).
* Providing three fit methods:

| Method               | Description                                           | Outcome                                    |
| -------------------- | ----------------------------------------------------- | ------------------------------------------ |
| **1. Min over runs** | Take minimal loss over all runs at each compute level | Power-law lower envelope                   |
| **2. IsoFLOP fit**   | Fix total FLOPs, vary params, take minima             | Smooth power law curve                     |
| **3. Joint Fit**     | Fit entire grid of (n, m) runs                        | Unified law (but flawed in original paper) |

Later re-analysis (Besiroglu et al. 2024) found Method 3 data errors — corrected fits match Methods 1 & 2.

---

## Beyond Training-Optimal Scaling

* Chinchilla finds the **train-optimal** (for given compute) ratio.
  But inference cost often dominates ⇒ **“overtrain” is better**.

| Model       | Tokens / Parameter |
| ----------- | ------------------ |
| GPT-3       | 2                  |
| Chinchilla  | 20                 |
| LLaMA-65B   | 22                 |
| LLaMA-2-70B | 29                 |
| Mistral-7B  | 110                |
| LLaMA-3-70B | 215                |

→ Real-world best models far exceed Chinchilla ratio.

---

## Modern Extensions

* **Diffusion Models** (Gulrajani 2023): IsoFLOP scaling also holds beyond LMs.
* Scaling laws extend to **compute, model size, dataset**, and **architecture choice**.

---

## Recap & Takeaways

| Topic                 | Key Idea                                                             |
| --------------------- | -------------------------------------------------------------------- |
| **Data Scaling**      | Log–log linear (power law), α from 0.05–0.1                          |
| **Model Scaling**     | Power law in params, diminishing returns                             |
| **Compute Scaling**   | Predicts optimal N–D ratio under fixed FLOPs                         |
| **Batch Scaling**     | Controlled by Gradient Noise Scale                                   |
| **LR Scaling (μP)**   | Makes hyperparams scale-invariant                                    |
| **Chinchilla Laws**   | Refit scaling to correct data/compute balance                        |
| **Practical Insight** | Predict performance, allocate compute, and plan training efficiently |

