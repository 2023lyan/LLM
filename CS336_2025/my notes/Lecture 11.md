# Lecture 11: Scaling Law 2

---

## 1. Motivation

**Key questions**

- What are best practices for scaling and hyperparameter tuning of LMs?
- Does the Chinchilla scaling approach work in practice?
- Can we reduce compute cost when fitting scaling laws?
- Which parametrizations (architectures or inits) scale most stably?

---

## 2. Scaling in Practice

Recent models with public scaling details (2022–2025):

1. **CerebrasGPT** — Chinchilla recipe with muP  
2. **MiniCPM** — careful scaling + muP stabilization  
3. **DeepSeek** — empirical batch/LR scaling (no muP)

Later models (LLaMA 3, Hunyuan-1, MiniMax-01) mainly use isoflop-style scaling.

---

## 3. Maximum Update Parametrization (muP)

**Idea:** keep activations and their updates scale-invariant as model width grows.

Goal — **scale-invariant hyperparameter tuning**  
so that optimal learning rate, initialization, etc. remain constant across widths.

Two key conditions:

1. **A1 (Initialization):** activations at init stay $\Theta(1)$ per neuron  
   → equivalently, $\|h_\ell\|_2 = \Theta(\sqrt{n_\ell})$
2. **A2 (Update):** activation changes after one gradient step stay $\Theta(1)$ per neuron  
   → equivalently, $\|\Delta h_\ell\|_2 = \Theta(\sqrt{n_\ell})$

### Derivation (simplified)

For a linear layer $h_\ell = W_\ell h_{\ell-1}$, with  
$W_\ell \sim \mathcal{N}(0, \sigma^2 I_{n_\ell \times n_{\ell-1}})$:

To keep $\|h_\ell\|_2 = \Theta(\sqrt{n_\ell})$, choose

$$
\sigma = \Theta\!\left( \frac{1}{\sqrt{n_{\ell-1}}} 
  \min\!\left(1, \sqrt{\frac{n_\ell}{n_{\ell-1}}}\right)\right).
$$

For SGD updates  
$\Delta W_\ell = -\eta_\ell \nabla_{W_\ell}\ell$,  
we want $\|\Delta h_\ell\|_2 = \Theta(\sqrt{n_\ell})$, implying

$$
\eta_\ell = \Theta\!\left(\frac{n_\ell}{n_{\ell-1}}\right).
$$

For Adam (with adaptive normalization),
$$
\eta_\ell = \Theta\!\left(\frac{1}{n_{\ell-1}}\right).
$$

Thus:
| Param | Standard | muP |
|:--|:--|:--|
| Initialization | $\Theta(1/\sqrt{n_{\ell-1}})$ | $\Theta(1/\sqrt{n_{\ell-1}})\min(1,\sqrt{n_\ell/n_{\ell-1}})$ |
| Learning rate (SGD) | $\Theta(1)$ | $\Theta(n_\ell/n_{\ell-1})$ |
| Learning rate (Adam) | $\Theta(1)$ | $\Theta(1/n_{\ell-1})$ |

**Key insight:** μP controls both activation magnitudes (via initialization)  
and update magnitudes (via learning rate), ensuring consistent training dynamics across widths.

---

## 4. Case Study 1 — CerebrasGPT

- Model sizes 0.1 B – 13 B, trained with **Chinchilla** + **muP**.
- muP makes hyperparameters nearly **scale-invariant**.  
- Stable hyperparameter sets achieved across scales.  
- Empirical initialization:  
  - $\text{scale\_emb}=10$, $\text{init\_std}=0.08$, $\text{lr}=6\times10^{-3}$.

**Finding:** muP leads to smoother scaling curves and more stable convergence.

---

## 5. Case Study 2 — MiniCPM (2024)

- Small, high-performance LM from Tsinghua (1–2.5 B params).  
- Outperforms most 2 B models; rivals modern 7 B models.  

### muP settings

$\text{scale\_emb}=12$, $\text{scale\_depth}=1.4$, $\text{init\_std}=0.1$, $\text{lr}=0.01$.

### Strategy

- Use **muP** for initialization.  
- Fix **aspect ratio**, scale up total size.  
- Fit optimal **batch**, **LR**, **token-to-model size** ratios via scaling analysis.

---

### 5.1 Optimal Batch and LR

Following **Kaplan 2020**, the optimal batch size increases polynomially as loss decreases.

- Empirical fits show clean trends.  
- Optimal learning rate remains roughly constant under muP.

---

### 5.2 Data–Model Trade-off

From **Chinchilla**, we need to train full runs (not early-stop)  
to fit the scaling law — cost $\mathcal{O}(n^2)$.

MiniCPM solution: **WSD learning rate schedule**

- Split LR into **Warmup–Stable–Decay** phases.  
- Restart training after stable phase for Chinchilla-style fitting.  
- Decay step ≈ 10 % per phase.

---

### 5.3 Chinchilla Methods

- **Method 1:** Lower-envelope fit  
- **Method 3:** Joint fit (MiniCPM uses this)

Findings:

- Diminishing returns with data are mild.  
- Very high data–to–model ratio ≈ 192.  
- Matches trend of **LLaMA 3**, which also uses higher data ratios than earlier “20×size” rule.

---

## 6. Case Study 3 — DeepSeek (2024)

- 7 B and 67 B parameter LMs with detailed scaling analysis.  
- Comparable to **LLaMA 2** of similar size.

### Strategy

- No muP; directly estimate optimal batch / LR from small-scale runs.  
- Fit near-optimal points (within 0.25 % of min loss).  
- Use **WSD-style** learning rate (fast warmup + two decays of 10 %).  
- Apply **Chinchilla Method 2** = IsoFLOPs analysis.

### Result

Scaling fits accurately predict final loss across sizes.

---

## 7. Recent Large-Scale Laws

| Model | Method | Ratio (data : param) | Notes |
|:--|:--|:--|:--|
| **LLaMA 3 (2024)** | IsoFLOPs | 39 : 1 | compute–to–downstream scaling |
| **Hunyuan-1 (2024)** | IsoFLOPs (MoE) | 96 : 1 | active-parameter scaling |
| **MiniMax-01 (2025)** | Chinchilla Method 1 | – | combines architecture + data scaling |

---

## 8. Summary of Scaling Recipes

| Model | Key Techniques |
|:--|:--|
| **CerebrasGPT** | muP + Chinchilla formula |
| **DeepSeek** | empirical batch/LR scaling + IsoFLOPs + piecewise LR |
| **MiniCPM** | muP + piecewise LR + Chinchilla Method 3 |
| **LLaMA 3 / Hunyuan / MiniMax** | mostly IsoFLOPs |

---

## 9. Validating muP

### What is muP robust to?

| Category | Robustness |
|:--|:--|
| Nonlinearities (SwiGLU, SqReLU) | same optimal LR, minor gains |
| Batch size | mostly stable, theory lacks batch term |
| Initialization (SP Unembedding, Zero Query) | compatible |
| RMSNorm gains | breaks muP, but can disable gain with little loss |
| Exotic optimizers (Lion, sign-based) | uncertain |
| Strong weight decay (≈ 0.1) | major failure mode |

### Practical value

muP generally improves stability and simplifies tuning;  
standard parametrizations tend to diverge more easily.

---

## 10. Scaling in the Wild — Practical Challenges

1. Choosing model architecture hyperparameters (width, depth, etc.)  
2. Choosing optimizer hyperparameters (LR, batch)  
3. Fitting full Chinchilla sweeps is compute-expensive.

**Practical solutions:**

1. Assume stability (use muP).  
2. Tune LR/batch on small models and extrapolate.  
3. Use alternative LR schedules (WSD-like piecewise linear).

---

## 11. Takeaways

- muP provides approximately **scale-invariant hyperparameter tuning**.  
- WSD schedules make Chinchilla-style fits cheaper.  
- MiniCPM and DeepSeek demonstrate scaling laws beyond earlier rules (20×).  
- IsoFLOPs remains the dominant method for recent frontier models.  
- Empirical validation confirms scaling predictions are increasingly reliable.

---
