# Lecture 12: Evaluation

## Concept

Given a **fixed model**, we want to know: *How “good” is it?*

Evaluation is not a mechanical metric — it fundamentally shapes how we think about progress in AI.

---

## 1. Benchmark Scores

### Recent Model Benchmarks

* DeepSeek-R1, Llama-4, OLMo-2, etc., are all evaluated on **similar but not identical** benchmarks.
* Common benchmarks: **MMLU, MATH, GSM8K, HellaSwag**, etc.
* Important: scores are not perfectly comparable due to variations in setups, prompts, and evaluation pipelines.

> “What do these numbers actually mean?”

### Benchmark Examples

* [HELM capabilities leaderboard](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard)
* [Artificial Analysis](https://artificialanalysis.ai/)
* [OpenRouter Model Rankings](https://openrouter.ai/rankings)
* [Chatbot Arena](https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard)

**Note:**
Performance must be weighed against **cost** (compute, inference, context length).
A model may look “worse” but be cheaper, safer, or more usable.

---

## 2. How to Think About Evaluation

Evaluation ≠ throw prompts + average scores.
It’s a **design problem** depending on your **goal**:

### Motivations

1. **Product Decision:** Company comparing model A vs B for their task.
2. **Research Measurement:** Assess model’s core capabilities (e.g., reasoning).
3. **Policy / Risk:** Understand benefits and harms.
4. **Model Dev Feedback:** Tune future iterations.

---

### Framework for Designing Evaluation

| Step                  | Key Questions                                                                                          |
| --------------------- | ------------------------------------------------------------------------------------------------------ |
| **1. Inputs**         | What use cases? Any hard-tail cases? Are they realistic or synthetic?                                  |
| **2. Model Call**     | How do we prompt? Is CoT / RAG / tools enabled? Are we evaluating the LM or the *system* around it?    |
| **3. Outputs**        | How do we grade? What metrics? Do we account for asymmetric errors (e.g., hallucinations in medicine)? |
| **4. Interpretation** | How do we interpret metrics? Is 91% “good enough”? Are we evaluating methods or deployed systems?      |

---

## 3. Perplexity

### Definition

A **language model** defines ( $p(x)$ ).
**Perplexity** measures how surprised the model is by real data ( D ):

$$
\text{Perplexity}(D) = \left( \frac{1}{p(D)} \right)^{1/|D|}
$$

Lower = better predictive fit.

### Classic Datasets

* **Penn Treebank (WSJ)**
* **WikiText-103 (Wikipedia)**
* **One Billion Word Benchmark (WMT11)**

### Key Points

* GPT-2: trained on WebText (Reddit links), evaluated zero-shot → first OOD test.
* Perplexity still useful because:

  * Smooth measure for **scaling laws**
  * **Universal** (no task dependence)
  * Enables **conditional perplexity** for downstream tasks
* Still, caution: requires **probability access**, not just output strings.

### “Perplexity Maximalist” View

* True distribution $t$, model $p$ → best achievable = entropy $H(t)$.
* If $p = t$, model solves all tasks → approaching AGI.
* But optimizing perplexity may push on irrelevant regions of $t$.

### Related Benchmarks

* **LAMBADA** – Cloze-style task (predict final word).
* **HellaSwag** – Commonsense sentence completion.

---

## 4. Knowledge Benchmarks

### MMLU

* 57 subjects (math, history, law, etc.)
* Few-shot multiple-choice.
* Measures **knowledge**, not reasoning.
* [HELM visualization](https://crfm.stanford.edu/helm/mmlu/latest/)

### MMLU-Pro

* Removes noisy questions, expands to 10 choices.
* Chain-of-thought allowed.
* Accuracy drops 16–33% → less saturated.

### GPQA

* Graduate-level questions written by PhDs.
* Human PhDs: 65%, GPT-4: 39%.
* [Leaderboard](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/gpqa)

### Humanity’s Last Exam (HLE)

* 2500 multimodal, multi-subject Qs.
* Mix of MCQ + short-answer.
* Co-authorship & $500K reward to question authors.
* [AGI.safe.ai leaderboard](https://agi.safe.ai/)

---

## 5. Instruction-Following Benchmarks

### Chatbot Arena

* Pairwise human comparisons → **ELO score**.
* Open-ended, live, adaptive to new models.
* [Leaderboard](https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard)

### IFEval

* Synthetic instructions with **verifiable constraints**.
* Easy to automate, less natural.
* [Leaderboard](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/ifeval)

### AlpacaEval

* 805 instructions, GPT-4 used as a judge.
* Metric: win-rate vs GPT-4 preview.

### WildBench

* Real human-chatbot conversations.
* GPT-4 Turbo as judge (CoT-based).
* Correlates (0.95) with Chatbot Arena.

---

## 6. Agent Benchmarks

Evaluate *system-level reasoning and tool use.*

| Benchmark    | Description                                               |
| ------------ | --------------------------------------------------------- |
| **SWEBench** | 2294 real GitHub issues → submit PRs; metric = unit tests |
| **CyBench**  | 40 cybersecurity CTF tasks; difficulty via solve time     |
| **MLEBench** | 75 Kaggle-like ML tasks; data + model training required   |

---

## 7. Pure Reasoning Benchmarks

Isolate reasoning from knowledge.

### ARC-AGI (Francois Chollet)

* Abstract reasoning from colored grid transformations.
* ARC-AGI-1: baseline
* ARC-AGI-2: much harder, unsolved
* [ARC Prize](https://arcprize.org/arc-agi)

Goal: measure “systematic generalization” not factual recall.

---

## 8. Safety Benchmarks

What does "safe" mean?

| Benchmark                  | Focus                                                |
| -------------------------- | ---------------------------------------------------- |
| **HarmBench**              | 510 harmful behaviors violating laws or norms        |
| **AIR-Bench**              | 314 risk categories, 5694 prompts, policy-aligned    |
| **Jailbreaking**           | Prompts optimized (GCG method) to bypass refusals    |
| **Pre-deployment Testing** | U.S./U.K. Safety Institutes evaluate frontier models |

### Key Insights

* Safety is contextual: politics, law, norms.
* Capabilities ≠ Propensity:

  * Closed API → must manage *propensity*
  * Open weights → must manage *capability*
* “Dual-use” examples: CyBench used both for safety and capability testing.

---

## 9. Realism

Benchmarks ≠ Real Use.

### Real-World Data

* OpenAI: 100B+ tokens/day
* Cursor: 1B+ lines generated
* Real prompts often messy, repetitive, not academic.

### Prompt Types

1. **Quizzing:** user already knows answer (tests system).
2. **Asking:** user doesn’t know (real usage).
   Second is more realistic and valuable.

### Examples

* **Clio (Anthropic):** analyze real user data.
* **MedHELM:** 121 clinician-sourced tasks → realistic, but privacy trade-offs.

---

## 10. Validity

How do we know our evaluations are *valid*?

### Train–Test Overlap

* Impossible to guarantee when training on Internet-scale data.
* Approaches:

  * Infer overlap statistically ([Exchangeability method](https://arxiv.org/pdf/2310.17623))
  * Reporting standards ([Confidence Intervals paper](https://arxiv.org/abs/2410.08385))

### Dataset Quality

* Verified benchmarks:

  * **SWE-Bench Verified** ([OpenAI blog](https://openai.com/index/introducing-swe-bench-verified/))
  * **Platinum Benchmarks** ([arXiv:2502.03461](https://arxiv.org/abs/2502.03461))

---

## 11. What Are We Evaluating?

| Era             | Evaluation Focus                                  |
| --------------- | ------------------------------------------------- |
| Pre-foundation  | **Methods** (fixed train/test splits)             |
| Post-foundation | **Models/Systems** (no fixed data, anything goes) |

* **DataComp-LM**: fixed raw dataset → optimize performance via better pipelines.
* **NanoGPT speedrun**: fixed compute → measure time to reach target loss.
* Need to clearly define the *rules of the game.*

---

## 12. Takeaways

* There is **no single true evaluation** — depends on purpose.
* Always **inspect examples**, not just metrics.
* Consider **capabilities**, **safety**, **costs**, **realism**.
* Clearly define **rules**: what counts as success?
