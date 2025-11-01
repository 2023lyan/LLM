# Lecture 14: Data 2

## 1. Overview

**Last lecture:** dataset overview (sources & curation)  
**This lecture:** filtering & deduplication mechanics

Topics:
- Filtering algorithms: KenLM, fastText, DSIR  
- Filtering applications: language / quality / toxicity  
- Deduplication: Bloom filter, MinHash, LSH

---

## 2. Filtering Algorithms

Goal:  
Given **target data** $T$ and **raw data** $R$, find subset $T' \subset R$ similar to $T$.

Desiderata:
- Generalize beyond examples in $T$  
- Extremely fast (operate at CommonCrawl scale)

---

### (a) KenLM — n-gram Language Model

- Counts n-gram statistics, applies Kneser–Ney smoothing  
- Estimate:  
  $$
  p(w_n|w_{n-1},...,w_1) = \frac{\text{count}(w_1...w_n)}{\text{count}(w_1...w_{n-1})}
  $$
- Sparse counts → use **Kneser–Ney smoothing** to redistribute probability mass.

KenLM:
- Fast C++ implementation for machine translation  
- Used in CCNet, LLaMA filtering  
- Sort paragraphs by **perplexity**, keep top 1/3 (lowest perplexity = more natural text)

**Summary:**  
KenLM = fast, heuristic, unsupervised quality filter.

---

### (b) fastText — Linear Text Classifier

Paper: *Bag of Tricks for Efficient Text Classification (2016)*

Model:
$$
\text{Input: } x = [w_1, ..., w_L], \quad y = \text{softmax}(U \cdot \frac{1}{L}\sum_i W[w_i])
$$
where:
- $W$: word embeddings $(V \times H)$  
- $U$: linear projection $(H \times K)$  
- Output: $K$ classes (e.g., good / bad)

**Hashing Trick:**  
Use $n$-grams, hashed into fixed bins to avoid explosion of vocabulary.  
Practical $K=2$: binary classifier → “good vs bad”.

Used in:
- Language identification  
- Quality filtering (GPT-3, LLaMA)  
- Toxicity detection (Dolma)

---

### (c) DSIR — Data Selection via Importance Resampling

Paper: *Data Selection for Language Models via Importance Resampling (2023)*

Setup:
- Target distribution $p(x)$ from small clean data $D_p$
- Raw distribution $q(x)$ from large raw data $D_q$

Resampling weights:
$$
w(x) = \frac{p(x)}{q(x)}, \quad \text{Resample with prob } \propto w(x)
$$

Approximate with **hashed n-gram statistics** (for scalability).

**Result:** Slightly better than fastText heuristic on GLUE.  
More principled, similar computational cost.

---

### (d) Summary of Filtering Frameworks

| Method | Model Type | Score Function | Criterion |
|:--|:--|:--|:--|
| KenLM | Generative | $p_T(x)$ | keep high-prob examples |
| fastText | Discriminative | $p(T\|x)$ | keep if $p(T\|x)>\tau$ |
| DSIR | Importance Resampling | $p_T(x)/p_R(x)$ | resample $\propto$ ratio |

---

## 3. Filtering Applications

### (a) Language Identification

- Identify English or other target languages.  
- Tradeoff: multilingual data = complex processing; monolingual = less coverage.

**Tool:** fastText language ID (176 languages)  
Trained on Wikipedia, Tatoeba, SETimes.  
Used in Dolma: keep if $p(\text{English}) ≥ 0.5$.

Challenges:
- Short sequences  
- Dialects / code-switching  
- Similar languages (Malay vs Indonesian)

**Example: OpenMathText**
- Extract math-related text from CommonCrawl  
- Rules: latex commands → math indicator  
- Keep if perplexity (KenLM ProofPile) < 15000  
- Classifier threshold: 0.17 (math) / 0.8 (non-math)

→ Produced 14.7B tokens; model trained on 20× less data but outperformed larger ones.

---

### (b) Quality Filtering

Two styles:
- Heuristic (C4, Gopher, RefinedWeb, Dolma)  
- Model-based (GPT-3, LLaMA, DCLM)

#### GPT-3 (2020)
- Positives: {Wikipedia, WebText2, Books1/2}  
- Negatives: CommonCrawl  
- Train linear classifier → keep with probability ∝ score.

#### LLaMA / RedPajama
- Positives: pages *referenced* by Wikipedia  
- Negatives: CommonCrawl  
- Keep positives.

#### phi-1 (Microsoft)
- Focus on **educational code** data  
- Use GPT-4 to score raw code samples  
- Train random forest on embeddings  
- Filter “educationally valuable” code.

Result:
- 1.3B LM trained on filtered data → higher HumanEval (17.68% vs 12.19%).

---

### (c) Toxicity Filtering

**Dataset:** Jigsaw Toxic Comment Classification (Kaggle, 2018)  
Labels: toxic / obscene / insult / identity hate / threat

**Dolma approach:**
- Two fastText models:
  - Hate: positive = {unlabeled, obscene}
  - NSFW: positive = {obscene}
- Threshold to exclude toxic spans.

Motivation:
- Reduce toxicity propagation  
- Avoid memorizing offensive data

---

## 4. Deduplication

**Goal:** Remove identical or near-identical documents.

### Types
- Exact duplicates: mirrors, reposts, forks  
- Near duplicates: minor token or format changes

### Why
- Efficiency: fewer redundant tokens  
- Generalization: less memorization  
- Legal: mitigate copyright/privacy

---

### (a) Design Space

1. **Item granularity:** sentence / paragraph / doc  
2. **Matching rule:** exact / partial overlap / fuzzy  
3. **Action:** remove all or keep one copy

Need **linear-time algorithms** for scale.

---

### (b) Hash Functions

- Map items → integers  
- Tradeoff: speed vs collisions  

| Type | Example | Collision Resistance | Speed |
|:--|:--|:--|:--|
| Cryptographic | SHA-256 | Strong | Slow |
| Non-crypto | MurmurHash, CityHash | Weak | Fast |

---

### (c) Exact Deduplication

Steps:
1. Hash items  
2. Group identical hashes  
3. Keep one item per group

Used in **C4**:
- Item: 3-sentence span  
- Match: exact hash match  
- Action: remove duplicates

Limitation: may break document coherence.

---

### (d) Bloom Filter

Approximate membership structure.

**Properties**
- Compact bit array of size $m$  
- $k$ hash functions  
- False positives possible; false negatives impossible.

Probability analysis:

1. Probability bit = 1 after inserting $n$ items:
   $$
   f = \left(1 - (1 - \frac{1}{m})^{kn}\right)^k
   $$
2. Optimal $k = (\ln 2)\frac{m}{n}$  
3. Resulting $f = (1/2)^k$

**Dolma:**  
Set $f = 10^{-15}$; dedup by paragraph.

Tradeoff:
- Larger $m$ ↓ false positive rate  
- More $k$ ↑ compute cost  

---

### (e) MinHash & Jaccard Similarity

Similarity between sets $A, B$:
$$
J(A,B) = \frac{|A \cap B|}{|A \cup B|}
$$

**MinHash Property:**  
$$
\Pr[h(A) = h(B)] = J(A,B)
$$

Estimate $J(A,B)$ using multiple random hash functions.

---

### (f) Locality-Sensitive Hashing (LSH)

Goal: efficiently find near-duplicate pairs.

- Use $n = b \times r$ hash functions.  
- Group into **$b$ bands**, each with **$r$ hashes**.  
- Items collide if **all hashes in any band match**.

Probability of collision:
$$
P_{\text{collision}} = 1 - (1 - s^r)^b
$$
where $s = J(A,B)$.

- Increasing $r$ → sharper threshold (harder to match)  
- Increasing $b$ → looser threshold (easier to match)

Example (Dedup paper):  
$n=9000$, $b=20$, $r=450$  
Threshold $≈ (1/b)^{1/r}$

---

## 5. Summary

- **KenLM:** fast n-gram LM for quality filtering.  
- **fastText:** classifier for language / toxicity / quality.  
- **DSIR:** principled importance resampling.  
- **Deduplication:** essential for efficiency & legality.  
- **Bloom filter / MinHash / LSH:** scalable approximate methods.  
- Filtering ≈ *data cleaning at scale* — now you have the mechanics, the rest is intuition.

---
