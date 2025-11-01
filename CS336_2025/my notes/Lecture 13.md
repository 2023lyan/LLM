# Lecture 13: Data

---

## 1. Overview

**Goal:** What data should we train on?

- Previous: training *given* data  
- Now: choosing and curating data

Data is the most important and least transparent part of LM training.

---

## 2. Importance of Data

### Reasons for secrecy
- Competitive advantage  
- Copyright & liability  

### Stages of training
1. **Pretraining:** raw large-scale web text  
2. **Mid-training:** curated high-quality data  
3. **Post-training:** instruction-following / RLHF  

**Base model:** after pre + mid  
**Chat model:** after post-training

### Example (OLMo)
1. Pretraining → raw web  
2. Mid-training → Dolmino  
3. Post-training → Tülu  

---

## 3. Pretraining Data

### BERT (2019)
- Wikipedia + BooksCorpus (7k books, 985M words)
- Sequences = full documents  
- BooksCorpus removed (ToS violation)

---

### Wikipedia
- 62M articles (2024), curated encyclopedia  
- Open editing, frequent dumps  
- Vulnerable to poisoning before dumps  

---

### GPT-2 WebText (2019)
- Reddit outbound links (≥3 karma)  
- 8M pages, 40GB text  
- OpenWebText = public replication  

---

### Common Crawl (2008–2025)
- Monthly web crawls (~100 total)  
- Apache Nutch crawler  
- Formats: WARC (HTML), WET (text)  
- Text extraction: trafilatura / resiliparse  
- Quality strongly affects downstream tasks  

---

### CCNet (2019)
- Deduplication + fastText language ID + quality filtering (Wikipedia-like)  
- Outperforms Wikipedia for BERT pretraining  

---

### C4 (2019)
- From Common Crawl (April 2019, 1.4T tokens)  
- Heuristics: punctuation, length ≥5 words, langdetect p(en) > 0.99  
- Remove code, profanity, boilerplate  
- Result: 806GB (~156B tokens)  
- Used by T5, standardized large web corpus  

---

### GPT-3 (2020)
- Common Crawl (processed) + WebText2 + Books1/2 + Wikipedia  
- Train classifier to distinguish high-quality sources  
- Fuzzy deduplication  
- 570GB (~400B tokens)  

---

### The Pile (2021)
- Open-source alternative to GPT-3 data  
- 22 curated domains, 825GB (~275B tokens)  
- Includes: Common Crawl, arXiv, PubMed, StackExchange, Enron, Project Gutenberg, Books3, GitHub code  
- Foundation for GPT-NeoX, GPT-J  

---

### LLaMA (2022)
- CCNet + C4 + GitHub + Wikipedia + StackExchange + Books3 + arXiv + Gutenberg  
- 1.2T tokens total  
- Replications:  
  - RedPajama v1: 1T  
  - SlimPajama: 627B (dedup)  
  - RedPajama v2: 30T  

---

### RefinedWeb & FineWeb (2023–2024)
- HTML→text via trafilatura (WARC-based)  
- Avoid model-based filtering  
- RefinedWeb: 600B tokens  
- FineWeb: 15T tokens (95 CC dumps, stricter filters, PII anonymized)  

---

### Dolma (2024)
- Sources: Reddit (Pushshift), Semantic Scholar, C4, Wikipedia, Gutenberg  
- Language ID, toxicity filtering, Bloom deduplication  
- Result: 3T tokens  

---

### DCLM (DataComp-LM, 2024)
- Benchmark for data filtering algorithms  
- DCLM-pool: 240T tokens (raw)  
- DCLM-baseline: 3.8T tokens (filtered)  
- fastText classifier:  
  - Positives: OpenHermes 2.5, ELI5  
  - Negatives: RefinedWeb  
- Outperforms heuristic filtering  

---

### Nemotron-CC (2024, NVIDIA)
- Goal: retain more tokens with quality  
- jusText extraction (more tokens than trafilatura)  
- Ensemble: DCLM classifier + Nemotron LLM scorer  
- Synthetic data rephrasing  
- 6.3T total (1.1T high-quality)  

---

## 4. Mid-Training and Post-Training Data

### Long Context
- Motivation: QA over long texts  
- Transformer cost $O(L^2)$ → use targeted finetuning  
- LongLoRA: extends LLaMA-2 7B from 4K → 100K tokens  
- Train on PG-19, ProofPile  

---

### Instruction & Task Data

**Super-Natural Instructions (2022)**  
- 1.6K+ tasks, templated prompts  
- Tk-Instruct (T5 fine-tune)

**FLAN (2022)**  
- 1.8K+ tasks (zero/few-shot + CoT)  
- Major gains in instruction following  

---

### Chat / Instruction Fine-tuning

| Model | Dataset | Method |
|:--|:--|:--|
| Alpaca | 52K GPT-3.5 samples | Self-Instruct |
| Vicuna | 70K ShareGPT chats | LLaMA fine-tune |
| Baize | 111K GPT-3.5 self-chats | Quora/StackOverflow seeds |
| WizardLM | Evol-Instruct | Expand breadth & difficulty |
| MAmmoTH2 | 10M WebInstruct | GPT-4 + Mixtral extraction |
| OpenHermes 2.5 | 1M GPT-4 samples | Mistral 7B fine-tune |
| LLaMA 2 Chat | 27.5K vendor annotations | High-quality SFT |
| LLaMA–Nemotron Post-train | Synthetic + reasoning | Multi-model mix |

---

## 5. Legal & Ethical Issues

### Copyright
- Protects *expression*, not *ideas*  
- Duration ~75 years → public domain  
- Registration not needed unless suing  

### Licenses
- Creative Commons (2001): open distribution  
- Examples: Wikipedia, Khan Academy  
- Commercial licensing common: Google–Reddit, OpenAI–StackExchange  

### Fair Use (Section 107)
1. Purpose & character (transformative > commercial)  
2. Nature (factual > creative)  
3. Amount (snippet > whole)  
4. Market effect (non-destructive > substitutive)  

Training ≈ transformative, but copying may still violate ToS.

### Terms of Service
- Can forbid usage even with CC license (e.g., YouTube CC videos)

---

## 6. Takeaways

- Data is the real differentiator among LMs.  
- Pretraining: large-scale, low-quality web text.  
- Mid-training: filtered high-quality sources.  
- Post-training: human or synthetic instruction data.  
- Legal risks: copyright, licensing, ToS.  
- Huge room for innovation in **data quality, coverage, and governance**.

---
