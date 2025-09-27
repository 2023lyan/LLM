# Lecture 1: Overview and Tokenization

Tokenization by letter: hard to handle the long sequence.

Tokenization by word: the vocabulary is too large, and the model cannot handle it.

Byte pair encoding (BPE):
- used in GPT-2 and GPT-3
- train the tokenizer on raw text to automatically determine the vocabulary
- process:
  1. Count the frequency of all pairs of characters in the text.
  2. Merge the most frequent pair into a new token.
  3. Repeat until the desired vocabulary size is reached.
