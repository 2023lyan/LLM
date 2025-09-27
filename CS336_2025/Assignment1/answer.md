# Problem: Understanding Unicode
(a) chr(0) will return '\x00'
(b) "'\\x00'" for representation, nothing for printing
(c) 'this is a test\x00string', for representation, 'this is a teststring' for printing. So chr(0) will disappear in the printed output.

# Problem: Unicode Encodings
(a) UTF-8 is more commonly used than UTF-16 and UTF-32, because UTF-16 and UTF-32 will generate many 0s in the output, which is useless and cause larger file sizes.
(b) This function fails for multibyte UTF-8 characters, because it tries to decode each byte independently instead of decoding the full byte sequence as a unit. For example, if the input is 你好, which is Chinese characters, the function can't work well.
(c) b = bytes([0xC2, 0x41])

# BPE Tokenizer Training – Common Pitfalls

Debugging Logs:

When optimizing the BPE merge step using caching (i.e., updating pair counts incrementally), watch out for two common edge cases:

1. **Overlapping merges**  
   For tokens like ("a", "a", "a"), the merge pair "aa" occurs at indices [0, 1].  
   After merging at index 0 → token becomes ("aa", "a"), so index 1 is no longer valid.  
   Solution: Preprocess the merge indices to remove overlapping or invalid positions.

2. **Incorrect pair context after merging**  
   Consider the word "banana" → tokenized as (b, a, n, a, n, a).  
   If "na" is the most frequent pair, merging the first "na" changes the token to (b, a, "na", n, a).  
   Next, when merging again, we **may incorrectly record ("a", "na")** instead of the correct ("na", "na").  
   Solution: Track and fix pair updates carefully after each merge by referencing the actual token structure, not just raw indices.

# Training Results
In the tinystories, the longest token is b' accomplishment'. 

In owt, it's b'\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82', which is special characters 'ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ'

The most time-consuming step in the tokenizer experiments is the pre-tokenization step.

# Tokenizer Experiments
Tinystories Compression Ratio: 4.116105588663757
OWT Compression Ratio: 4.374199757483666
Tokenize OWT with Tinystories tokenizer, Compression Ratio: 3.1777645825695338
Tinystories Tokenizer Throughput: 558237.6306076008
OWT Tokenizer Throughput: 450743.45190814993


# Transformer LM Resource Accounting

GPT-2 XL:
vocab_size : 50,257
context_length : 1,024
num_layers : 48
d_model : 1,600
num_heads : 25
d_ff : 6,400

## The parameters of the model
Embedding parameters: vocab_size * d_model = 50,257 * 1,600 = 80,411,200

Transformer parameters:
- Each layer has:
  - 2 RMSNorm layers: 2 * d_model = 2 * 1,600 = 3,200
  - 1 MultiheadAttention layer: 4 * d_model * d_model = 4 * 1,600 * 1,600 = 10,240,000 (Wq, Wk, Wv, Wo)
  - 1 FeedForward layer (SwiGLU): 3 * d_model * d_ff = 3 * 1,600 * 6,400 = 30,720,000 (W1, W2, W3)
- Total parameters per layer: 3,200 + 10,240,000 + 30,720,000 = 40,963,200
- Total parameters for all layers: 48 * 40,963,200 = 1,966,233,600

Norm: d_model = 1,600
Output linear layer: vocab_size * d_model = 50,257 * 1,600 = 80,411,200
Total parameters in the model:
Total parameters = Embedding + Transformer + Norm + Output = 80,411,200 + 1,966,233,600 + 1,600 + 80,411,200 = 2,127,057,600

If all the parameters are stored in single-precision floating point (float32), the memory required is:
Memory = Total parameters * 4 bytes = 2,127,057,600 * 4 = 8,508,230,400 bytes = 7.90 GB

## The Matrix Multiplication of the Model
x = (batch_size, context_length, d_model) = (1, 1,024, 1,600)

Transformer:
- Each layer has:
  - MultiheadAttention: 
    - 4 matrix multiplications (Wq, Wk, Wv, Wo) FLOPs = 8ld^2
    - Dot product attention: QK and (QK)V, FLOPs = 4dl^2
  - FeedForward (SwiGLU): 3 matrix multiplications FLOPs = 6dl(d_ff)

output linear layer: 1 matrix multiplication FLOPs = vdl

Total FLOPs in the model:
Total FLOPs = n * (8ld^2 + 4dl^2 + 6dl(d_ff)) + vdl
where 8ld^2 = 20,971,520,000, 4dl^2 = 6,710,886,400, 6dl(d_ff) = 62,914,560,000, vdl = 82,341,068,800
FLOPs = 4,430,995,456,000

## The most FLOPs consuming step in the model
The most FLOPs consuming step in the model is the FeedForward layer (SwiGLU), accounting for 62,914,560,000 * 48 / 4,430,995,456,000 = 68.15% of the total FLOPs.

## Results for different parameter settings
Suppose d_ff = 8/3 * d_model, then the parameters and FLOPs for different GPT-2 models are:
- GPT-2 small:
12 layers, 768 d_model, 12 heads
MultiheadAttention: 8ld^2 + 4dl^2 = 8,053,063,680
FeedForward: 16ld^2 = 9,663,676,416

- GPT-2 medium:
24 layers, 1,024 d_model, 16 heads
MultiheadAttention: 8ld^2 + 4dl^2 = 12,884,901,888
FeedForward: 16ld^2 = 16 * 1,024^2 * 1,024 = 17,179,869,184

- GPT-2 large:
36 layers, 1,280 d_model, 20 heads
MultiheadAttention: 8ld^2 + 4dl^2 = 18,790,481,920
FeedForward: 16ld^2 = 16 * 1,280^2 * 1,024 = 26,843,545,600

Conclusion:
1. MultiheadAttention FLOPs is more sensitive to the l, which is the context length, than the FeedForward FLOPs.
2. In general, the FeedForward FLOPs is larger than the MultiheadAttention FLOPs, especially when d_model is large.

## Relation between FLOPs and context length
MultiheadAttention: 2,053,531,238,400
FeedForward: 671,088,640,000
The feedforward layer is much quicker than the attention layer if the context length is large.

# Results for lr tuning:
Final loss for lr=1e1: 0.011756808497011662
Final loss for lr=1e2: 0.0
Final loss for lr=1e3: inf

With the lr increasing, the loss will decrease at first, but then it will be hard to converge if the lr is too large.
