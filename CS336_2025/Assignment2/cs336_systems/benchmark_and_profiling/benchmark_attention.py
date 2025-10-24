from cs336_basics.model import scaled_dot_product_attention
import timeit
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

D_MODEL = [16, 32, 64, 128]
BATCH_SIZE = 8
SEQUENCE_LENGTH = [256, 1024, 4096, 8192, 16384]
NUM_WARMUP = 5
NUM_ITER = 100

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, q, k, v):
        return scaled_dot_product_attention(q, k, v)


def benchmark_attention(q, k, v, need_compile, device):
    try:
        model = Attention()
        model.to(device)
        if need_compile:
            model = torch.compile(model)

        forward_time = []
        backward_time = []
        forward_memory = []
        backward_memory = []
        
        # Warm-up
        for _ in range(NUM_WARMUP):
            out = model(q, k, v)
            loss = out.sum()
            loss.backward()
            model.zero_grad()

        # Benchmark
        for _ in range(NUM_ITER):
            torch.cuda.reset_peak_memory_stats(device)
            model.zero_grad()
            start = timeit.default_timer()
            out = model(q, k, v)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_time.append(timeit.default_timer() - start)
            forward_memory.append(torch.cuda.max_memory_allocated(device) / (1024 * 1024))  # in MB
            
            torch.cuda.reset_peak_memory_stats(device)
            start = timeit.default_timer()
            loss = out.sum()
            loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            backward_time.append(timeit.default_timer() - start)
            backward_memory.append(torch.cuda.max_memory_allocated(device) / (1024 * 1024))  # in MB
        return np.mean(forward_time), np.mean(backward_time), np.mean(forward_memory), np.mean(backward_memory)
    except Exception as e:
        if "CUDA out of memory" in str(e):
            return -1, -1, -1, -1
        else:
            raise e

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    for d_model in D_MODEL:
        for seq_len in SEQUENCE_LENGTH:
            for need_compile in [False, True]:
                q = torch.randn(BATCH_SIZE, seq_len, d_model, requires_grad=True).to(device)
                k = torch.randn(BATCH_SIZE, seq_len, d_model, requires_grad=True).to(device)
                v = torch.randn(BATCH_SIZE, seq_len, d_model, requires_grad=True).to(device)
                forward_time, backward_time, forward_memory, backward_memory = benchmark_attention(q, k, v, need_compile, device)
                
                results.append({
                    "d_model": d_model,
                    "seq_len": seq_len,
                    "forward_time": forward_time * 1000,  # convert to ms
                    "backward_time": backward_time * 1000,  # convert to ms
                    "forward_memory": forward_memory,
                    "backward_memory": backward_memory,
                    "need_compile": need_compile
                })
    df = pd.DataFrame(results)
    df_no_compile = df[df["need_compile"] == False]
    df_compile = df[df["need_compile"] == True]

    df1 = df_no_compile.pivot(index='seq_len', columns='d_model', values='forward_time')
    df2 = df_no_compile.pivot(index='seq_len', columns='d_model', values='backward_time')
    df3 = df_no_compile.pivot(index='seq_len', columns='d_model', values='forward_memory')
    df4 = df_no_compile.pivot(index='seq_len', columns='d_model', values='backward_memory')

    df5 = df_compile.pivot(index='seq_len', columns='d_model', values='forward_time')
    df6 = df_compile.pivot(index='seq_len', columns='d_model', values='backward_time')
    df7 = df_compile.pivot(index='seq_len', columns='d_model', values='forward_memory')
    df8 = df_compile.pivot(index='seq_len', columns='d_model', values='backward_memory')
    
    print("Without JIT Compilation")
    print("\nForward Time (ms):")
    print(df1.to_markdown())

    print("\nBackward Time (ms):")
    print(df2.to_markdown())

    print("\nForward Memory (MB):")
    print(df3.to_markdown())

    print("\nBackward Memory (MB):")
    print(df4.to_markdown())
    
    print("\nWith JIT Compilation")
    print("\nForward Time (ms):")
    print(df5.to_markdown())

    print("\nBackward Time (ms):")
    print(df6.to_markdown())

    print("\nForward Memory (MB):")
    print(df7.to_markdown())

    print("\nBackward Memory (MB):")
    print(df8.to_markdown())