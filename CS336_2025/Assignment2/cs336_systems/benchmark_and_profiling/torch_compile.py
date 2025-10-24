import cs336_basics.model as model
from cs336_basics.optimizer import AdamW
import argparse
import timeit
import torch
import numpy as np
import pandas as pd

MODEL_CONFIGS = {
    "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl":     dict(d_model=1600, d_ff=6400,  num_layers=48, num_heads=25),
    "2.7B":   dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmarking script for CS336 systems assignment.")
    parser.add_argument("--model_size", type=str, choices=MODEL_CONFIGS.keys(), default="small", help="Model size to benchmark.")
    parser.add_argument("--vocab_size", type=int, default = 10000, help="Vocabulary size.")
    parser.add_argument("--batch_size", type=int, default = 4, help="Batch size.")
    parser.add_argument("--context_length", type=int, default = 512, help="Context length.")
    parser.add_argument("--rope_theta", type=float, default = 10000.0, help="RoPE theta parameter.")
    parser.add_argument("--num_warmup", type=int, default = 5, help="Number of warmup runs.")
    parser.add_argument("--num_iters", type=int, default = 10, help="Number of iterations to time.")
    return parser.parse_args()

def benchmark(model, x, num_warmup, num_iters, backward):
    # Warmup
    optimizer = AdamW(model.parameters())
    for _ in range(num_warmup):
        logits = model(x)
        if backward:
            loss = logits.sum()
            loss.backward()
            optimizer.step()
            model.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Timing
    times = []
    for _ in range(num_iters):
        start_time = timeit.default_timer()
        logits = model(x)
        if not backward:
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Ensure all CUDA ops are done
            end_time = timeit.default_timer() # Time after forward pass
            times.append(end_time - start_time)
        else:
            loss = logits.sum()
            loss.backward()
            optimizer.step()
            model.zero_grad()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = timeit.default_timer() # Time after backward pass
            times.append(end_time - start_time)

    return np.mean(times)


if __name__ == "__main__":

    args = parse_args()
    
    device = "cuda" if model.torch.cuda.is_available() else "cpu"
    
    transformer = model.BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        **MODEL_CONFIGS[args.model_size],
        rope_theta=args.rope_theta,
    )
    
    data = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    transformer = transformer.to(device)
    compiled_transformer = torch.compile(transformer)
    
    results = []
    for need_compile in [False, True]:
        for backward in [False, True]:
            time = benchmark(compiled_transformer if need_compile else transformer, data, args.num_warmup, args.num_iters, backward)
            results.append({
                "need_compile": need_compile,
                "backward": backward,
                "time": time * 1000 # Convert to ms
            })
    df = pd.DataFrame(results)
    df = df.pivot(index='need_compile', columns='backward', values='time')
    df.columns = ['forward_time_ms', 'backward_time_ms']
    df.index = ['no_compile', 'compile']
    print(df.to_markdown())
