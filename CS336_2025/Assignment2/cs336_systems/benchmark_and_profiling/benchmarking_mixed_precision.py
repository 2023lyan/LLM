import cs336_basics.model as model
import argparse
import timeit
import torch
from contextlib import nullcontext
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
    parser.add_argument("--context_length", type=int, default = 128, help="Context length.")
    parser.add_argument("--rope_theta", type=float, default = 10000.0, help="RoPE theta parameter.")
    parser.add_argument("--num_warmup", type=int, default = 5, help="Number of warmup runs.")
    return parser.parse_args()

def benchmark(model, x, num_warmup, autocast_context):
    # Warmup
    for _ in range(num_warmup):
        with autocast_context:
            logits = model(x)
            loss = logits.sum()
            loss.backward()
        model.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
    # Timing

    start_time = timeit.default_timer()
    with autocast_context:
        logits = model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Ensure all CUDA ops are done
    end_time_1 = timeit.default_timer() # Time after forward pass
    time_forward = (end_time_1 - start_time)
    with autocast_context:
        loss = logits.sum()
        loss.backward()
    model.zero_grad()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time_2 = timeit.default_timer() # Time after backward pass
    time_backward = (end_time_2 - end_time_1)

    return time_forward, time_backward

def experiment(model_size, context_length, vocab_size, rope_theta, batch_size, num_warmup, device, autocast_context):
    try:
        transformer = model.BasicsTransformerLM(
            vocab_size=vocab_size,
            context_length=context_length,
            **MODEL_CONFIGS[model_size],
            rope_theta=rope_theta,
        ).to(device)

        data = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

        time_forward, time_backward = benchmark(
            transformer, data, num_warmup, autocast_context
        )

        del transformer, data
        torch.cuda.empty_cache()

        return time_forward, time_backward

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"CUDA OOM at model={model_size}, context={context_length}")
            torch.cuda.empty_cache()
            return -1.0, -1.0
        else:
            raise e

if __name__ == "__main__":

    args = parse_args()
    
    device = "cuda" if model.torch.cuda.is_available() else "cpu"
    
    results = []
    for model_size in ["small", "medium", "large", "xl", "2.7B"]:
        for precision, autocast_context in [
                ("FP32", nullcontext()),
                ("FP16", torch.autocast(device_type=device, dtype=torch.float16))
            ]:
            time_forward, time_backward = experiment(model_size, args.context_length, args.vocab_size, args.rope_theta, args.batch_size, args.num_warmup, device, autocast_context)
            results.append({
                "model_size": model_size,
                "autocast": precision,
                "time_forward": time_forward * 1000,
                "time_backward": time_backward * 1000
            })

    df = pd.DataFrame(results)

    df_forward = df.pivot(index="model_size", columns="autocast", values="time_forward")
    df_backward = df.pivot(index="model_size", columns="autocast", values="time_backward")
    print("\n=== Forward Pass Time (ms) ===")
    print(df_forward.to_markdown(floatfmt=".4f"))
    print("\n=== Backward Pass Time (ms) ===")
    print(df_backward.to_markdown(floatfmt=".4f"))