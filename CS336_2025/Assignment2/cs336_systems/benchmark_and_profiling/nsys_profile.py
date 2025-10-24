import cs336_basics.model as model
from cs336_basics.nn_utils import softmax
from cs336_basics.optimizer import AdamW
from jaxtyping import Float, Bool
from torch import Tensor
from einops import einsum
import argparse
import timeit
import torch
import torch.cuda.nvtx as nvtx
import math
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
    parser.add_argument("--vocab_size", type=int, default = 10000, help="Vocabulary size.")
    parser.add_argument("--batch_size", type=int, default = 4, help="Batch size.")
    parser.add_argument("--rope_theta", type=float, default = 10000.0, help="RoPE theta parameter.")
    parser.add_argument("--num_warmup", type=int, default = 5, help="Number of warmup runs.")
    return parser.parse_args()

def benchmark(model, x, optimizer, num_warmup, model_size, context_length):
    # Warmup
    
    for _ in range(num_warmup):
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        model.zero_grad()
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    with nvtx.range(f"forward pass ({model_size}, {context_length})"):
        start_time = timeit.default_timer()
        logits = model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Ensure all CUDA ops are done
        end_time_1 = timeit.default_timer() # Time after forward pass
        times_forward = end_time_1 - start_time
    with nvtx.range(f"backward pass ({model_size}, {context_length})"):
        loss = logits.sum()
        loss.backward()
        model.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time_2 = timeit.default_timer() # Time after backward pass
        times_backward = end_time_2 - end_time_1
    with nvtx.range(f"optimizer step ({model_size}, {context_length})"):
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time_3 = timeit.default_timer() # Time after optimizer step
        times_optimizer = end_time_3 - end_time_2

    return times_forward, times_backward, times_optimizer

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:

    with nvtx.range("computing attention scores"):
        d_k = K.shape[-1]
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))
    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension
    with nvtx.range("final matmul"):
        final_matmul = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    return final_matmul

def experiment(model_size, context_length, vocab_size, rope_theta, batch_size, num_warmup, device):
    """
    Runs one experiment with given model size and context length.
    Returns (time_forward, time_backward, time_optimizer)
    If CUDA OOM occurs, returns (-1, -1, -1)
    """
    try:
        transformer = model.BasicsTransformerLM(
            vocab_size=vocab_size,
            context_length=context_length,
            **MODEL_CONFIGS[model_size],
            rope_theta=rope_theta,
        ).to(device)

        optimizer = AdamW(transformer.parameters())
        model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

        data = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

        time_forward, time_backward, time_optimizer = benchmark(
            transformer, data, optimizer, num_warmup, model_size, context_length
        )

        del transformer, optimizer, data
        torch.cuda.empty_cache()

        return time_forward, time_backward, time_optimizer

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"CUDA OOM at model={model_size}, context={context_length}")
            torch.cuda.empty_cache()
            return -1.0, -1.0, -1.0
        else:
            raise e


if __name__ == "__main__":

    args = parse_args()
    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for model_size in ["small", "medium", "large", "xl", "2.7B"]:
        for context_length in [128, 256, 512, 1024]:
            time_forward, time_backward, time_optimizer = experiment(model_size, context_length, args.vocab_size, args.rope_theta, args.batch_size, args.num_warmup, device)
            results.append({
                "Model Size": model_size,
                "Context Length": context_length,
                "Forward (ms)": time_forward * 1000,
                "Backward (ms)": time_backward * 1000,
                "Optimizer (ms)": time_optimizer * 1000
            })
    df = pd.DataFrame(results)

    df_forward = df.pivot(index="Model Size", columns="Context Length", values="Forward (ms)")
    df_backward = df.pivot(index="Model Size", columns="Context Length", values="Backward (ms)")
    df_optimizer = df.pivot(index="Model Size", columns="Context Length", values="Optimizer (ms)")

    print("\n=== Forward Pass Time (ms) ===")
    print(df_forward.to_markdown(floatfmt=".4f"))

    print("\n=== Backward Pass Time (ms) ===")
    print(df_backward.to_markdown(floatfmt=".4f"))

    print("\n=== Optimizer Step Time (ms) ===")
    print(df_optimizer.to_markdown(floatfmt=".4f"))