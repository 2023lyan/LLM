import cs336_basics.model as model
from cs336_basics.optimizer import AdamW
import argparse
import torch
from contextlib import nullcontext
import os


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
    parser.add_argument("--rope_theta", type=float, default = 10000.0, help="RoPE theta parameter.")
    parser.add_argument("--num_warmup", type=int, default = 5, help="Number of warmup runs.")
    return parser.parse_args()

def benchmark(model, x, optimizer, num_warmup, context_length, autocast_context, backward):
    # Warmup
    with autocast_context:
        for _ in range(num_warmup):
            logits = model(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if backward:
                loss = logits.sum()
                model.zero_grad()
                loss.backward()
                optimizer.step()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

    torch.cuda.memory._record_memory_history(max_entries=1000000)
    with autocast_context:
        logits = model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if backward:
            loss = logits.sum()
            model.zero_grad()
            loss.backward()
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    precision = 'fp16' if isinstance(autocast_context, torch.autocast) else 'fp32'
    os.makedirs("../results/memory_profiling", exist_ok=True)
    torch.cuda.memory._dump_snapshot(f"../../results/memory_profiling/memory_{context_length}_{precision}_{backward}.pickle")

    torch.cuda.memory._record_memory_history(enabled=None)



def experiment(model_size, context_length, vocab_size, rope_theta, batch_size, num_warmup, device, autocast_context, backward):
    transformer = model.BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        **MODEL_CONFIGS[model_size],
        rope_theta=rope_theta,
    ).to(device)

    optimizer = AdamW(transformer.parameters())

    data = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

    benchmark(
        transformer, data, optimizer, num_warmup, context_length, autocast_context, backward
    )

    del transformer, optimizer, data
    torch.cuda.empty_cache()
    

if __name__ == "__main__":

    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for context_length in [128, 256, 512]:
        for autocast_context in [nullcontext(), torch.autocast(device_type=device, dtype=torch.float16)]:
            for backward in [False, True]:
                experiment(args.model_size, context_length, args.vocab_size, args.rope_theta, args.batch_size, args.num_warmup, device, autocast_context, backward)