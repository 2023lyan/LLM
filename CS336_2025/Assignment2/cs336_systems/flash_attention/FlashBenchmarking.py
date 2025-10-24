import torch
import triton.testing as testing
import pandas as pd
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.resolve()))
from FlashAttentionTriton import FlashAttentionTriton


def pytorch_attention(Q, K, V, is_causal=True):
    scale = 1.0 / (Q.shape[-1] ** 0.5)
    attn = torch.matmul(Q, K.transpose(-2, -1)) * scale
    if is_causal:
        mask = torch.triu(torch.ones_like(attn), diagonal=1) * -1e6
        attn = attn + mask
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, V)


def bench_forward(fn, Q, K, V):
    def run():
        out = fn(Q, K, V)
        torch.cuda.synchronize()
        return out
    return testing.do_bench(run) * 1e3 


def bench_backward(fn, Q, K, V):
    def run():
        Q_ = Q.clone().requires_grad_(True)
        K_ = K.clone().requires_grad_(True)
        V_ = V.clone().requires_grad_(True)
        out = fn(Q_, K_, V_)
        grad = torch.randn_like(out)
        out.backward(grad)
        torch.cuda.synchronize()
    return testing.do_bench(run) * 1e3


def run_benchmark():
    torch.cuda.init()
    torch.manual_seed(0)

    batch_size = 1
    is_causal = True

    seq_lens = [128, 256, 512, 1024]
    dims = [16, 32, 64]
    dtypes = [torch.float32, torch.bfloat16]

    results = []

    for N in seq_lens:
        for D in dims:
            for dtype in dtypes:

                Q = torch.randn(batch_size, N, D, device="cuda", dtype=dtype)
                K = torch.randn(batch_size, N, D, device="cuda", dtype=dtype)
                V = torch.randn(batch_size, N, D, device="cuda", dtype=dtype)

                triton_fwd = bench_forward(lambda Q, K, V: FlashAttentionTriton.apply(Q, K, V, is_causal), Q, K, V)
                triton_bwd = bench_backward(lambda Q, K, V: FlashAttentionTriton.apply(Q, K, V, is_causal), Q, K, V)
                results.append({
                    "Seq": N,
                    "Dim": D,
                    "Dtype": str(dtype).split(".")[-1],
                    "Impl": "Triton",
                    "Forward (ms)": triton_fwd,
                    "Backward (ms)": triton_bwd,
                    "Total (ms)": triton_fwd + triton_bwd
                })

                torch_fwd = bench_forward(lambda Q, K, V: pytorch_attention(Q, K, V, is_causal), Q, K, V)
                torch_bwd = bench_backward(lambda Q, K, V: pytorch_attention(Q, K, V, is_causal), Q, K, V)
                results.append({
                    "Seq": N,
                    "Dim": D,
                    "Dtype": str(dtype).split(".")[-1],
                    "Impl": "PyTorch",
                    "Forward (ms)": torch_fwd,
                    "Backward (ms)": torch_bwd,
                    "Total (ms)": torch_fwd + torch_bwd
                })

    df = pd.DataFrame(results)

    df_pivot = df.pivot_table(index=["Seq", "Dim", "Dtype"], columns="Impl", values="Total (ms)")
    if "PyTorch" in df_pivot.columns and "Triton" in df_pivot.columns:
        df["Speedup"] = df.apply(
            lambda r: df_pivot.loc[(r["Seq"], r["Dim"], r["Dtype"]), "PyTorch"] /
                      df_pivot.loc[(r["Seq"], r["Dim"], r["Dtype"]), "Triton"]
            if r["Impl"] == "Triton" else None, axis=1
        )

    print(df.to_markdown(index=False, floatfmt=".3f"))


if __name__ == "__main__":
    run_benchmark()
