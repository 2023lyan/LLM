import torch
from torch import Tensor
import torch.nn as nn
from einops import rearrange, einsum
from jaxtyping import Float, Int
from collections.abc import Callable, Iterable
from typing import Optional
import math
import numpy.typing as npt

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        std = (2 / (in_features + out_features)) ** 0.5
        weight_init: Float[Tensor, "d_out d_in"] = torch.randn(out_features, in_features, device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(weight_init, mean=0, std=std, a=-3., b=3.)
        self.weight = nn.Parameter(weight_init)

    def forward(self, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        weight_init: Float[Tensor, " vocab_size d_model"] = torch.randn(num_embeddings, embedding_dim, device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(weight_init, mean=0, std=1, a=-3., b=3.)
        self.weight = nn.Parameter(weight_init)
    
    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        weight_init: Float[Tensor, " d_model"] = torch.ones(d_model, device=device, dtype=dtype)
        self.weight = nn.Parameter(weight_init)
        self.eps = eps
        
    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        rms = torch.sqrt(torch.mean(x ** 2, dim = -1, keepdim = True) + self.eps)
        return x * (self.weight / rms)

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
        return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else d_model * 8 // 3
        self.w1 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)
        self.w3 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.silu = SiLU()
                         
    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        return self.w2(self.silu(self.w1(x)) * self.w3(x))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        freqs: Float[Tensor, " d_freq"] = theta ** ( -torch.arange(0, d_k, 2, device=device).float() / d_k)
        pos: Float[Tensor, " pos"] = torch.arange(max_seq_len, device=device).float()
        rot_theta: Float[Tensor, " pos d_freq"] = einsum(pos, freqs, "pos, d_freq -> pos d_freq")
        self.register_buffer("sin", torch.sin(rot_theta), persistent=False)
        self.register_buffer("cos", torch.cos(rot_theta), persistent=False)
    def forward(self, x: Float[Tensor, " ... sequence_length d_k"], token_positions: Int[Tensor, " ... sequence_length"]) -> Float[Tensor, " ... sequence_length d_k"]:
        x1, x2 = rearrange(x, "... (half_d xy) -> xy ... half_d", xy = 2) # ... sequence_length half_d
        sin_pos = self.sin[token_positions]
        cos_pos = self.cos[token_positions]
        x1_rot = einsum(x1, cos_pos, "... pos d_freq, ... pos d_freq -> ... pos d_freq") - einsum(x2, sin_pos, "... pos d_freq, ... pos d_freq -> ... pos d_freq")
        x2_rot = einsum(x1, sin_pos, "... pos d_freq, ... pos d_freq -> ... pos d_freq") + einsum(x2, cos_pos, "... pos d_freq, ... pos d_freq -> ... pos d_freq")
        x_out = rearrange(torch.stack([x1_rot, x2_rot], dim = 0), "xy ... half_d -> ... (half_d xy)")
        return x_out   
        
class Softmax(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.Log_Softmax = Log_Softmax(dim)
    def forward(self, x: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
        return torch.exp(self.Log_Softmax(x))

class Log_Softmax(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
        max_x = torch.max(x, dim=self.dim, keepdim=True).values
        x = x - max_x
        return x - torch.log(torch.sum(torch.exp(x), dim=self.dim, keepdim=True))
    
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = Softmax(dim=-1) # in the "keys" dimension
    def forward(self, Q: Float[Tensor, " ... queries d_k"],
                K: Float[Tensor, " ... keys d_k"],
                V: Float[Tensor, " ... values d_v"], # size of values must equal size of keys
                mask: Float[Tensor, " ... queries keys"] | None = None) -> Float[Tensor, " ... queries d_v"]:
        pre_softmax = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / torch.sqrt(torch.tensor(Q.shape[-1], dtype=Q.dtype, device=Q.device))
        if mask is not None:
            pre_softmax = pre_softmax.masked_fill(mask == 0, float("-inf"))
        return einsum(self.softmax(pre_softmax), V, "... queries keys, ... keys d_v -> ... queries d_v")

class Multihead_Self_Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None, rope: any = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.h = num_heads
        self.device = device
        self.dtype = dtype
        self.Wq_k_v = Linear(d_model, 3 * d_model, device=device, dtype=dtype)
        self.Wo = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = rope
        self.attention = Attention()


    def forward(self, x: Float[Tensor, " ... sequence_length d_in"], 
                token_positions: Int[Tensor, " ... sequence_length"] | None = None) -> Float[Tensor, " ... sequence_length d_out"]:
        Q, K, V = rearrange(self.Wq_k_v(x), " ... (kind d) -> kind ...  d", kind=3)
        Q = rearrange(Q, "... sequence_length (heads d_k) -> ... heads sequence_length d_k", heads=self.h)
        K = rearrange(K, "... sequence_length (heads d_k) -> ... heads sequence_length d_k", heads=self.h)
        V = rearrange(V, "... sequence_length (heads d_v) -> ... heads sequence_length d_v", heads=self.h)
        mask = 1 - torch.triu(torch.ones(Q.shape[-2], K.shape[-2], device=self.device, dtype=self.dtype), diagonal=1)
        if token_positions is not None:
            token_positions = rearrange(token_positions, "... sequence_length -> ... 1 sequence_length")
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        return self.Wo(rearrange(self.attention(Q, K, V, mask), " ... heads queries d_v -> ... queries (heads d_v)", heads = self.h))
    
class Transformer_Block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, device=None, dtype=None, rope: any = None):
        super().__init__()
        self.attention = Multihead_Self_Attention(d_model, num_heads, device=device, dtype=dtype, rope=rope)
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ff = SwiGLU(d_model= d_model, d_ff = d_ff,device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, " batch sequence_length d_model"], token_positions: Int[Tensor, " ... sequence_length"] | None = None):
        x = x + self.attention(self.norm1(x), 
                               token_positions=token_positions)
        x = x + self.ff(self.norm2(x))
        return x
    
class Transformer_LM(nn.Module):

    def __init__(self, 
                 vocab_size: int,
                 context_length: int,
                 num_layers: int,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float,
                 device=None,
                 dtype=None):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(rope_theta, d_model // num_heads, context_length, device=device)
        self.layers = nn.ModuleList([
            Transformer_Block(d_model=d_model, num_heads=num_heads, d_ff=d_ff, device=device, rope = self.rope, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.context_length = context_length
        self.device = device
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.perplexity = Perplexity(dim=-1)
        self.sequence_loss = Sequence_Loss(dim=-1)
        
    def forward(self, x: Int[Tensor, " batch sequence_length"]) -> Float[Tensor, " batch sequence_length vocab_size"]:
        batch_size, sequence_length  = x.shape
        token_positions = torch.arange(sequence_length, device=x.device, dtype=x.dtype).expand(batch_size, -1)
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)
        x = self.norm(x)
        x = self.lm_head(x)
        return x
    @torch.no_grad()
    def generate(self, 
                 x: Int[Tensor, " sequence_length"],
                 max_length: int,
                 temperature: float = 1.0,
                 top_p: float | None = None,
                 end_token: int | None = None) -> Int[Tensor, " sequence_length"]:
        output = x.unsqueeze(0)  # Add batch dimension
        for _ in range(max_length):
            x = output[:, -self.context_length:]  # Keep only the last context_length tokens
            logits = self.forward(x)
            last_logits = logits[:, -1, :] / temperature
            last_logits = last_logits.squeeze(0)  # Remove batch dimension, now shape is (vocab_size,)
            probabilities = Softmax(dim=-1)(last_logits)
            if top_p is not None:
                sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
                prob = 0.0
                pros_chosen = []
                indices_chosen = []
                while prob < top_p:
                    prob += sorted_probs[0].item()
                    pros_chosen.append(sorted_probs[0].item())
                    indices_chosen.append(sorted_indices[0].item())
                    sorted_probs = sorted_probs[1:]
                    sorted_indices = sorted_indices[1:]
                probabilities = torch.zeros_like(probabilities)
                for idx, prob in zip(indices_chosen, pros_chosen):
                    probabilities[idx] = prob
            next_token = torch.multinomial(probabilities, num_samples=1)
            if end_token is not None and abs(next_token.item() - end_token[0]) < 1e-5:
                break
            output = torch.cat([output, next_token.unsqueeze(0)], dim=1)
        return output.squeeze(0)

    @torch.no_grad()
    def compute_perplexity(self, x: Int[Tensor, " batch sequence_length"], targets: Int[Tensor, " batch sequence_length"]) -> Float[Tensor, ""]:
        logits = self.forward(x)
        return self.perplexity(logits, targets)
    
    @torch.no_grad()
    def compute_validation_loss(self, x: Int[Tensor, " batch sequence_length"], targets: Int[Tensor, " batch sequence_length"]) -> Float[Tensor, ""]:
        logits = self.forward(x)
        return self.sequence_loss(logits, targets)

class Cross_Entropy_Loss(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.log_softmax = Log_Softmax(dim)
    def forward(self, inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
        log_probs = self.log_softmax(inputs)
        loss = - torch.mean(log_probs[torch.arange(targets.shape[0]), targets])
        return loss

class Sequence_Loss(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.cross_entropy_loss = Cross_Entropy_Loss(dim)

    def forward(self, inputs: Float[Tensor, " batch_size sequence_length vocab_size"], targets: Int[Tensor, " batch_size sequence_length"]) -> Float[Tensor, ""]:
        inputs_reshaped = rearrange(inputs, " batch_size sequence_length vocab_size -> (batch_size sequence_length) vocab_size")
        targets_reshaped = rearrange(targets, " batch_size sequence_length -> (batch_size sequence_length)")
        
        loss = self.cross_entropy_loss(inputs_reshaped, targets_reshaped)
        return loss

class Perplexity(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.sequence_loss = Sequence_Loss(dim)

    def forward(self, inputs: Float[Tensor, " batch_size sequence_length vocab_size"], targets: Int[Tensor, " batch_size sequence_length"]) -> Float[Tensor, ""]:

        return torch.exp(self.sequence_loss(inputs, targets))

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if betas[0] < 0 or betas[1] < 0:
            raise ValueError(f"Invalid beta values: {betas}") 
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 1
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                grad = p.grad.data
                t = state["step"]
                exp_avg = betas[0] * exp_avg + (1 - betas[0]) * grad
                exp_avg_sq = betas[1] * exp_avg_sq + (1 - betas[1]) * grad ** 2
                alpha = lr * math.sqrt(1 - betas[1] ** t) / (1 - betas[0] ** t)
                p.data = p.data - alpha * exp_avg / (torch.sqrt(exp_avg_sq) + eps)
                p.data = p.data - lr * weight_decay * p.data
                state["step"] += 1
                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq
        return loss

class Learning_Rate_Schedule():
    def __init__(self, lr_min: int, lr_max: int, warmup_steps: int, decay_steps: int):
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.lr_max * step / self.warmup_steps
        elif step < self.decay_steps:
            return self.lr_min + 1 / 2 * (1 + math.cos(math.pi * (step - self.warmup_steps) / (self.decay_steps - self.warmup_steps))) * (self.lr_max - self.lr_min)
        else:
            return self.lr_min

class Gradient_Clipping():
    def __init__(self, max_norm: float, eps: float = 1e-6):
        self.max_norm = max_norm
        self.eps = eps
    def clip(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        l2_norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in parameters if p.grad is not None))
        if l2_norm > self.max_norm:
            scale = self.max_norm / (l2_norm + self.eps)
            for p in parameters:
                if p.grad is not None:
                    p.grad.data.mul_(scale)
