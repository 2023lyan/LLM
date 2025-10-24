import torch
from einops import einsum

class FlashAttentionPyTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        Nq, Nk, d = Q.shape[-2], K.shape[-2], Q.shape[-1]
        scale = 1.0 / (d ** 0.5)
        Bq, Bk = 16, 16  # Block sizes for queries and keys
        Tq, Tk = (Nq + Bq - 1) // Bq, (Nk + Bk - 1) // Bk
        O = torch.zeros_like(Q)
        L = torch.zeros(Q.shape[:-1], device=Q.device)
        
        for i in range(Tq):
            q_start = i * Bq
            q_end = min((i + 1) * Bq, Nq)
            Q_block = Q[..., q_start:q_end, :]  # (Bq, D)
            m = torch.full(Q_block.shape[:-1], -1e6, device=Q.device) # (Bq,)
            l = torch.zeros(Q_block.shape[:-1], device=Q.device) # (Bq,)
            o = torch.zeros_like(Q_block) # (Bq, D)

            for j in range(Tk):
                k_start = j * Bk
                k_end = min((j + 1) * Bk, Nk)
                K_block = K[..., k_start:k_end, :]  # (Bk, D)
                V_block = V[..., k_start:k_end, :]  # (Bk, D)
                scores = einsum(Q_block, K_block, '... Bq D, ... Bk D -> ... Bq Bk') * scale  # (Bq, Bk)
                if is_causal:
                    k_index = torch.arange(k_start, k_end, device=Q.device)
                    q_index = torch.arange(q_start, q_end, device=Q.device)
                    mask = q_index[..., None] >= k_index[None, ...]
                    scores = scores.masked_fill(~mask, -1e6)
                m_prev = m
                m = torch.maximum(m, scores.max(dim = -1).values)
                alpha = torch.exp(m_prev - m)
                P = torch.exp(scores - m[..., None])
                l = alpha * l + P.sum(dim = -1)
                o = alpha[..., None] * o + einsum(P, V_block, '... Bq Bk, ... Bk D -> ... Bq D')
            o = o / l[..., None]
            l = m + torch.log(l)
            O[..., q_start:q_end, :] = o
            L[..., q_start:q_end] = l
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        ctx.Bq = Bq
        ctx.Bk = Bk
        return O
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        Bq = ctx.Bq
        Bk = ctx.Bk
        compiled_backward_flash_attention_pytorch = torch.compile(backward_flash_attention_pytorch)
        dQ, dK, dV = compiled_backward_flash_attention_pytorch(Q, K, V, O, L, dO, Q.shape[-1], Bq, Bk, is_causal)
        return dQ, dK, dV, None

def backward_flash_attention_pytorch(Q, K, V, O, L, dO, D, Bq, Bk, is_causal):
    Nq, Nk, d = Q.shape[-2], K.shape[-2], Q.shape[-1]
    scale = 1.0 / (d ** 0.5)
    D = torch.zeros(O.shape[:-1], device=O.device) # (batch_size, Nq)
    Tq = (Nq + Bq - 1) // Bq
    Tk = (Nk + Bk - 1) // Bk
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)
    dQ = torch.zeros_like(Q)
    for i in range(Tq):
        q_start = i * Bq
        q_end = min((i + 1) * Bq, Nq)
        Oi = O[..., q_start:q_end, :] # (Bq, D)
        dOi = dO[..., q_start:q_end, :] # (Bq, D)
        D[..., q_start:q_end] = torch.sum(dOi * Oi, dim=-1)
    for j in range(Tk):
        k_start = j * Bk
        k_end = min((j + 1) * Bk, Nk)
        Kj = K[..., k_start:k_end, :]
        Vj = V[..., k_start:k_end, :]
        dKj = torch.zeros_like(Kj)
        dVj = torch.zeros_like(Vj)
        Tq = (Nq + Bq - 1) // Bq
        for i in range(Tq):
            q_start = i * Bq
            q_end = min((i + 1) * Bq, Nq)
            Qi = Q[..., q_start:q_end, :]
            Oi = O[..., q_start:q_end, :]
            dOi = dO[..., q_start:q_end, :]
            dQi = dQ[..., q_start:q_end, :]
            Sij = einsum(Qi, Kj, "... Bq D, ... Bk D -> ... Bq Bk") * scale
            if is_causal:
                k_index = torch.arange(k_start, k_end, device=Q.device)
                q_index = torch.arange(q_start, q_end, device=Q.device)
                mask = q_index[..., None] >= k_index[None, ...] # q_index[..., None]: (Bq, 1), k_index[None, ...]: (1, Bk), mask: (Bq, Bk)
                Sij = Sij.masked_fill(~mask, -1e6)
            Pij = torch.exp(Sij - L[..., q_start:q_end, None])
            dVj += einsum(Pij, dOi, "... Bq Bk, ... Bq D -> ... Bk D")
            dPij = einsum(dOi, Vj, "... Bq D, ... Bk D -> ... Bq Bk")
            dSij = Pij * (dPij - D[..., q_start: q_end, None]) * scale
            dQi += einsum(dSij, Kj, "... Bq Bk, ... Bk D -> ... Bq D")
            dQ[..., q_start:q_end, :] = dQi
            dKj += einsum(dSij, Qi, "... Bq Bk, ... Bq D -> ... Bk D")
        dK[..., k_start:k_end, :] = dKj
        dV[..., k_start:k_end, :] = dVj
    return dQ, dK, dV