import torch
import triton
import triton.language as tl

class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        Nq, Nk, D, batch_size = Q.shape[-2], K.shape[-2], Q.shape[-1], Q.shape[-3]
        scale = 1.0 / (D ** 0.5)
        Bq, Bk = 16, 16  # Block sizes for queries and keys
        Tq = (Nq + Bq - 1) // Bq
        O = torch.zeros_like(Q, device=Q.device)
        L = torch.zeros(Q.shape[:-1], device=Q.device)

        flash_fwd_kernel[(Tq, batch_size)](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            Nq, Nk,
            scale,
            D=D,
            Q_TILE_SIZE=Bq,
            K_TILE_SIZE=Bk,
            is_causal=is_causal,
        )
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
        Nq, Nk, d, batch_size = Q.shape[-2], K.shape[-2], Q.shape[-1], Q.shape[-3]
        scale = 1.0 / (d ** 0.5)
        D = torch.zeros(O.shape[:-1], device=O.device) # (batch_size, Nq)
        Tk = (Nk + Bk - 1) // Bk
        Tq = (Nq + Bq - 1) // Bq

        rowsum_of_two_matrix_kernel[(Tq, batch_size)](
            O, dO, D,
            O.stride(0), O.stride(1), O.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            D.stride(0), D.stride(1),
            Nq, d,
            Q_TILE_SIZE=Bq,
        )

        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)
        dQ = torch.zeros_like(Q)
        flash_bwd_kernel_dkv[(Tk, batch_size)](
            Q, K, V, D,
            L, dO,
            dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            D.stride(0), D.stride(1),
            L.stride(0), L.stride(1),
            dO.stride(0), dO.stride(1), dO.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            Nq, Nk,
            scale,
            D=d,
            Q_TILE_SIZE=Bq,
            K_TILE_SIZE=Bk,
            is_causal=is_causal,
            )
        flash_bwd_kernel_dq[(Tq, batch_size)](
            Q, K, V, D,
            L, dO,
            dQ,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            D.stride(0), D.stride(1),
            L.stride(0), L.stride(1),
            dO.stride(0), dO.stride(1), dO.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            Nq, Nk,
            scale,
            D=d,
            Q_TILE_SIZE=Bq,
            K_TILE_SIZE=Bk,
            is_causal=is_causal,
            )
            
        return dQ, dK, dV, None

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
    ):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
        )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
        )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
        )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
        )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
        )

    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32) # (Q_TILE_SIZE, D)
    m = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32) # (Q_TILE_SIZE,)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32) # (Q_TILE_SIZE,)
    o = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32) # (Q_TILE_SIZE, D)
    q_valid = (query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)) < N_QUERIES
    
    T_TILE_SIZE = tl.cdiv(N_KEYS, K_TILE_SIZE)
    for j in range(T_TILE_SIZE):
        Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        k_valid = (j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)) < N_KEYS
        boundary = q_valid[:, None] & k_valid[None, :]
        scores = tl.dot(Qi, tl.trans(Kj)) * scale
        scores = tl.where(boundary, scores, -1e6)
        if is_causal:
            k_index = tl.arange(0, K_TILE_SIZE) + j * K_TILE_SIZE
            q_index = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE
            mask = q_index[:, None] < k_index[None, :]
            scores = tl.where(mask, -1e6, scores)
        m_prev = m
        m = tl.maximum(m, tl.max(scores, axis = -1))
        alpha = tl.exp(m_prev - m)
        P = tl.exp(scores - m[:, None])
        l = alpha * l + tl.sum(P, axis = -1)
        o = tl.dot(P, Vj, acc= alpha[:, None] * o)
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    o = o / l[:, None]
    l = m + tl.log(tl.maximum(l, 1e-6))
    
    o = o.to(O_block_ptr.type.element_ty)
    l = l.to(L_block_ptr.type.element_ty)
    tl.store(O_block_ptr, o, boundary_check=(0, 1))
    tl.store(L_block_ptr, l, boundary_check=(0,))

@triton.jit
def rowsum_of_two_matrix_kernel(
    O_ptr, dO_ptr, D_ptr,
    stride_ob, stride_oq, stride_od,
    stride_dob, stride_doq, stride_dod,
    stride_db, stride_dq,
    N_QUERIES,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    ):
    
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets = (query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
        )
    
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets = (query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
        )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets = (query_tile_index * Q_TILE_SIZE, ),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
        )
    Oi = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero") 
    dOi = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero") 
    Di = tl.sum(dOi * Oi, axis = -1)
    Di = Di.to(D_block_ptr.type.element_ty)
    tl.store(D_block_ptr, Di, boundary_check=(0,))
    
@triton.jit
def flash_bwd_kernel_dkv(
    Q_ptr, K_ptr, V_ptr, D_ptr,
    L_ptr, dO_ptr,
    dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_db, stride_dq,
    stride_lb, stride_lq,
    stride_dob, stride_doq, stride_dod,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
    ):
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    k_start = key_tile_index * K_TILE_SIZE

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
        )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(k_start, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
        )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(k_start, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
        )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
        )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
        )
    
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
        )

    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, D),
        strides=(stride_dkk, stride_dkd),
        offsets=(k_start, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
        )

    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, D),
        strides=(stride_dvk, stride_dvd),
        offsets=(k_start, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
        )

    Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    dKj = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dVj = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    Tq = tl.cdiv(N_QUERIES, Q_TILE_SIZE)
    for i in range(Tq):
        q_start = i * Q_TILE_SIZE
        Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        dOi = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        Li = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
        Di = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
        Sij = tl.dot(Qi, tl.trans(Kj)) * scale
        
        if is_causal:
            k_index = k_start + tl.arange(0, K_TILE_SIZE)
            q_index = q_start + tl.arange(0, Q_TILE_SIZE)
            mask = q_index[:, None] >= k_index[None, :] # q_index[..., None]: (Bq, 1), k_index[None, ...]: (1, Bk), mask: (Bq, Bk)
            Sij = tl.where(mask, Sij, -1e6)
        Pij = tl.exp(Sij - Li[:, None])
        dVj += tl.dot(tl.trans(Pij), dOi)
        dPij = tl.dot(dOi, tl.trans(Vj))
        dSij = Pij * (dPij - Di[:, None]) * scale

        dKj += tl.dot(tl.trans(dSij), Qi)
        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE,))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
    
    dKj = dKj.to(dK_block_ptr.type.element_ty)
    dVj = dVj.to(dV_block_ptr.type.element_ty)
    tl.store(dK_block_ptr, dKj, boundary_check=(0, 1))
    tl.store(dV_block_ptr, dVj, boundary_check=(0, 1))

@triton.jit
def flash_bwd_kernel_dq(
    Q_ptr, K_ptr, V_ptr, D_ptr,
    L_ptr, dO_ptr,
    dQ_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_db, stride_dq,
    stride_lb, stride_lq,
    stride_dob, stride_doq, stride_dod,
    stride_dqb, stride_dqq, stride_dqd,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
    ):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    q_start = query_tile_index * Q_TILE_SIZE

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(q_start, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
        )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
        )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
        )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(q_start,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
        )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(q_start,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
        )
    
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(q_start, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
        )

    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dqb,
        shape=(N_QUERIES, D),
        strides=(stride_dqq, stride_dqd),
        offsets=(q_start, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
        )

    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    dOi = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    dQi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    Li = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
    Di = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
    Tk = tl.cdiv(N_KEYS, K_TILE_SIZE)
    for j in range(Tk):
        Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        k_start = j * K_TILE_SIZE
        Sij = tl.dot(Qi, tl.trans(Kj)) * scale
        
        if is_causal:
            k_index = k_start + tl.arange(0, K_TILE_SIZE)
            q_index = q_start + tl.arange(0, Q_TILE_SIZE)
            mask = q_index[:, None] >= k_index[None, :] # q_index[..., None]: (Bq, 1), k_index[None, ...]: (1, Bk), mask: (Bq, Bk)
            Sij = tl.where(mask, Sij, -1e6)
        Pij = tl.exp(Sij - Li[:, None])
        dPij = tl.dot(dOi, tl.trans(Vj))
        dSij = Pij * (dPij - Di[:, None]) * scale
        dQi += tl.dot(dSij, Kj)
        
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    tl.store(dQ_block_ptr, dQi.to(dQ_block_ptr.type.element_ty), boundary_check=(0, 1))
