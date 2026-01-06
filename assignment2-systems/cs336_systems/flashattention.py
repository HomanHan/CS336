import torch
import triton
import triton.language as tl
import einops


class FlashAttentionFunc_pytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # assert Q.is_cuda and K.is_cuda and V.is_cuda, "Expected CUDA tensors"
        assert Q.shape[-1] == K.shape[-1] == V.shape[-1], "Dimension mismatch"
        assert K.shape[-2] == V.shape[-2], "Dimension mismatch"
        N_q = Q.shape[-2]
        N_k = K.shape[-2]
        D = Q.shape[-1]

        ctx.Q_TILE_SIZE = 16
        ctx.KV_TILE_SIZE = 16
        output = torch.zeros_like(Q)
        L = torch.zeros(Q.shape[:-1], device=Q.device)

        for i in range(0, N_q, ctx.Q_TILE_SIZE):
            Q_i = Q[..., i : min(i + ctx.Q_TILE_SIZE, N_q), :]  # (TILE_SIZE_Q, D)
            o_i = torch.zeros((min(ctx.Q_TILE_SIZE, N_q - i), D), device=Q.device)
            l_i = torch.zeros((min(ctx.Q_TILE_SIZE, N_q - i),), device=Q.device)
            m_i = torch.full(
                (min(ctx.Q_TILE_SIZE, N_q - i),), float("-inf"), device=Q.device
            )
            for j in range(0, N_k, ctx.KV_TILE_SIZE):
                K_j = K[
                    ..., j : min(j + ctx.KV_TILE_SIZE, N_k), :
                ]  # (..., TILE_SIZE_KV, D)
                V_j = V[
                    ..., j : min(j + ctx.KV_TILE_SIZE, N_k), :
                ]  # (..., TILE_SIZE_KV, D)

                scores = einops.einsum(Q_i, K_j, "... q d, ... k d -> ... q k") / (
                    D**0.5
                )  # (..., TILE_SIZE_Q, TILE_SIZE_KV)
                if is_causal:
                    mask = torch.triu(
                        torch.ones_like(scores), diagonal=1 + j - i
                    ).bool()
                    scores = scores.masked_fill(mask, float("-inf"))

                old_m_i = m_i
                m_i = torch.max(m_i, torch.max(scores, dim=-1).values)  # (TILE_SIZE_Q,)
                p_i = torch.exp(
                    scores - m_i.unsqueeze(-1)
                )  # (..., TILE_SIZE_Q, TILE_SIZE_KV)
                l_i = torch.exp(old_m_i - m_i) * l_i + torch.sum(
                    p_i, dim=-1
                )  # (TILE_SIZE_Q,)
                o_i = torch.exp(old_m_i - m_i).unsqueeze(-1) * o_i + einops.einsum(
                    p_i, V_j, "... q k, ... k d -> ... q d"
                )  # (TILE_SIZE_Q, D)

            output[..., i : min(i + ctx.Q_TILE_SIZE, N_q), :] = o_i / l_i.unsqueeze(-1)
            L[..., i : min(i + ctx.Q_TILE_SIZE, N_q)] = m_i + torch.log(l_i)

        ctx.save_for_backward(Q, K, V, output, L)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("FlashAttention backward pass not implemented yet")


@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,  # query
    stride_kb,
    stride_kk,
    stride_kd,  # key
    stride_vb,
    stride_vk,
    stride_vd,  # value
    stride_ob,
    stride_oq,
    stride_od,  # output
    stride_lb,
    stride_lq,  # L
    N_QUERIES,
    N_KEYS,  # total number of queries/keys
    scale,  # 1/sqrt(d)
    D: tl.constexpr,  # dimension
    Q_TILE_SIZE: tl.constexpr,  # query tile size
    K_TILE_SIZE: tl.constexpr,  # key/value tile size
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

    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    Q_i = tl.load(
        Q_block_ptr, boundary_check=(0,), padding_option="zero"
    )  # (Q_TILE_SIZE, D)
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_j = tl.load(
            K_block_ptr, boundary_check=(0,), padding_option="zero"
        )  # (K_TILE_SIZE, D)
        V_j = tl.load(
            V_block_ptr, boundary_check=(0,), padding_option="zero"
        )  # (K_TILE_SIZE, D)
        scores = (
            tl.dot(Q_i.to(tl.float32), K_j.T.to(tl.float32)) * scale
        )  # (Q_TILE_SIZE, K_TILE_SIZE)

        if is_causal:
            # Causal masking: query position >= key position
            # Use static ranges and add offsets
            q_indices = (
                tl.arange(0, Q_TILE_SIZE)[:, None] + query_tile_index * Q_TILE_SIZE
            )
            k_indices = tl.arange(0, K_TILE_SIZE)[None, :] + j * K_TILE_SIZE
            mask = q_indices >= k_indices
            scores = tl.where(mask, scores, float("-inf"))

        old_m_i = m_i
        m_i = tl.maximum(m_i, tl.max(scores, axis=1))  # (Q_TILE_SIZE,)
        P_i = tl.exp(scores - m_i[:, None])  # (Q_TILE_SIZE, K_TILE_SIZE)
        l_i = tl.exp(old_m_i - m_i) * l_i + tl.sum(P_i, axis=1)  # (Q_TILE_SIZE,)
        O_i = tl.exp(old_m_i - m_i)[:, None] * O_i + tl.dot(
            P_i.to(tl.float32), V_j.to(tl.float32)
        )  # (Q_TILE_SIZE, D)

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    O_i = O_i / l_i[:, None]
    tl.store(O_block_ptr, O_i.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    L_i = m_i + tl.log(l_i)
    tl.store(L_block_ptr, L_i.to(L_block_ptr.type.element_ty), boundary_check=(0,))


class FlashAttentionFunc_triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # assert Q.is_cuda and K.is_cuda and V.is_cuda, "Expected CUDA tensors"
        assert Q.shape[-1] == K.shape[-1] == V.shape[-1], "Dimension mismatch"
        assert K.shape[-2] == V.shape[-2], "Dimension mismatch"
        N_q = Q.shape[-2]
        N_k = K.shape[-2]
        D = Q.shape[-1]
        ctx.Q_TILE_SIZE = 16
        ctx.KV_TILE_SIZE = 16
        ctx.is_causal = is_causal
        output = torch.zeros_like(Q)
        L = torch.zeros(Q.shape[:-1], device=Q.device)
        scale = 1.0 / (D**0.5)
        grid = (
            triton.cdiv(N_q, ctx.Q_TILE_SIZE),
            Q.shape[0],
        )  # (num_query_tiles, batch_size)

        flash_fwd_kernel[grid](
            Q,
            K,
            V,
            output,
            L,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            L.stride(0),
            L.stride(1),
            N_q,
            N_k,
            scale,
            D,
            ctx.Q_TILE_SIZE,
            ctx.KV_TILE_SIZE,
            ctx.is_causal,
        )
        ctx.save_for_backward(Q, K, V, output, L)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("FlashAttention backward pass not implemented yet")
