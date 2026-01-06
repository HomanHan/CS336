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
