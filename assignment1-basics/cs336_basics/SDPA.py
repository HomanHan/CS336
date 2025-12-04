import torch
import torch.nn as nn
from torch import Tensor
import einops
from jaxtyping import Bool, Float


# 注意在第 i 个维度上进行计算（max, mean...），是对平行于该维度的每一个切片分别进行计算
# 即控制住其他维度不变。该维度大小为多少，每个切片中就有多少个元素参与计算。最终所有元素都会参与到计算
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    exp_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


def scaled_dot_product_attention(
    query: Float[Tensor, " ... seq_len_q d_k"],
    key: Float[Tensor, " ... seq_len_k d_k"],
    value: Float[Tensor, " ... seq_len_k d_v"],
    mask: Bool[Tensor, " ... seq_len_q seq_len_k"] | None = None,
) -> Float[Tensor, " ... seq_len_q d_v"]:
    """
    Compute the scaled dot-product attention.
    Args:
        query (torch.Tensor): shape (batch, seq_len_q, d_k)
        key (torch.Tensor): shape (batch, seq_len_k, d_k)
        value (torch.Tensor): shape (batch, seq_len_k, d_v)
        mask (torch.Tensor | None): shape (batch, seq_len_q, seq_len_k), Boolen.
    Returns:
        torch.Tensor: shape (batch, seq_len_q, d_v)
    """
    d_k = query.size(-1)
    scores = einops.einsum(
        query, key, "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k"
    ) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32, device=query.device))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn_weights = softmax(scores, dim=-1)
    output = einops.einsum(
        attn_weights,
        value,
        "... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v",
    )
    return output
