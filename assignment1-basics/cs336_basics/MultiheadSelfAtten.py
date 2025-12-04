import torch
import torch.nn as nn
import einops
from cs336_basics import SDPA, Linear, RoPE


class causal_multihead_self_attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        with_rope: bool = False,
        max_seq_len: int = 512,
        theta: float = 10000.0,
        token_positions: torch.Tensor | None = None,
    ):
        """
        Args:
            d_model (int): dimension of the model
            num_heads (int): number of attention heads
        """
        super(causal_multihead_self_attention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k  # usually d_v = d_k = d_model / num_heads

        self.W_qkv = Linear.Linear(d_model, 3 * d_model)
        self.W_o = Linear.Linear(d_model, d_model)

        self.rope = RoPE.RoPE(theta, self.d_k, max_seq_len) if with_rope else None
        self.token_positions = token_positions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor of shape (batch, seq_len, d_model)
        Returns:
            torch.Tensor: output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()

        # compute Q, K, V in one matrix multiplication
        qkv = self.W_qkv(x)  # (batch, seq_len, 3 * d_model)

        # split heads and qkv
        qkv = einops.rearrange(
            qkv,
            "batch seq_len (three heads d_model) -> three batch heads seq_len d_model",
            three=3,
            heads=self.num_heads,
            d_model=self.d_k,
        )  # (3, batch, num_heads, seq_len, d_k)
        query, key, value = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # each is (batch, num_heads, seq_len, d_k)

        # apply RoPE if needed
        if self.rope is not None:
            if self.token_positions is None:
                token_positions = (
                    torch.arange(seq_len, device=x.device)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )  # (batch, seq_len)
            else:
                token_positions = self.token_positions.to(x.device)
            query = self.rope(
                query, token_positions
            )  # (batch, num_heads, seq_len, d_k)
            key = self.rope(key, token_positions)  # (batch, num_heads, seq_len, d_k)

        # Create causal mask
        mask = torch.triu(
            torch.ones((seq_len, seq_len), device=x.device), diagonal=1
        ).bool()  # (seq_len, seq_len)
        mask = ~mask  # Invert mask: True where allowed, False where masked
        mask = mask.unsqueeze(0).unsqueeze(
            0
        )  # (1, 1, seq_len, seq_len) for broadcasting

        out = SDPA.scaled_dot_product_attention(
            query, key, value, mask
        )  # (batch, num_heads, seq_len, d_v)

        # concat heads
        out = einops.rearrange(
            out,
            "batch heads seq_len d_v -> batch seq_len (heads d_v)",
            heads=self.num_heads,
            d_v=self.d_v,
        )  # (batch, seq_len, d_model)

        # final linear layer
        out = self.W_o(out)  # (batch, seq_len, d_model)
        return out
