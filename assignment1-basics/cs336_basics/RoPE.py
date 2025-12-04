import torch
import torch.nn as nn


class RoPE(nn.Module):
    cos_cache: torch.Tensor
    sin_cache: torch.Tensor

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        """
        Construct the RoPE module and create buffers if needed.
        Args:
            theta (float): base frequency
            d_k (int): dimension of the key/query vectors
            max_seq_len (int): maximum sequence length
            device: device to store the buffers
        """
        super(RoPE, self).__init__()
        self.d_k = d_k
        self.theta = theta
        self.device = device if device is not None else torch.device("cpu")

        # Create the frequency buffer
        freq_seq = torch.arange(0, max_seq_len, device=device)  # (max_seq_len,)
        dim_seq = torch.arange(0, d_k // 2, device=device)  # (d_k/2,)
        inv_freq = 1.0 / (theta ** ((dim_seq * 2) / d_k))  # (d_k/2,)
        freqs = torch.einsum(
            "i,j->ij", freq_seq, inv_freq
        )  # outer product (max_seq_len, d_k/2)

        # Create the cos and sin buffers
        # self.cos_cache = torch.cos(freqs)
        # self.sin_cache = torch.sin(freqs)
        self.register_buffer(
            "cos_cache",
            torch.cos(freqs),
            persistent=False,  # persistent decides whether to save the buffer in the state_dict
        )  # (max_seq_len, d_k/2)
        self.register_buffer(
            "sin_cache", torch.sin(freqs), persistent=False
        )  # (max_seq_len, d_k/2)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
        Note that you should tolerate x with an arbitrary number of batch dimensions.
        You should assume that the token positions are a tensor of shape (..., seq_len)
        specifying the token positions of x along the sequence dimension.
        Args:
            x (torch.Tensor): input tensor of shape (..., seq_len, d_k)
            token_positions (torch.Tensor): token positions of shape (..., seq_len)
        """
        cos = self.cos_cache[token_positions].to(x.device)
        sin = self.sin_cache[token_positions].to(x.device)

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos

        out = torch.empty_like(x)
        out[..., ::2] = rot_even
        out[..., 1::2] = rot_odd
        return out
