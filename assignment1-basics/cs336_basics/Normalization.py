import torch
import torch.nn as nn
import einops


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct an RMSNorm module.
        Args:
            d_model (int): Dimension of the model.
            eps (float): Epsilon value for numerical stability.
            device (torch.device | None): Device on which to store the params.
            dtype (torch.dtype | None): Data type of the params.
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization to the input.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, d_model).
        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x_normalized = x / rms * self.scale

        return x_normalized.to(in_dtype)
