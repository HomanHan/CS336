import torch
import torch.nn as nn
import einops


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct a linear transformation module.
        Args:
            in_features (int): final dimension of the input
            out_features (int): final dimension of the output
            device (torch.device, optional): Device to store the parameters on
            dtype (torch.dtype, optional): Data type of the parameters
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        sigma = (2 / (in_features + out_features)) ** 0.5
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0, std=sigma, a=-3*sigma, b=3*sigma) # 初始化一个张量为截断正态分布

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.
        """
        return einops.einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
