import torch
import torch.nn as nn
import einops
from cs336_basics.Linear import Linear


def SwishLU(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        """
        Construct a Position-wise Feed-Forward Network.
        Args:
            d_model (int): dimension of the model
            d_ff (int): dimension of the feed-forward network. normally 8/3 d_model, a multiple of 64.
        """
        super(PositionWiseFFN, self).__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the Position-wise Feed-Forward Network to the input.
        Args:
            x (torch.Tensor): input tensor of shape (..., d_model)
        Returns:
            torch.Tensor: output tensor of shape (..., d_model)
        """
        return self.w2(SwishLU(self.w1(x)) * self.w3(x))
