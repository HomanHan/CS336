from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class AdamW(torch.optim.Optimizer):
    """
    Implements AdamW optimization algorithm.
    Reference: "Decoupled Weight Decay Regularization" by Loshchilov and Hutter, 2017.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        """
        Initializes the AdamW optimizer.
        Args:
            params (Iterable[torch.nn.Parameter]): Iterable of parameters to optimize.
            lr (float): Learning rate.
            betas (tuple[float, float]): Coefficients used for computing running averages of gradient and its square.
            eps (float): Term added to the denominator to improve numerical stability.
            weight_decay (float): Weight decay (L2 penalty) coefficient.
        """
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        Performs a single optimization step.
        Args:
            closure (Optional[Callable]): A closure that reevaluates the model and returns the loss.
        Returns:
            Optional[float]: The loss value if a closure is provided, else None.
        """
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("step", 0) + 1

                g = p.grad.data

                if "m" not in state:
                    state["m"] = torch.zeros_like(p.data)
                if "v" not in state:
                    state["v"] = torch.zeros_like(p.data)
                m = state["m"]
                v = state["v"]

                beta1, beta2 = group["betas"]
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * (g * g)

                lr_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.data -= (
                    lr_t * (m / (torch.sqrt(v) + group["eps"]))
                    + lr * weight_decay * p.data
                )

                state["m"] = m
                state["v"] = v
                state["step"] = t
        return loss
