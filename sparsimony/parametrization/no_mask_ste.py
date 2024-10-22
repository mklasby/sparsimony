from typing import Tuple
import torch
from torch import autograd


# TODO: test


class STE(autograd.Function):
    @staticmethod
    def setup_context(
        ctx: torch.Any, inputs: Tuple[torch.Any], output: torch.Any
    ) -> torch.Any:
        pass

    @staticmethod
    def forward(weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return mask * weights

    @staticmethod
    def backward(ctx, grad_outputs: Tuple[torch.Tensor]) -> torch.Tensor:
        return grad_outputs, None, None


class STE_no_mask(autograd.Function):
    @staticmethod
    def setup_context(
        ctx: torch.Any, inputs: Tuple[torch.Any], output: torch.Any
    ) -> torch.Any:
        pass

    @staticmethod
    def forward(
        weights: torch.Tensor,
        n: int = 2,
        m: int = 4,
    ) -> torch.Tensor:
        weights_view = weights.view(-1, m)
        _, idx = torch.topk(torch.abs(weights_view), k=n, dim=-1, larget=True)
        mask = torch.scatter(
            torch.zeros_like(weights_view),
            -1,
            idx,
            torch.ones_like(weights_view),
        )
        return mask * weights

    @staticmethod
    def backward(ctx, grad_outputs: Tuple[torch.Tensor]) -> torch.Tensor:
        return grad_outputs, None, None


class SRSTE(autograd.Function):
    @staticmethod
    def setup_context(
        ctx: torch.Any, inputs: Tuple[torch.Any], output: torch.Any
    ) -> torch.Any:
        weights, mask, decay = inputs
        ctx.decay = decay
        ctx.save_for_backwards(weights, mask)

    @staticmethod
    def forward(
        ctx,
        weights: torch.Tensor,
        mask: torch.Tensor,
        decay: float = 2e-4,
    ) -> torch.Tensor:
        return mask * weights

    @staticmethod
    def backward(ctx, grad_outputs: Tuple[torch.Tensor]) -> torch.Tensor:
        weights, mask = ctx.saved_tensors
        return (
            grad_outputs + ctx.decay * ~mask * weights,
            None,
            None,
            None,
        )
