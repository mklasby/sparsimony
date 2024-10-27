from typing import Tuple, Any

import torch
import torch.nn as nn
from torch import autograd

from sparsimony.mask_calculators import MagnitudeScorer, NMPruner


class STE(autograd.Function):
    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any], output: Any) -> Any:
        pass

    @staticmethod
    def forward(weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return mask * weights

    @staticmethod
    def backward(ctx, grad_outputs: Tuple[torch.Tensor]) -> torch.Tensor:
        return grad_outputs, None


class SRSTE(autograd.Function):
    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any], output: Any) -> Any:
        weights, mask, decay = inputs
        ctx.decay = decay
        ctx.save_for_backward(weights, mask)

    @staticmethod
    def forward(
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
        )


class FakeSparsitySTE(nn.Module):
    def __init__(self, n: int = 2, m: int = 4, *args, **kwargs):
        super().__init__()
        self.sparsity = n / m

    def forward(self, weights):
        pruner = NMPruner(MagnitudeScorer, n=self.n, m=self.m)
        mask = pruner.calculate_mask(
            self.n / self.m,
            torch.ones_like(weights, dtype=torch.bool),
            values=weights,
        )
        self.mask = mask
        return STE.apply(weights, mask)

    def __name__(self):
        return "FakeSparsitySTE"

    @property
    def sparsity(self):
        return self.n / self.m


class FakeSparsitySRSTE(nn.Module):
    def __init__(
        self, n: int = 2, m: int = 4, decay: float = 2e-4, *args, **kwargs
    ):
        super().__init__()
        self.n = n
        self.m = m
        self.decay = decay

    def forward(self, weights):
        pruner = NMPruner(MagnitudeScorer, n=self.n, m=self.m)
        mask = pruner.calculate_mask(
            self.n / self.m,
            torch.ones_like(weights, dtype=torch.bool),
            values=weights,
        )
        self.mask = mask
        return SRSTE.apply(weights, mask, self.decay)

    def __name__(self):
        return "FakeSparsitySRSTE"

    @property
    def sparsity(self):
        return self.n / self.m
