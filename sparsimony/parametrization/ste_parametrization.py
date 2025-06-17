from typing import Tuple, Any

import torch
from torch import autograd

from sparsimony.parametrization.fake_sparsity import FakeSparsity


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


class FakeSparsitySTE(FakeSparsity):
    def __init__(
        self, mask: torch.Tensor, n: int = 2, m: int = 4, *args, **kwargs
    ):
        super().__init__(mask)
        self.n = n
        self.m = m

    def forward(self, weights):
        return STE.apply(weights, self.mask)

    def __name__(self):
        return "FakeSparsitySTE"

    @property
    def sparsity(self):
        return 1 - (self.n / self.m)


class FakeSparsitySRSTE(FakeSparsity):
    def __init__(
        self,
        mask: torch.Tensor,
        n: int = 2,
        m: int = 4,
        decay: float = 2e-4,
        *args,
        **kwargs,
    ):
        super().__init__(mask)
        self.n = n
        self.m = m
        self.decay = decay

    def forward(self, weights):
        return SRSTE.apply(weights, self.mask, self.decay)

    def __name__(self):
        return "FakeSparsitySRSTE"

    @property
    def sparsity(self):
        return 1 - (self.n / self.m)
