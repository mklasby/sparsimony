import torch

from sparsimony.mask_calculators.base import (
    RandomGrower,
    RandomPruner,
    MagnitudePruner,
    GradientGrower,
)
from sparsimony.utils import view_tensors_as


class UnstructuredRandomPruner(RandomPruner):

    @view_tensors_as((1, -1))
    @classmethod
    def calculate_mask(
        cls,
        sparsity: float,
        mask: torch.Tensor,
        score_override: torch.Tensor | None = None,
        *args,
        **kwargs
    ) -> torch.Tensor:
        return super().calculate_mask(
            sparsity, mask, score_override, *args, **kwargs
        )


class UnstructuredMagnitudePruner(MagnitudePruner):

    @view_tensors_as((1, -1))
    @classmethod
    def calculate_mask(
        cls,
        sparsity: float,
        mask: torch.Tensor,
        score_override: torch.Tensor | None = None,
        *args,
        **kwargs
    ) -> torch.Tensor:
        return super().calculate_mask(
            sparsity, mask, score_override, *args, **kwargs
        )


class UnstructuredRandomGrower(RandomGrower):

    @view_tensors_as((1, -1))
    @classmethod
    def calculate_mask(
        cls,
        sparsity: float,
        mask: torch.Tensor,
        score_override: torch.Tensor | None = None,
        *args,
        **kwargs
    ) -> torch.Tensor:
        return super().calculate_mask(
            sparsity, mask, score_override, *args, **kwargs
        )


class UnstructuredGradientGrower(GradientGrower):

    @view_tensors_as((1, -1))
    @classmethod
    def calculate_mask(
        cls,
        sparsity: float,
        mask: torch.Tensor,
        score_override: torch.Tensor | None = None,
        *args,
        **kwargs
    ) -> torch.Tensor:
        return super().calculate_mask(
            sparsity, mask, score_override, *args, **kwargs
        )
