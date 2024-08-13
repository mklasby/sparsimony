import torch

from sparsimony.mask_calculators.base import (
    RandomGrower,
    RandomPruner,
    MagnitudePruner,
    GradientGrower,
    FineGrainedPruner,
    FineGrainedGrower,
)
from sparsimony.utils import view_tensors_as


class UnstructuredRandomPruner(FineGrainedPruner, RandomPruner):
    _TILE_VIEW = (1, -1)

    @classmethod
    @view_tensors_as(_TILE_VIEW)
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


class UnstructuredMagnitudePruner(FineGrainedPruner, MagnitudePruner):
    _TILE_VIEW = (1, -1)

    @classmethod
    @view_tensors_as(_TILE_VIEW)
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


class UnstructuredRandomGrower(FineGrainedGrower, RandomGrower):
    _TILE_VIEW = (1, -1)

    @classmethod
    @view_tensors_as(_TILE_VIEW)
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


class UnstructuredGradientGrower(FineGrainedGrower, GradientGrower):
    _TILE_VIEW = (1, -1)

    @classmethod
    @view_tensors_as(_TILE_VIEW)
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
