import torch

from sparsimony.mask_calculators.base import (
    RandomGrower,
    RandomPruner,
    MagnitudePruner,
    GradientGrower,
    FineGrainedPruner,
    FineGrainedGrower,
)
from sparsimony.utils import view_tensors_as_neurons


class FFIRandomPruner(FineGrainedPruner, RandomPruner):
    _TILE_VIEW = "neuron"

    @classmethod
    @view_tensors_as_neurons
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


class FFIMagnitudePruner(FineGrainedPruner, MagnitudePruner):
    _TILE_VIEW = "neuron"

    @classmethod
    @view_tensors_as_neurons
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


class FFIRandomGrower(FineGrainedGrower, RandomGrower):
    _TILE_VIEW = "neuron"

    @classmethod
    @view_tensors_as_neurons
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


class FFIGradientGrower(FineGrainedGrower, GradientGrower):
    _TILE_VIEW = "neuron"

    @classmethod
    @view_tensors_as_neurons
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
