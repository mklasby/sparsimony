import torch

from sparsimony.mask_calculators.base import (
    RandomGrower,
    RandomPruner,
    MagnitudePruner,
    GradientGrower,
    StructuredPruner,
    StructuredGrower,
)
from sparsimony.utils import view_tensors_as_neurons


class NeuronRandomPruner(StructuredPruner, RandomPruner):
    _TILE_VIEW = "neuron"

    @classmethod
    @view_tensors_as_neurons
    def calculate_mask(
        cls,
        sparsity: float,
        mask: torch.Tensor,
        score_override: torch.Tensor | None = None,
        aggregate_norm_ord: str | int = 2,
        *args,
        **kwargs
    ) -> torch.Tensor:
        return super().calculate_mask(
            sparsity,
            mask,
            score_override,
            aggregate_norm_ord=aggregate_norm_ord,
            *args,
            **kwargs
        )


class NeuronMagnitudePruner(StructuredPruner, MagnitudePruner):
    _TILE_VIEW = "neuron"

    @classmethod
    @view_tensors_as_neurons
    def calculate_mask(
        cls,
        sparsity: float,
        mask: torch.Tensor,
        score_override: torch.Tensor | None = None,
        aggregate_norm_ord: str | int = 2,
        *args,
        **kwargs
    ) -> torch.Tensor:
        return super().calculate_mask(
            sparsity,
            mask,
            score_override,
            aggregate_norm_ord=aggregate_norm_ord,
            *args,
            **kwargs
        )


class NeuronRandomGrower(StructuredGrower, RandomGrower):
    _TILE_VIEW = "neuron"

    @classmethod
    @view_tensors_as_neurons
    def calculate_mask(
        cls,
        sparsity: float,
        mask: torch.Tensor,
        score_override: torch.Tensor | None = None,
        aggregate_norm_ord: str | int = 2,
        *args,
        **kwargs
    ) -> torch.Tensor:
        return super().calculate_mask(
            sparsity,
            mask,
            score_override,
            aggregate_norm_ord=aggregate_norm_ord,
            *args,
            **kwargs
        )


class NeuronGradientGrower(StructuredGrower, GradientGrower):
    _TILE_VIEW = "neuron"

    @classmethod
    @view_tensors_as_neurons
    def calculate_mask(
        cls,
        sparsity: float,
        mask: torch.Tensor,
        score_override: torch.Tensor | None = None,
        aggregate_norm_ord: str | int = 2,
        *args,
        **kwargs
    ) -> torch.Tensor:
        return super().calculate_mask(
            sparsity,
            mask,
            score_override,
            aggregate_norm_ord=aggregate_norm_ord,
            *args,
            **kwargs
        )
