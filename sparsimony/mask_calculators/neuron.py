import math
import torch

from sparsimony.mask_calculators.base import (
    RandomGrower,
    RandomPruner,
    MagnitudePruner,
    GradientGrower,
    StructuredPruner,
    StructuredGrower,
    ThresholdPruner,
)
from sparsimony.mask_calculators.unstructured import (
    UnstructuredGradientGrower,
    UnstructuredMagnitudePruner,
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


class NeuronSRigLPruner(StructuredPruner, ThresholdPruner):
    _TILE_VIEW = "neuron"

    @classmethod
    def calculate_mask(
        cls,
        sparsity: float,
        mask: torch.Tensor,
        weights: torch.Tensor,
        grads: torch.Tensor,
        score_override: torch.Tensor | None = None,
        *args,
        **kwargs
    ):
        n_ones = math.floor(mask.numel() * (1 - sparsity))
        candidate_tiles = cls._get_candidate_tiles(mask, score_override)
        mask_slice = mask[candidate_tiles]
        mask_slice = mask_slice.view(-1, mask_slice.shape[1:])
        magnitude_scores = UnstructuredMagnitudePruner.get_scores(
            mask_slice, None, weights=weights
        )
        magnitude_scores = torch.where(
            magnitude_scores == float("inf"),
            torch.full_like(magnitude_scores, -float("inf")),
            magnitude_scores,
        )
        grad_scores = UnstructuredGradientGrower.get_scores(
            mask_slice, None, grads=grads
        )
        grad_scores = torch.where(
            grad_scores == -float("inf"),
            torch.full_like(grad_scores, float("inf")),
            grad_scores,
        )
        # TODO: Figure out how to get union of these sets
        score = torch.zeros_like(mask_slice)
        _, mag_topk = torch.topk(magnitude_scores, k=n_ones, largest=True)
        _, grad_idx_to_grow = torch.topk(grad_scores, k=n_ones, largest=True)
        score = torch.scatter(
            score, dim=-1, index=mag_topk, src=torch.ones_like()
        )
        score = torch.scatter(
            score, dim=-1, index=grad_idx_to_grow, src=torch.ones_like()
        )

        return super().calculate_mask(
            sparsity, mask, score_override, *args, **kwargs
        )
