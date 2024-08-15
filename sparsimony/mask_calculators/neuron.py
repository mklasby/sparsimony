import math
import torch

from sparsimony.mask_calculators.base import (
    RandomGrower,
    RandomPruner,
    MagnitudePruner,
    GradientGrower,
    StructuredPruner,
    StructuredGrower,
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


class NeuronSRigLPruner(StructuredPruner):
    # TODO: Potentially extract a class / utility func to count salient elements
    # per tile by passing two or more scorers.
    _TILE_VIEW = "neuron"

    @classmethod
    @view_tensors_as_neurons
    def calculate_mask(
        cls,
        sparsity: float,
        mask: torch.Tensor,
        weights: torch.Tensor,
        grads: torch.Tensor,
        gamma_sal: float = 0.3,
        score_override: torch.Tensor | None = None,
        *args,
        **kwargs
    ):
        n_ones = math.floor(mask.numel() * (1 - sparsity))
        candidate_tiles = cls._get_candidate_tiles(mask, score_override)
        mask_slice = mask[candidate_tiles]
        magnitude_scores = UnstructuredMagnitudePruner.get_scores(
            mask_slice.view(-1), None, weights=weights.view(-1)
        )
        magnitude_scores = magnitude_scores.view(mask_slice.shape)
        # Since we need to count topk elements, we flip the negative
        # cls._SCORE_FILL_VALUE for pruner from +inf to -inf
        # TODO: Pass as param to .get_scores?
        magnitude_scores = torch.where(
            magnitude_scores == UnstructuredMagnitudePruner._SCORE_FILL_VALUE,
            torch.full_like(magnitude_scores, -float("inf")),
            magnitude_scores,
        )
        # Now largest magnitude weights that are active will be in
        # topk(largest=True)
        grad_scores = UnstructuredGradientGrower.get_scores(
            mask_slice.view(-1), None, grads=grads.view(-1)
        )
        grad_scores = grad_scores.view(mask_slice.shape)
        score = torch.zeros_like(mask_slice)
        _, mag_topk = torch.topk(
            magnitude_scores.view(-1), k=n_ones, largest=True
        )
        _, grad_idx_to_grow = torch.topk(
            grad_scores.view(-1), k=n_ones, largest=True
        )
        score = torch.scatter(
            score.view(-1),
            dim=-1,
            index=mag_topk,
            src=torch.ones_like(score.view(-1)),
        ).view(mask_slice.shape)
        score = torch.scatter(
            score.view(-1),
            dim=-1,
            index=grad_idx_to_grow,
            src=torch.ones_like(score.view(-1)),
        ).view(mask_slice.shape)
        # Get count of salient elements per tile
        saliency_count = torch.count_nonzero(score, dim=-1)
        neuron_score = torch.zeros_like(mask_slice)  # per neuron
        # We divide n_ones by FFI target given mask shape
        # TODO: Try passing score override for neurons already ablated
        ffi_ones_target = math.floor(n_ones / mask_slice.shape[0])
        # TODO: Vectorize, potentially with vmap
        for n_idx, salient_els in enumerate(saliency_count):
            neuron_score[n_idx] = salient_els / ffi_ones_target
        neuron_idx_to_ablate = torch.argwhere(neuron_score < gamma_sal)[
            :, 0
        ].unique()
        mask_slice[neuron_idx_to_ablate] = 0
        mask[candidate_tiles] = mask_slice
        return mask
