import torch

from .base import (
    FineGrainedGrower,
    FineGrainedPruner,
)
from sparsimony.utils import view_tensors_as


class UnstructuredPruner(FineGrainedPruner):
    _TILE_VIEW = (1, -1)

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


class UnstructuredGrower(FineGrainedGrower):
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
