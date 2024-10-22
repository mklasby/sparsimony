from typing import Optional
import torch

from .base import (
    BaseGrower,
    BasePruner,
    FineGrainedPruner,
    FineGrainedGrower,
    ABCMaskCalculator,
)
from .scorers import ABCScorer
from sparsimony.utils import view_tensors_as, view_tensors_as_neurons


class FFIPruner(FineGrainedPruner):
    _TILE_VIEW = "neuron"

    @view_tensors_as_neurons
    def calculate_mask(
        cls,
        sparsity: float,
        mask: torch.Tensor,
        score_override: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return super().calculate_mask(
            sparsity, mask, score_override, *args, **kwargs
        )


class FFIGrower(FineGrainedGrower):
    _TILE_VIEW = "neuron"

    @view_tensors_as_neurons
    def calculate_mask(
        cls,
        sparsity: float,
        mask: torch.Tensor,
        score_override: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return super().calculate_mask(
            sparsity, mask, score_override, *args, **kwargs
        )


class NMCalculatorBase(ABCMaskCalculator):

    def __init__(
        self,
        scorer: ABCScorer,
        n: int,
        m: int,
        pad: bool = False,
        padding_dim: int = 1,
        permute_conv_to_nhwc: bool = True,
    ):
        super().__init__(scorer)
        self.n = n
        self.m = m
        self.pad = pad
        self.padding_dim = padding_dim
        self.permute_conv_to_nhwc = permute_conv_to_nhwc
        self._TILE_VIEW = (-1, self.m)

    def calculate_mask(
        self,
        sparsity: float,
        mask: torch.Tensor,
        score_override: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if sparsity is not None:
            if sparsity != self.n / self.m:
                self._logger.debug(
                    f"Sparsity value of {sparsity} passed to N:M calculator, "
                    f"mask may not conform to {self.n}:{self.m} depending on "
                    "algorithm i.e., if you are NOT gradual pruning."
                )
        wrapped_func = super().calculate_mask

        @view_tensors_as(
            self._TILE_VIEW,
            self.pad,
            self.padding_dim,
            self.permute_conv_to_nhwc,
        )
        def reshaped_calc_mask(mask, score_override, *args, **kwargs):
            return wrapped_func(
                sparsity,
                mask,
                score_override,
                *args,
                **kwargs,
            )

        return reshaped_calc_mask(mask, score_override, *args, **kwargs)


class NMPruner(NMCalculatorBase, FineGrainedPruner, BasePruner):
    pass


class NMGrower(NMCalculatorBase, FineGrainedGrower, BaseGrower):
    pass
