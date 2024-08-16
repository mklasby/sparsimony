from typing import Any
import torch

from sparsimony.mask_calculators.base import (
    RandomGrower,
    RandomPruner,
    MagnitudePruner,
    GradientGrower,
    FineGrainedPruner,
    FineGrainedGrower,
)
from sparsimony.utils import view_tensors_as, view_tensors_as_neurons


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
        **kwargs,
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
        **kwargs,
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
        **kwargs,
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
        **kwargs,
    ) -> torch.Tensor:
        return super().calculate_mask(
            sparsity, mask, score_override, *args, **kwargs
        )


class NMMagnitudePruner(FineGrainedPruner, MagnitudePruner):

    def __init__(
        self,
        n: int,
        m: int,
        pad: bool = False,
        padding_dim: int = 1,
    ):
        self.n = n
        self.m = m
        self.pad = pad
        self.padding_dim = padding_dim
        self._TILE_VIEW = (-1, self.m)

    def calculate_mask(
        self,
        mask: torch.Tensor,
        score_override: torch.Tensor | None = None,
        sparsity: Any | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if sparsity is not None:
            self._logger.warning(
                f"Sparsity value of {sparsity} passed to N:M calculator, will "
                f"be ignored and calculated for {self.n}:{self.m} instead"
            )
        func = super().calculate_mask

        @view_tensors_as(self._TILE_VIEW, self.pad, self.padding_dim)
        def reshaped_calc_mask(mask, score_override, *args, **kwargs):
            sparsity = 1 - (self.n / self.m)

            return func(
                sparsity,
                mask,
                score_override,
                n_ones_per_tile_target=self.n,
                *args,
                **kwargs,
            )

        return reshaped_calc_mask(mask, score_override, *args, **kwargs)


class NMGradientGrower(FineGrainedGrower, GradientGrower):

    def __init__(
        self,
        n: int,
        m: int,
        pad: bool = False,
        padding_dim: int = 1,
    ):
        self.n = n
        self.m = m
        self.pad = pad
        self.padding_dim = padding_dim
        self._TILE_VIEW = (-1, self.m)

    def calculate_mask(
        self,
        mask: torch.Tensor,
        score_override: torch.Tensor | None = None,
        sparsity: Any | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        func = super().calculate_mask
        if sparsity is not None:
            self._logger.warning(
                f"Sparsity value of {sparsity} passed to N:M calculator, will "
                f"be ignored and calculated for {self.n}:{self.m} instead"
            )

        @view_tensors_as(self._TILE_VIEW, self.pad, self.padding_dim)
        def reshaped_calc_mask(mask, score_override, *args, **kwargs):
            sparsity = 1 - (self.n / self.m)

            return func(
                sparsity,
                mask,
                score_override,
                n_ones_per_tile_target=self.n,
                *args,
                **kwargs,
            )

        return reshaped_calc_mask(mask, score_override, *args, **kwargs)
