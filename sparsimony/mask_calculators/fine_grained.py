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


class NMCalculatorBase:
    def __init__(
        self,
        n: int,
        m: int,
        pad: bool = False,
        padding_dim: int = 1,
        permute_conv_to_nhwc: bool = True,
    ):
        self.n = n
        self.m = m
        self.pad = pad
        self.padding_dim = padding_dim
        self.permute_conv_to_nhwc = permute_conv_to_nhwc
        self._TILE_VIEW = (-1, self.m)

    def calculate_mask(
        self,
        mask: torch.Tensor,
        score_override: torch.Tensor | None = None,
        sparsity: Any | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # Will take calculate_mask from next class in MRO. Order of inheritance
        # matters for descendants of this class
        func = super().calculate_mask
        if sparsity is not None:
            self._logger.warning(
                f"Sparsity value of {sparsity} passed to N:M calculator, will "
                f"be ignored and calculated for {self.n}:{self.m} instead"
            )

        @view_tensors_as(
            self._TILE_VIEW,
            self.pad,
            self.padding_dim,
            self.permute_conv_to_nhwc,
        )
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


class NMMagnitudePruner(NMCalculatorBase, FineGrainedPruner, MagnitudePruner):

    def __init__(
        self,
        n: int,
        m: int,
        pad: bool = False,
        padding_dim: int = 1,
        permute_conv_to_nhwc: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(n, m, pad, padding_dim, permute_conv_to_nhwc)


class NMGradientGrower(NMCalculatorBase, FineGrainedGrower, GradientGrower):

    def __init__(
        self,
        n: int,
        m: int,
        pad: bool = False,
        padding_dim: int = 1,
        permute_conv_to_nhwc: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(
            n, m, pad, padding_dim, permute_conv_to_nhwc, *args, **kwargs
        )


class NMRandomGrower(NMCalculatorBase, FineGrainedGrower, RandomGrower):

    def __init__(
        self,
        n: int,
        m: int,
        pad: bool = False,
        padding_dim: int = 1,
        permute_conv_to_nhwc: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(
            n, m, pad, padding_dim, permute_conv_to_nhwc, *args, **kwargs
        )
