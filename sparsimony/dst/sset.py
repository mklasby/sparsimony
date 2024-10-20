from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.ao.pruning.sparsifier.base_sparsifier import BaseSparsifier

from sparsimony.distributions.base import BaseDistribution
from sparsimony.schedulers.base import BaseScheduler
from sparsimony.utils import get_mask, get_original_tensor
from sparsimony.dst.base import DSTMixin
from sparsimony.mask_calculators import (
    NMPruner,
    NMGrower,
    MagnitudeScorer,
    RandomScorer,
)


class SSET(DSTMixin, BaseSparsifier):

    def __init__(
        self,
        scheduler: BaseScheduler,
        distribution: BaseDistribution,
        optimizer: torch.optim.Optimizer,
        defaults: Optional[Dict[str, Any]] = None,
        sparsity: float = 0.5,
        grown_weights_init: float = 0.0,
        init_method: Optional[str] = "grad_flow",
        n: int = 2,
        m: int = 4,
        pad: bool = False,
        padding_dim: int = 1,
        permute_conv_to_nhwc: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            scheduler,
            distribution,
            optimizer,
            defaults,
            sparsity,
            grown_weights_init,
            init_method,
            *args,
            **kwargs,
        )
        self.grown_weights_init = grown_weights_init
        self.init_method = init_method
        self.n = n
        self.m = m
        self.pad = pad
        self.padding_dim = padding_dim
        self.permute_conv_to_nhwc = permute_conv_to_nhwc
        if defaults is None:
            defaults = dict()
        if self.sparsity is None:
            self.sparsity = 1 - (self.n / self.m)
        if self.sparsity != 1 - (self.n / self.m):
            self._logger.warning(
                f"Must set sparsity to None or {1-self.n/self.m} for "
                f"{self.n}:{self.m} sparse training. Setting to "
                f" {1-self.n/self.m}"
            )
            self.sparsity = 1 - (self.n / self.m)
        if not self.permute_conv_to_nhwc:
            self._logger.warning(
                "permute_conv_to_nhwc is False. Typically 2:4"
                " kernels for conv require this option set to"
                " true."
            )
        super().__init__(optimizer=optimizer, defaults=defaults)
        self.pruner = NMPruner(
            scorer=MagnitudeScorer,
            n=self.n,
            m=self.m,
            pad=self.pad,
            padding_dim=self.padding_dim,
            permute_conv_to_nhwc=self.permute_conv_to_nhwc,
        )
        self.grower = NMGrower(
            scorer=RandomScorer,
            n=self.n,
            m=self.m,
            pad=self.pad,
            padding_dim=self.padding_dim,
            permute_conv_to_nhwc=self.permute_conv_to_nhwc,
        )
        if self.global_pruning:
            raise ValueError("Cannot use global pruning with SSET")

    def _step(self) -> bool:
        _topo_updated = False
        self._step_count += 1
        prune_ratio = self.scheduler(self._step_count)
        if prune_ratio is not None:
            self._distribute_sparsity(self.sparsity)
            for config in self.groups:
                config["prune_ratio"] = prune_ratio
                self.update_mask(**config)
            self._broadcast_masks()
            _topo_updated = True
        self._step_count += 1
        return _topo_updated

    def update_mask(
        self,
        module: nn.Module,
        tensor_name: str,
        sparsity: float,
        prune_ratio: float,
        **kwargs,
    ):
        mask = get_mask(module, tensor_name)
        if sparsity == 0:
            mask.data = torch.ones_like(mask)
        else:
            weights = getattr(module, tensor_name)
            original_weights = get_original_tensor(module, tensor_name)
            target_sparsity = self.get_sparsity_from_prune_ratio(prune_ratio)
            self.prune_mask(target_sparsity, mask, weights)
            self.grow_mask(sparsity, mask, original_weights)
            self._assert_sparsity_level(mask, sparsity)

    def _initialize_masks(self) -> None:
        self._distribute_sparsity(self.sparsity)
        for config in self.groups:
            # Prune to target sparsity for this step
            mask = get_mask(config["module"], config["tensor_name"])
            weights = getattr(config["module"], config["tensor_name"])
            self.prune_mask(config["sparsity"], mask, weights)
