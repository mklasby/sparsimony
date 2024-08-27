import math
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.ao.pruning.sparsifier.base_sparsifier import BaseSparsifier

from sparsimony.distributions.base import BaseDistribution
from sparsimony.schedulers.base import BaseScheduler
from sparsimony.utils import get_mask, get_original_tensor
from sparsimony.dst.base import DSTMixin
from sparsimony.mask_calculators import NMMagnitudePruner, NMRandomGrower


class SSET(DSTMixin, BaseSparsifier):

    def __init__(
        self,
        scheduler: BaseScheduler,
        distribution: BaseDistribution,
        optimizer: torch.optim.Optimizer,
        m: int = 4,
        padding_dim: int = 2,
        defaults: Optional[Dict[str, Any]] = None,
        sparsity: float = 0.5,
        grown_weights_init: float = 0.0,
        init_method: Optional[str] = "grad_flow",
        *args,
        **kwargs,
    ):
        self.scheduler = scheduler
        self.distribution = distribution
        self.sparsity = sparsity
        self.grown_weights_init = grown_weights_init
        self.init_method = init_method
        self.m = m
        self.padding_dim = padding_dim
        if defaults is None:
            defaults = dict()
        super().__init__(optimizer=optimizer, defaults=defaults)

    def prune_mask(
        self,
        sparsity: float,
        mask: torch.Tensor,
        weights: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # always prune at least one el
        n = max(math.floor((1 - sparsity) * self.m), 1)
        pad = True
        if len(mask.shape) == 4:
            pad = False
        pruner = NMMagnitudePruner(
            n,
            self.m,
            pad=pad,
            padding_dim=self.padding_dim,
            permute_conv_to_nhwc=True,
        )
        pruner_kwargs = dict(mask=mask, weights=weights)
        mask.data = pruner.calculate_mask(**pruner_kwargs)
        return mask

    def grow_mask(
        self,
        sparsity: float,
        mask: torch.Tensor,
        original_weights: torch.Tensor,
    ):
        n = max(math.floor((1 - sparsity) * self.m), 1)
        pad = True
        if len(mask.shape) == 4:  # Don't pad convs
            pad = False
        grower = NMRandomGrower(
            n,
            self.m,
            pad=pad,
            padding_dim=self.padding_dim,
            permute_conv_to_nhwc=True,
        )
        old_mask = mask.clone()
        # Grow new weights
        new_mask = grower.calculate_mask(
            sparsity,
            mask,
        )
        # Assign newly grown weights to self.grown_weights_init
        torch.where(
            new_mask != old_mask,
            torch.full_like(
                original_weights, fill_value=self.grown_weights_init
            ),
            original_weights,
        )
        # Overwrite old mask
        mask.data = new_mask.data

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
