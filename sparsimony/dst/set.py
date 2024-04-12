from typing import Optional
import torch
import torch.nn as nn
from torch.ao.pruning.sparsifier.base_sparsifier import BaseSparsifier

from sparsimony.distributions.base import BaseDistribution
from sparsimony.schedulers.base import BaseScheduler
from sparsimony.utils import get_mask, get_original_tensor
from sparsimony.dst.base import DSTMixin
from sparsimony.pruners.unstructured import (
    UnstructuredMagnitudePruner,
    UnstructuredRandomGrower,
)

"""        "module_fqn": module_fqn,
        "module": module,
        "tensor_name": tensor_name,
        "tensor_fqn": tensor_fqn,

"""


class SET(DSTMixin, BaseSparsifier):

    def __init__(
        self,
        scheduler: BaseScheduler,
        distribution: BaseDistribution,
        optimizer: torch.optim.Optimizer,
        sparsity: float = 0.5,
        grown_weights_init: float = 0.0,
        init_method: Optional[str] = "grad_flow",
    ):
        self.scheduler = scheduler
        self.distribution = distribution
        self.sparsity = sparsity
        self.grown_weights_init = grown_weights_init
        self.init_method = init_method
        defaults = dict()
        super().__init__(optimizer=optimizer, defaults=defaults)

    def prune_mask(
        self,
        prune_ratio: float,
        mask: torch.Tensor,
        weights: torch.Tensor,
        *args,
        **kwargs
    ) -> torch.Tensor:
        mask.data = UnstructuredMagnitudePruner.calculate_mask(
            prune_ratio, mask, weights
        )
        return mask

    def grow_mask(
        self,
        sparsity: float,
        mask: torch.Tensor,
        original_weights: torch.Tensor,
    ):
        # Grow new weights
        new_mask = UnstructuredRandomGrower.calculate_mask(
            sparsity,
            mask,
        )
        assert new_mask.data_ptr() != mask.data_ptr()
        # Assign newly grown weights to self.grown_weights_init
        torch.where(
            new_mask != mask,
            torch.full_like(
                original_weights, fill_value=self.grown_weights_init
            ),
            original_weights,
        )
        # Overwrite old mask
        mask.data = new_mask.data

    def _step(self) -> None:
        self._step_count += 1
        prune_ratio = self.scheduler(self._step_count)
        if prune_ratio is not None:
            self._distribute_sparsity(self.sparsity)
            for config in self.groups:
                config["prune_ratio"] = prune_ratio
                self.update_mask(**config)
            self._broadcast_masks()
        self._step_count += 1

    def _assert_sparsity_level(self, mask, sparsity_level):
        assert mask.sum() == int(mask.numel() * sparsity_level)

    def update_mask(
        self,
        module: nn.Module,
        tensor_name: str,
        sparsity: float,
        prune_ratio: float,
        **kwargs
    ):
        mask = get_mask(module, tensor_name)
        if sparsity == 0:
            mask.data = torch.ones_like(mask)
        else:
            weights = getattr(module, tensor_name)
            original_weights = get_original_tensor(module, tensor_name)
            self.prune_mask(prune_ratio, mask, weights)
            self.grow_mask(sparsity, mask, original_weights)
            self._assert_sparsity_level(mask, sparsity)

    def _initialize_masks(self) -> None:
        self._distribute_sparsity(self.sparsity)
        for config in self.groups:
            # Prune to target sparsity for this step
            mask = get_mask(config["module"], config["tensor_name"])
            weights = getattr(config["module"], config["tensor_name"])
            self.prune_mask(config["sparsity"], mask, weights)
