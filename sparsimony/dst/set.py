from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.ao.pruning.sparsifier.base_sparsifier import BaseSparsifier

from sparsimony.parametrization.fake_sparsity import FakeSparsity
from sparsimony.distributions.base import BaseDistribution
from sparsimony.schedulers.base import BaseScheduler
from sparsimony.utils import get_mask, get_original_tensor
from sparsimony.dst.base import DSTMixin, GlobalPruningDataHelper
from sparsimony.mask_calculators import (
    UnstructuredPruner,
    UnstructuredGrower,
    MagnitudeScorer,
    RandomScorer,
)


class SET(DSTMixin, BaseSparsifier):

    def __init__(
        self,
        scheduler: BaseScheduler,
        distribution: BaseDistribution,
        optimizer: torch.optim.Optimizer,
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
        if defaults is None:
            defaults = dict(parametrization=FakeSparsity)
        super().__init__(
            optimizer=optimizer, defaults=defaults, *args, **kwargs
        )
        self.pruner = UnstructuredPruner(scorer=MagnitudeScorer)
        self.grower = UnstructuredGrower(scorer=RandomScorer)

    def _step(self) -> bool:
        _topo_updated = False
        self._step_count += 1
        prune_ratio = self.scheduler(self._step_count)
        if prune_ratio is not None:
            if self.global_pruning:
                self._global_step(prune_ratio)
            else:
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
            self.prune_mask(target_sparsity, mask, values=weights)
            self.grow_mask(sparsity, mask, original_weights)
            self._assert_sparsity_level(mask, sparsity)

    def _initialize_masks(self) -> None:
        self._distribute_sparsity(self.sparsity)
        if self.global_pruning:
            self._global_init_prune()
            return
        for config in self.groups:
            # Prune to target sparsity for this step
            mask = get_mask(config["module"], config["tensor_name"])
            weights = getattr(config["module"], config["tensor_name"])
            self.prune_mask(config["sparsity"], mask, values=weights)

    def _global_step(self, prune_ratio: float) -> None:
        global_data_helper = GlobalPruningDataHelper(
            self.groups, self.global_buffers_cpu_offload
        )
        target_sparsity = self.get_sparsity_from_prune_ratio(
            global_data_helper.masks, prune_ratio
        )
        self.prune_mask(
            target_sparsity,
            global_data_helper.masks,
            values=global_data_helper.original_weights,
        )
        self.grow_mask(
            self.sparsity,
            global_data_helper.masks,
            global_data_helper.original_weights,
        )
        self._assert_sparsity_level(global_data_helper.masks, self.sparsity)
        global_data_helper.reshape_and_assign_masks()
