from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.ao.pruning.sparsifier.base_sparsifier import BaseSparsifier

from sparsimony.distributions.base import BaseDistribution
from sparsimony.schedulers.base import BaseScheduler
from sparsimony.parametrization.fake_sparsity import FakeSparsity
from sparsimony.utils import get_mask
from sparsimony.dst.base import DSTMixin, GlobalPruningDataHelper
from sparsimony.mask_calculators import UnstructuredPruner, MagnitudeScorer


class GMP(DSTMixin, BaseSparsifier):
    """GMP* implementation by Kurtic et al.
    https://proceedings.mlr.press/v234/kurtic24a.html
    """

    def __init__(
        self,
        scheduler: BaseScheduler,
        distribution: BaseDistribution,
        optimizer: torch.optim.Optimizer,
        defaults: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        self.scheduler = scheduler
        self.distribution = distribution
        self.sparsity = self.scheduler.initial_sparsity
        if defaults is None:
            defaults = dict(parametrization=FakeSparsity)
        super().__init__(
            optimizer=optimizer, defaults=defaults, *args, **kwargs
        )
        self.pruner = UnstructuredPruner(MagnitudeScorer)

    def grow_mask(
        self,
    ) -> torch.Tensor:
        pass

    def _step(self) -> bool:
        _topo_updated = False
        self._step_count += 1
        sparsity = self.scheduler(self._step_count)
        if sparsity is not None and sparsity != self.scheduler.initial_sparsity:
            self.sparsity = sparsity
            self._logger.info(
                f"Pruning to {self.sparsity*100:.2f}% sparsity at step "
                f"{self._step_count}"
            )
            if self.global_pruning:
                self._global_step()
            else:
                self._distribute_sparsity(sparsity)
                for config in self.groups:
                    if self._is_replica(**config):
                        continue
                    self.update_mask(**config)
            self._broadcast_masks()
            _topo_updated = True
        return _topo_updated

    def _initialize_masks(self) -> None:
        self._distribute_sparsity(self.scheduler.initial_sparsity)
        if self.global_pruning:
            self._global_init_prune()
            return
        for config in self.groups:
            # Prune to target sparsity for this step
            mask = get_mask(config["module"], config["tensor_name"])
            weights = weights = getattr(config["module"], config["tensor_name"])
            mask.data = self.pruner.calculate_mask(
                config["sparsity"], mask, values=weights
            )

    def update_mask(
        self,
        module: nn.Module,
        tensor_name: str,
        sparsity: float,
        **kwargs,
    ):
        mask = get_mask(module, tensor_name)
        if sparsity == 0:
            mask.data = torch.ones_like(mask)
        else:
            weights = getattr(module, tensor_name)
            self.prune_mask(sparsity, mask, values=weights)
            self._assert_sparsity_level(mask, sparsity)

    def _global_step(self) -> None:
        global_data_helper = GlobalPruningDataHelper(
            self.groups, self.global_buffers_cpu_offload
        )
        self.prune_mask(
            self.sparsity,
            global_data_helper.masks,
            values=global_data_helper.sparse_weights,
        )
        self._assert_sparsity_level(global_data_helper.masks, self.sparsity)
        global_data_helper.reshape_and_assign_masks()
