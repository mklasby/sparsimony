from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.ao.pruning.sparsifier.base_sparsifier import BaseSparsifier

from sparsimony.distributions.base import BaseDistribution
from sparsimony.schedulers.base import BaseScheduler
from sparsimony.parametrization.fake_sparsity import FakeSparsity
from sparsimony.utils import get_mask
from sparsimony.dst.base import DSTMixin
from sparsimony.pruners.unstructured import (
    UnstructuredMagnitudePruner,
)


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
    ):
        self.scheduler = scheduler
        self.distribution = distribution
        self.sparsity = self.scheduler.initial_sparsity
        if defaults is None:
            defaults = dict(parametrization=FakeSparsity)
        super().__init__(optimizer=optimizer, defaults=defaults)

    def prune_mask(
        self,
        prune_ratio: float,
        mask: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        mask.data = UnstructuredMagnitudePruner.calculate_mask(
            prune_ratio, mask, weights
        )
        return mask

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
            self._distribute_sparsity(sparsity)
            for config in self.groups:
                self.update_mask(**config)
            self._broadcast_masks()
            _topo_updated = True
        return _topo_updated

    def _initialize_masks(self) -> None:
        self._distribute_sparsity(self.scheduler.initial_sparsity)
        for config in self.groups:
            # Prune to target sparsity for this step
            mask = get_mask(config["module"], config["tensor_name"])
            weights = weights = getattr(config["module"], config["tensor_name"])
            mask.data = UnstructuredMagnitudePruner.calculate_mask(
                config["sparsity"], mask, weights
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
            prune_ratio = DSTMixin.get_prune_ratio_from_sparsity(mask, sparsity)
            self.prune_mask(prune_ratio, mask, weights)
            self._assert_sparsity_level(mask, sparsity)

    def _assert_sparsity_level(self, mask: torch.Tensor, sparsity_level: float):
        n_ones = mask.sum()
        actual_n_ones = int(mask.numel() * (1 - sparsity_level))
        if n_ones != actual_n_ones and abs(n_ones - actual_n_ones) > 1:
            # With very large mask tensors, we may have some prec ision errors
            # with exact n_ones. Therefore, we simply log the warning instead of
            # raising.
            self._logger.warning(
                f"n_ones actual{n_ones} != n_one target {actual_n_ones}"
            )
