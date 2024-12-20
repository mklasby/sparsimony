from typing import Optional, Dict, Any

import torch
from torch.ao.pruning.sparsifier.base_sparsifier import BaseSparsifier

from sparsimony.dst.base import DSTMixin
from sparsimony.distributions.base import BaseDistribution
from sparsimony.parametrization.fake_sparsity import FakeSparsity
from sparsimony.mask_calculators import UnstructuredPruner, MagnitudeScorer
from sparsimony.utils import get_mask
from sparsimony.schedulers.base import BaseScheduler, StaticScheduler


class StaticMagnitudeSparsifier(DSTMixin, BaseSparsifier):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        distribution: BaseDistribution,
        sparsity: float,
        defaults: Optional[Dict[str, Any]] = None,
        scheduler: BaseScheduler | None = None,
        *args,
        **kwargs,
    ):
        optimizer = optimizer
        if scheduler is None:
            scheduler = StaticScheduler()
        self.scheduler = scheduler
        self.distribution = distribution
        self.sparsity = sparsity
        if defaults is None:
            defaults = dict(parametrization=FakeSparsity)
        super().__init__(
            optimizer=optimizer, defaults=defaults, *args, **kwargs
        )
        self.pruner = UnstructuredPruner(scorer=MagnitudeScorer)

    def _assert_sparsity_level(self, mask: torch.Tensor, sparsity_level: float):
        n_ones = mask.sum(dtype=torch.int)
        actual_n_ones = int(mask.numel() * (1 - sparsity_level))
        if abs(n_ones - actual_n_ones) > 1:
            raise RuntimeError(f"Found sparsity of {n_ones} != {actual_n_ones}")

    def _initialize_masks(self):
        self._distribute_sparsity(self.sparsity)
        if self.global_pruning:
            self._global_init_prune()
            return
        for config in self.groups:
            # Prune to target sparsity for this step
            mask = get_mask(config["module"], config["tensor_name"])
            weights = getattr(config["module"], config["tensor_name"])
            mask.data = self.prune_mask(
                config["sparsity"], mask, values=weights
            )
            self._assert_sparsity_level(mask.data, self.sparsity)

    def _step(self):
        self._step_count += 1
        # Basically do nothing to change the mask

    def grow_mask(self):
        pass

    def update_mask(self):
        pass
