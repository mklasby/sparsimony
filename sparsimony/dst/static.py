import torch
from torch.ao.pruning.sparsifier.base_sparsifier import BaseSparsifier

from sparsimony.dst.base import DSTMixin
from sparsimony.distributions.base import BaseDistribution
from sparsimony.parametrization.fake_sparsity import FakeSparsity
from sparsimony.pruners.unstructured import UnstructuredMagnitudePruner
from sparsimony.utils import get_mask


class StaticMagnitudeSparsifier(DSTMixin, BaseSparsifier):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        distribution: BaseDistribution,
        sparsity: float,
    ):
        optimizer = optimizer
        self.distribution = distribution
        self.sparsity = sparsity
        defaults = dict(parametrization=FakeSparsity)
        super().__init__(optimizer=optimizer, defaults=defaults)

    def _assert_sparsity_level(self, mask: torch.Tensor, sparsity_level: float):
        n_ones = mask.sum()
        actual_n_ones = int(mask.numel() * (1 - sparsity_level))
        if abs(n_ones - actual_n_ones) > 1:
            raise RuntimeError(f"Found sparsity of {n_ones} != {actual_n_ones}")

    def _initialize_masks(self):
        self._distribute_sparsity(self.sparsity)
        for config in self.groups:
            # Prune to target sparsity for this step
            mask = get_mask(config["module"], config["tensor_name"])
            weights = getattr(config["module"], config["tensor_name"])
            mask.data = UnstructuredMagnitudePruner.calculate_mask(
                config["sparsity"], mask, weights
            )
            self._assert_sparsity_level(mask.data, self.sparsity)

    def _step(self):
        self._step_count += 1
        # Basically do nothing to change the mask

    def grow_mask(self):
        pass

    def prune_mask(self):
        pass

    def update_mask(self):
        pass
