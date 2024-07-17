import torch
from torch.ao.pruning.sparsifier.base_sparsifier import BaseSparsifier

from sparsimony.dst.base import DSTMixin
from sparsimony.distributions.base import BaseDistribution
from sparsimony.parametrization.fake_sparsity import FakeSparsity
from sparsimony.pruners.unstructured import UnstructuredMagnitudePruner
from sparsimony.utils import get_mask


# TODO - double check init_method
class StaticMagnitudeSparsifier(DSTMixin, BaseSparsifier):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        distribution: BaseDistribution,
        sparsity: float,
        init_method: str = "sparse_torch",
    ):
        optimizer = optimizer
        self.distribution = distribution
        self.sparsity = sparsity
        self.init_method = init_method
        defaults = dict(parametrization=FakeSparsity)
        super().__init__(optimizer=optimizer, defaults=defaults)

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

    def __str__(self) -> str:
        # TODO: Errors if sparsifier has not been prepared. Fix me
        def neuron_is_active(neuron):
            return neuron.any()

        global_sparsity = self.calculate_global_sparsity().item()
        layerwise_sparsity_target = []
        layerwise_sparsity_actual = []
        active_neurons = []
        total_neurons = []
        for config in self.groups:
            layerwise_sparsity_target.append(config["sparsity"])
            mask = get_mask(**config)
            layerwise_sparsity_actual.append(
                self.calculate_mask_sparsity(mask).item()
            )
            active_neurons.append(
                torch.vmap(neuron_is_active)(mask).sum().item()
            )
            total_neurons.append(len(mask))
        active_vs_total_neurons = []
        for a, t in list(zip(active_neurons, total_neurons)):
            active_vs_total_neurons.append(f"{a}/{t}")
        # TODO: Should list ignored_layers from distribution
        return (
            f"{self.__class__.__name__}\n"
            f"Step No.: {self._step_count}\n"
            f"Distribution: {self.distribution.__class__.__name__}\n"
            f"Global Sparsity Target: {self.sparsity}\n"
            f"Global Sparsity Actual: {global_sparsity}\n"
            f"Layerwise Sparsity Targets: {layerwise_sparsity_target}\n"
            f"Layerwise Sparsity Actual: {layerwise_sparsity_actual}\n"
            f"Active/Total Neurons: {active_vs_total_neurons}"
        )
