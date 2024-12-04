from typing import Optional, Dict, Any

import torch
from torch.ao.pruning.sparsifier.base_sparsifier import BaseSparsifier
from torch.nn.utils.parametrize import (
    register_parametrization,
)
import deepspeed

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
        super().__init__(optimizer=optimizer, defaults=defaults, *args, **kwargs)
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
            mask.data = self.prune_mask(config["sparsity"], mask, values=weights)
            self._assert_sparsity_level(mask.data, self.sparsity)

    def _step(self):
        self._step_count += 1
        # Basically do nothing to change the mask

    def grow_mask(self):
        pass

    def update_mask(self):
        pass


class StaticSparsifier(DSTMixin, BaseSparsifier):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        sparsity: float,
        defaults: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        optimizer = optimizer
        self.sparsity = sparsity
        if defaults is None:
            defaults = dict(parametrization=FakeSparsity)
        super().__init__(optimizer=optimizer, defaults=defaults, *args, **kwargs)

    def _prepare(
        self,
        *args,
        **kwargs,
    ):
        for config in self.groups:
            module = config["module"]
            tensor_name = config["tensor_name"]
            parametrization = config.get("parametrization")
            param = getattr(module, tensor_name)
            unsafe = False
            if len(param) == 0 and hasattr(param, "_z3_optimizer"):
                # zero init from deepspeed
                param = deepspeed.utils.safe_get_full_fp32_param(param)
                unsafe = True  # param shape will be [0]
            mask = torch.ones_like(param, dtype=torch.bool)
            register_parametrization(
                module, tensor_name, parametrization(mask), unsafe=unsafe
            )

    def _assert_sparsity_level(self, mask: torch.Tensor, sparsity_level: float):
        n_ones = mask.sum(dtype=torch.int)
        actual_n_ones = int(mask.numel() * (1 - sparsity_level))
        if abs(n_ones - actual_n_ones) > 1:
            # raise RuntimeError(f"Found sparsity of {n_ones} != {actual_n_ones}")
            self._logger.warning(
                "Actual n_ones != target_n_ones " f"({n_ones} != {actual_n_ones})"
            )

    def _initialize_masks(self):
        for config in self.groups:
            config["sparsity"] = self.sparsity
            # Prune to target sparsity for this step
            mask = get_mask(config["module"], config["tensor_name"])
            weights = getattr(config["module"], config["tensor_name"])
            mask.data = weights != 0
            self._assert_sparsity_level(mask.data, self.sparsity)

    def _step(self) -> bool:
        self._step_count += 1
        return False

    def grow_mask(self):
        pass

    def update_mask(self):
        pass

    def __str__(self) -> str:
        def neuron_is_active(neuron):
            return neuron.any()

        if self.prepared_:
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
                active_neurons.append(torch.vmap(neuron_is_active)(mask).sum().item())
                total_neurons.append(len(mask))
            active_vs_total_neurons = []
            for a, t in list(zip(active_neurons, total_neurons)):
                active_vs_total_neurons.append(f"{a}/{t}")
            # TODO: Should list ignored_layers from distribution
        else:
            err_message = "Error: Sparsifier's prepare() method must be called first."
            global_sparsity = err_message
            layerwise_sparsity_actual = err_message
            layerwise_sparsity_target = err_message
            active_vs_total_neurons = err_message
        return (
            f"{self.__class__.__name__}\n"
            f"Step No.: {self._step_count}\n"
            # TODO: adjust for skipped mods/layers
            f"Global Sparsity Target: {self.sparsity}\n"
            f"Global Sparsity Actual: {global_sparsity}\n"
            f"Layerwise Sparsity Targets: {layerwise_sparsity_target}\n"
            f"Layerwise Sparsity Actual: {layerwise_sparsity_actual}\n"
            f"Active/Total Neurons: {active_vs_total_neurons}\n"
        )
