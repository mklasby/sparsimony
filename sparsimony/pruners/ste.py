import math
import time
from typing import Optional, Dict, Any
import logging

import torch
import torch.nn as nn
from torch.nn.utils.parametrize import (
    register_parametrization,
)
from torch.ao.pruning.sparsifier.base_sparsifier import BaseSparsifier

from sparsimony.distributions.base import BaseDistribution
from sparsimony.utils import get_mask, get_parametrization
from sparsimony.schedulers.base import BaseScheduler

from sparsimony.parametrization.ste_parametrization import (
    FakeSparsitySRSTE,
    FakeSparsitySTE,
)


class SRSTESparsifier(BaseSparsifier):
    def __init__(
        self,
        scheduler: BaseScheduler,
        distribution: BaseDistribution,
        n: int = 2,
        m: int = 4,
        decay: None | float = 2e-4,
        defaults: Optional[Dict[str, Any]] = None,
    ):
        self.scheduler = scheduler
        self.distribution = distribution
        self.n = n
        self.m = m
        self.sparsity = n / m
        self.decay = decay
        if defaults is None:
            defaults = {}
        ste_parametrization = (
            FakeSparsitySTE if decay is None else FakeSparsitySRSTE
        )
        defaults["parametrization"] = ste_parametrization
        super().__init__(defaults)
        self._logger = logging.getLogger(__name__)
        self.prepared_ = False
        self._step_count = 0

    # @overide
    @torch.no_grad()
    def prepare(
        self,
        model: nn.Module,
        sparse_config: Dict[str, Any],
    ):
        _start = time.time()
        self._logger.info("Preparing masks...")
        super().prepare(model, sparse_config)
        self.distribution(self.n / self.m, self.groups)
        for config in self.groups:
            # Update n in case distribution modified sparsity targets
            self.update_mask(**config)
        self.prepared_ = True
        self._logger.info(f"Masks prepared in {time.time()-_start} seconds.")

    # @override
    def step(self) -> bool:
        _topo_updated = False
        self._step_count += 1
        prune_ratio = self.scheduler(self._step_count)
        if prune_ratio is not None:
            _topo_updated = True
            self.distribution(self.n / self.m, self.groups)
            for config in self.groups:
                self._update_mask(**config)
        return _topo_updated

    # @override
    def update_mask(
        self, module: nn.Module, tensor_name: str, sparsity: float, **kwargs
    ):
        parametrization = get_parametrization(module, tensor_name)
        new_n = math.floor(sparsity * self.m)
        parametrization.n = new_n

    # @override
    def _prepare(self, *args, **kwargs):
        r"""Adds mask parametrization to the layer weight"""
        for config in self.groups:
            module = config["module"]
            tensor_name = config["tensor_name"]
            parametrization = config.get("parametrization")
            register_parametrization(
                module,
                tensor_name,
                parametrization(n=self.n, m=self.m, decay=self.decay),
            )

    def calculate_global_sparsity(self):
        total_el = 0
        nnz_el = 0
        for config in self.groups:
            mask = get_mask(config["module"], config["tensor_name"])
            total_el += mask.numel()
            nnz_el += mask.sum()
        return 1 - (nnz_el / total_el)

    def calculate_mask_sparsity(self, mask: torch.Tensor):
        return 1 - (mask.sum() / mask.numel())

    def __str__(self) -> str:
        if self.prepared_:
            global_sparsity = self.calculate_global_sparsity().item()
            layerwise_sparsity_target = []
            layerwise_sparsity_actual = []
            for config in self.groups:
                layerwise_sparsity_target.append(float(config["sparsity"]))
                mask = get_mask(**config)
                layerwise_sparsity_actual.append(
                    self.calculate_mask_sparsity(mask).item()
                )
        else:
            err_message = (
                "Error: Sparsifier's prepare() method must be called first."
            )
            global_sparsity = err_message
            layerwise_sparsity_actual = err_message
            layerwise_sparsity_target = err_message
        return (
            f"{self.__class__.__name__}\n"
            f"Scheduler: {self.scheduler.__class__.__name__}\n"
            f"Distribution: {self.distribution.__class__.__name__}\n"
            # TODO: adjust for skipped mods/layers
            f"Step No.: {self._step_count}\n"
            f"Global Sparsity Target: {self.sparsity}\n"
            f"Global Sparsity Actual: {global_sparsity}\n"
            f"Layerwise Sparsity Targets: {layerwise_sparsity_target}\n"
            f"Layerwise Sparsity Actual: {layerwise_sparsity_actual}\n"
        )

    def __repr__(self) -> str:
        return self.__str__()
