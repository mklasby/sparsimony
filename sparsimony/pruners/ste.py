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
from sparsimony.utils import get_mask
from sparsimony.schedulers.base import BaseScheduler
from sparsimony.mask_calculators import MagnitudeScorer, NMPruner
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
        self.sparsity = 1 - n / m
        self.decay = decay
        self._logger = logging.getLogger(__name__)
        self.prepared_ = False
        self._step_count = 0
        self.pruner = NMPruner(MagnitudeScorer, n=self.n, m=self.m)
        if defaults is None:
            defaults = {}
        ste_parametrization = (
            FakeSparsitySTE if decay is None else FakeSparsitySRSTE
        )
        defaults["parametrization"] = ste_parametrization
        super().__init__(defaults)

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
        self.distribution((1 - self.n / self.m), self.groups)
        for config in self.groups:
            # Update n in case distribution modified sparsity targets
            self.update_mask(**config)
        self.prepared_ = True
        self._logger.info(f"Masks prepared in {time.time()-_start} seconds.")

    # @override
    def step(self) -> bool:
        self._logger.debug("SR-STE step() in prog...")
        start = time.time()
        _topo_updated = False
        self._step_count += 1
        prune_ratio = self.scheduler(self._step_count)
        if prune_ratio is not None:
            _topo_updated = True
            self.distribution((1 - self.n / self.m), self.groups)
            for config in self.groups:
                self.update_mask(**config)
        self._logger.debug(
            f"SR-STE step completd in {time.time() - start} seconds"
        )
        return _topo_updated

    # @override
    def update_mask(
        self,
        module: nn.Module,
        tensor_name: str,
        sparsity: float,
        tensor_fqn: str,
        **kwargs,
    ):
        self._logger.debug(f"Updating mask for {tensor_fqn}...")
        mask = get_mask(module, tensor_name)
        # set all elements to active after optim step and reprune
        mask.data = torch.ones_like(mask, dtype=torch.bool)
        if sparsity == 0:
            return
        original_weights = getattr(
            module.parametrizations, tensor_name
        ).original
        mask.data = self.pruner.calculate_mask(
            self.sparsity, mask, values=original_weights
        )
        self._assert_sparsity_level(mask, sparsity)
        self._assert_structure(mask, tensor_fqn)

    # @override
    def _prepare(self, *args, **kwargs):
        r"""Adds mask parametrization to the layer weight"""
        for config in self.groups:
            module = config["module"]
            tensor_name = config["tensor_name"]
            parametrization = config.get("parametrization")
            mask = torch.ones_like(
                getattr(module, tensor_name), dtype=torch.bool
            )
            register_parametrization(
                module,
                tensor_name,
                parametrization(mask, n=self.n, m=self.m, decay=self.decay),
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

    def _assert_sparsity_level(self, mask: torch.Tensor, sparsity_level: float):
        n_ones = mask.count_nonzero()
        target_n_ones = int(mask.numel() * (1 - sparsity_level))
        # We ignore off-by-one errors as these will be due to floor ops
        if n_ones != target_n_ones and abs(n_ones - target_n_ones) > 1:
            # With very large mask tensors, we may have some precision errors
            # with exact n_ones. Therefore, we simply log the warning instead of
            # raising.
            # Also naturally occurs in structured pruning
            # TODO: For structured pruning we may wish to calculate
            # actual_n_ones based on network topology
            self._logger.warning(
                f"n_ones actual {n_ones} != n_one target {target_n_ones}"
            )

    def _assert_structure(self, mask, fqn: str) -> None:
        if self.n == 2 and self.m == 4:
            if mask.shape[1] % 64 != 0:
                self._logger.warning(
                    f"Mask shape is not a multiple of 64, this weight tensor "
                    "may not work with torch 2:4 semi-structured kernels!\n"
                    f"Mask shape: {mask.shape} found at {fqn}"
                )
        try:
            mask_view = mask.view(-1, self.m)
        except RuntimeError as e:
            self._logger.error(f"fqn: {fqn}")
            raise e
        ones = torch.count_nonzero(mask_view, dim=-1)
        if (ones != self.n).all():
            self._logger.warning(
                f"{fqn} mask is not {self.n}:{self.m} pruned! Ones Tensor:\n"
                f"{ones}"
            )
            raise RuntimeError(
                f"N:M Violation found: {ones.unique()} n's in layer {fqn}"
            )
