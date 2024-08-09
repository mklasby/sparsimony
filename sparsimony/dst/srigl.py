from typing import Optional, Dict, Any
from math import prod

import torch
import torch.nn as nn
from torch.ao.pruning.sparsifier.base_sparsifier import BaseSparsifier

from sparsimony.distributions.base import BaseDistribution
from sparsimony.schedulers.base import BaseScheduler
from sparsimony.parametrization.fake_sparsity import FakeSparsityDenseGradBuffer
from sparsimony.utils import get_mask, get_parametrization
from sparsimony.dst.base import DSTMixin
from sparsimony.mask_calculators import (
    FFIRandomPruner,
    FFIGradientGrower,
    UnstructuredMagnitudePruner,
)


class SRigL(DSTMixin, BaseSparsifier):

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
            defaults = dict(parametrization=FakeSparsityDenseGradBuffer)
        super().__init__(
            optimizer=optimizer, defaults=defaults, *args, **kwargs
        )

    def prune_mask(
        self,
        sparsity: float,
        mask: torch.Tensor,
        weights: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        mask.data = UnstructuredMagnitudePruner.calculate_mask(
            sparsity, mask, weights=weights
        )
        return mask

    def grow_mask(
        self,
        sparsity: float,
        mask: torch.Tensor,
        original_weights: torch.Tensor,
        dense_grads: torch.Tensor,
    ) -> torch.Tensor:
        old_mask = mask.clone()
        # Grow new weights
        new_mask = FFIGradientGrower.calculate_mask(
            sparsity, mask, grads=dense_grads
        )
        assert new_mask.data_ptr() != old_mask.data_ptr()
        # Assign newly grown weights to self.grown_weights_init in place
        original_weights.data = torch.where(
            new_mask != old_mask,
            torch.full_like(
                original_weights, fill_value=self.grown_weights_init
            ),
            original_weights,
        )
        # Overwrite old mask
        mask.data = new_mask.data
        return mask

    def _step(self) -> bool:
        _topo_updated = False
        self._step_count += 1
        prune_ratio = self.scheduler(self._step_count)
        if prune_ratio is not None:
            self._logger.info(f"Updating topology at step {self._step_count}")
            self._distribute_sparsity(self.sparsity)
            for config in self.groups:
                parametrization = get_parametrization(
                    config["module"], config["tensor_name"]
                )
                if (
                    hasattr(parametrization, "is_replica_")
                    and parametrization.is_replica_
                ):
                    continue
                config["prune_ratio"] = prune_ratio
                config["dense_grads"] = self._get_dense_grads(**config)
                self.update_mask(**config)
            self._broadcast_masks()
            _topo_updated = True
        if self.scheduler.next_step_update(self._step_count):
            self._accumulate_grads()
        return _topo_updated

    def _get_dense_grads(
        self, module: nn.Module, tensor_name: str, **kwargs
    ) -> torch.Tensor:
        parametrization = getattr(module.parametrizations, tensor_name)[0]
        parametrization.accumulate = False
        return parametrization.dense_grad

    def _accumulate_grads(self) -> None:
        for config in self.groups:
            parametrization = get_parametrization(
                config["module"], tensor_name=config["tensor_name"]
            )
            parametrization.accumulate = True

    def _initialize_masks(self) -> None:
        self._distribute_sparsity(self.sparsity)
        for config in self.groups:
            # Prune to target sparsity for this step
            mask = get_mask(config["module"], config["tensor_name"])
            if self.random_mask_init:
                # Randomly prune for step 1
                mask.data = FFIRandomPruner.calculate_mask(
                    config["sparsity"], mask
                )
            else:
                # use pruning criterion
                weights = getattr(config["module"], config["tensor_name"])
                mask.data = self.prune_mask(config["sparsity"], mask, weights)

    def update_mask(
        self,
        module: nn.Module,
        tensor_name: str,
        sparsity: float,
        prune_ratio: float,
        dense_grads: torch.Tensor,
        **kwargs,
    ):
        mask = get_mask(module, tensor_name)
        if sparsity == 0:
            mask.data = torch.ones_like(mask)
        else:
            original_weights = getattr(
                module.parametrizations, tensor_name
            ).original
            weights = getattr(module, tensor_name)
            target_sparsity = self.get_sparsity_from_prune_ratio(
                mask, prune_ratio
            )
            self.prune_mask(target_sparsity, mask, weights)
            self.grow_mask(sparsity, mask, original_weights, dense_grads)
            self._assert_sparsity_level(mask, sparsity)

    def _assert_sparsity_level(self, mask: torch.Tensor, sparsity_level: float):
        n_ones = mask.sum()
        actual_n_ones = int(mask.numel() * (1 - sparsity_level))
        if n_ones != actual_n_ones:
            # With very large mask tensors, we may have some precision errors
            # with exact n_ones. Therefore, we simply log the warning instead of
            # raising.
            # Also naturally occurs in structured pruning
            # TODO: For structured pruning we may wish to calculate
            # actual_n_ones based on network topology
            self._logger.debug(
                f"n_ones actual{n_ones} != n_one target {actual_n_ones}"
            )
        ffi = torch.tensor([m.sum() for m in mask])
        _error = False
        if len(ffi.unique()) == 1:
            return
        elif len(ffi.unique()) == 2:
            if not (ffi.unique() == 0).any():
                _error = True
        else:
            _error = True
        if _error:
            self._logger.debug(
                "FFI error: Multiple non-zero FFI values detected: "
                f"{ffi.unique()}"
            )

    def __str__(self) -> str:
        ffi = []
        for config in self.groups:
            mask = get_mask(**config)
            mask_flat = mask.view(mask.shape[0], prod(mask.shape[1:]))
            ffi.append(mask_flat.sum(dim=1, dtype=torch.int).unique().item())
        s = super().__str__()
        s += f"FFI: {ffi}\n"
        return s
