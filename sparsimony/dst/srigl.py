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
    FFIMagnitudePruner,
    NMGradientGrower,
    NMRandomPruner,
    NMMagnitudePruner,
    UnstructuredMagnitudePruner,
    NeuronMagnitudePruner,
    HierarchicalMaskCalculator,
    NeuronSRigLPruner,
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
        random_mask_init: bool = False,
        gamma_sal: None | float = 0.3,
        *args,
        **kwargs,
    ):
        self.scheduler = scheduler
        self.distribution = distribution
        if sparsity != 0.5:
            self._logger.warning("Must set sparsity to 0.5 for 2:4")
            sparsity = 0.5
        self.sparsity = sparsity
        self.grown_weights_init = grown_weights_init
        self.init_method = init_method
        self.gamma_sal = gamma_sal
        if defaults is None:
            defaults = dict(parametrization=FakeSparsityDenseGradBuffer)
        super().__init__(
            optimizer=optimizer,
            defaults=defaults,
            random_mask_init=random_mask_init,
            *args,
            **kwargs,
        )

    def prune_mask(
        self,
        sparsity: float,
        mask: torch.Tensor,
        weights: torch.Tensor,
        dense_grads: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if self.gamma_sal is None:
            # Specify 50% sparsity per level
            # TODO: Specify a distribution for this?
            calcs = [NeuronMagnitudePruner, UnstructuredMagnitudePruner]
            sparsities = [sparsity / i for i in range(len(calcs), 0, -1)]
            calc_kwargs = [
                dict(weights=weights),
                dict(weights=weights),
            ]
        else:
            # Enable gamma_sal driven ablation
            calcs = [NeuronSRigLPruner, UnstructuredMagnitudePruner]
            sparsities = [sparsity, sparsity]
            calc_kwargs = [
                dict(weights=weights, grads=dense_grads),
                dict(weights=weights),
            ]
        mask.data = HierarchicalMaskCalculator.calculate_mask(
            sparsities, mask, calcs, calc_kwargs
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
        score_override = HierarchicalMaskCalculator._get_score_override(
            mask, tile_view=NeuronMagnitudePruner._TILE_VIEW
        )
        # Grow new weights
        new_mask = FFIGradientGrower.calculate_mask(
            sparsity,
            mask,
            grads=dense_grads,
            score_override=score_override,
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
            if config["sparsity"] == 0:
                continue
            # Prune to target sparsity for this step
            weights = getattr(config["module"], config["tensor_name"])
            mask = get_mask(config["module"], config["tensor_name"])
            if self.gamma_sal is None:
                calcs = [NeuronMagnitudePruner]
                calc_kwargs = [
                    dict(weights=weights),
                ]
                sparsities = [
                    config["sparsity"] / i for i in range(len(calcs), 0, -1)
                ]
            else:
                calcs = []
                calc_kwargs = []
                sparsities = [config["sparsity"]]
            if self.random_mask_init:
                # Randomly prune for step 1
                calcs.append(FFIRandomPruner)
                calc_kwargs.append({})
            else:
                # use FFI mag pruning criterion
                calcs.append(FFIMagnitudePruner)
                calc_kwargs.append(dict(weights=weights))
            mask.data = HierarchicalMaskCalculator.calculate_mask(
                sparsities, mask, calcs, calc_kwargs
            )
            self._assert_structure(mask, config["tensor_fqn"])

    def update_mask(
        self,
        module: nn.Module,
        tensor_name: str,
        sparsity: float,
        prune_ratio: float,
        dense_grads: torch.Tensor,
        tensor_fqn: str,
        **kwargs,
    ):
        self._logger.debug(f"Updating mask for {tensor_fqn}...")
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
            self.prune_mask(target_sparsity, mask, weights, dense_grads)
            self.grow_mask(sparsity, mask, original_weights, dense_grads)
            self._assert_sparsity_level(mask, sparsity)
            self._assert_structure(mask, tensor_fqn)

    def _assert_sparsity_level(self, mask: torch.Tensor, sparsity_level: float):
        n_ones = mask.sum(dtype=torch.int)
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
        ffi = torch.tensor([m.sum(dtype=torch.int) for m in mask])
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
        s = super().__str__()
        if self.prepared_:
            ffi = []
            for config in self.groups:
                mask = get_mask(**config)
                mask_flat = mask.view(mask.shape[0], prod(mask.shape[1:]))
                this_ffi = mask_flat.sum(dim=1, dtype=torch.int).unique()
                if len(this_ffi) > 1:
                    this_ffi = this_ffi[this_ffi != 0]
                ffi.append(this_ffi.item())
            s += f"FFI: {ffi}\n"
        return s

    def _assert_structure(self, mask, fqn: str) -> None:
        mask_flat = mask.view(mask.shape[0], prod(mask.shape[1:]))
        ffi = torch.count_nonzero(mask_flat, dim=-1)
        if len(ffi) > 1:
            # Remove ablated neurons
            ffi = ffi[ffi != 0]
        if len(ffi.unique()) > 1:
            raise RuntimeError(
                f"FFI Violation found: {ffi.unique()} ffi's in layer {fqn}"
            )


class SRigLTwoFour(SRigL):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.gamma_sal is not None:
            self._logger.warning(
                f"Neuron ablation is not applicable for "
                "SRigL 2:4. Gamma sal was set to "
                f"{self.gamma_sal}"
            )
        self.grower = NMGradientGrower(
            n=2,
            m=4,
            pad=False,
            permute_conv_to_nhwc=False,  # no kernels for conv, don't waste time
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
            sparsity=sparsity, mask=mask, weights=weights
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
        new_mask = self.grower.calculate_mask(
            mask,
            grads=dense_grads,
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

    def _initialize_masks(self) -> None:
        self._distribute_sparsity(self.sparsity)
        if self.random_mask_init:
            pruner = NMRandomPruner(
                n=2, m=4, pad=False, permute_conv_to_nhwc=False
            )
        else:
            pruner = NMMagnitudePruner(
                n=2, m=4, pad=False, permute_conv_to_nhwc=False
            )
        for config in self.groups:
            if config["sparsity"] == 0:
                continue
            # Prune to target sparsity for this step
            weights = getattr(config["module"], config["tensor_name"])
            mask = get_mask(config["module"], config["tensor_name"])
            mask.data = pruner.calculate_mask(mask, weights=weights)
            self._assert_structure(mask, config["tensor_fqn"])

    def _assert_structure(self, mask, fqn: str) -> None:
        if mask.shape[1] % 64 != 0:
            self._logger.warning(
                f"Mask shape is not a multiple of 64, this weight tensor may "
                "not work with torch semi-structured kernels!\n"
                f"Mask shape: {mask.shape} found at {fqn}"
            )
        try:
            mask_2_4 = mask.view(-1, 4)
        except RuntimeError as e:
            self._logger.error(f"fqn: {fqn:}")
            raise e
        ones = torch.count_nonzero(mask_2_4, dim=-1)
        if (ones != 2).all():
            self._logger.warning(
                f"{fqn} mask is not 2:4 pruned! Ones Tensor:\n" f"{ones}"
            )
            # raise RuntimeError(
            #     f"FFI Violation found: {ffi.unique()} ffi's in layer {fqn}"
            # )

    def _assert_sparsity_level(self, mask: torch.Tensor, sparsity_level: float):
        # For 2:4 we should end up with precise counts
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
