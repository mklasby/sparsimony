from typing import List, Optional, Dict, Any
from math import prod

import torch
import torch.nn as nn
from torch.ao.pruning.sparsifier.base_sparsifier import BaseSparsifier

from sparsimony.distributions.base import BaseDistribution
from sparsimony.schedulers.base import BaseScheduler
from sparsimony.parametrization.fake_sparsity import FakeSparsityDenseGradBuffer
from sparsimony.utils import get_mask, get_n_ones, get_parametrization
from sparsimony.dst.base import DSTMixin
from sparsimony.mask_calculators import (
    UnstructuredPruner,
    FFIPruner,
    FFIGrower,
    AblatedTileScorer,
    RandomScorer,
    HierarchicalMaskCalculator,
    SequentialScorer,
    TopKElementScorer,
    MagnitudeScorer,
    NeuronSRigLPruner,
    NMGrower,
    NMPruner,
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
        no_ablation_last_layer: bool = True,
        *args,
        **kwargs,
    ):
        self.scheduler = scheduler
        self.distribution = distribution
        self.sparsity = sparsity
        self.grown_weights_init = grown_weights_init
        self.init_method = init_method
        self.gamma_sal = gamma_sal
        self.no_ablation_last_layer = no_ablation_last_layer
        if defaults is None:
            defaults = dict(parametrization=FakeSparsityDenseGradBuffer)
        super().__init__(
            optimizer=optimizer,
            defaults=defaults,
            random_mask_init=random_mask_init,
            *args,
            **kwargs,
        )
        if self.gamma_sal is not None:  # dynamic ablation
            # TODO: Override scores wipes out our inactive scores
            def agg_fn(scores: List[torch.Tensor]):
                scores = tuple(scores)
                return torch.logical_or(*scores)

            self.pruning_calcs = [
                NeuronSRigLPruner(
                    scorer=SequentialScorer(
                        [
                            TopKElementScorer(MagnitudeScorer),
                            TopKElementScorer(MagnitudeScorer),
                        ],
                        agg_fn=agg_fn,
                    )
                ),
                UnstructuredPruner(MagnitudeScorer),
            ]
        else:
            self.pruner = UnstructuredPruner(MagnitudeScorer)
        # TODO: Refactor to use ablatedTileScorer in hierarchical calc?
        self.grower = FFIGrower(scorer=MagnitudeScorer)

    def prune_mask(
        self,
        sparsity: float,
        mask: torch.Tensor,
        weights: torch.Tensor,
        dense_grads: torch.Tensor,
        gamma_sal: float,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if self.gamma_sal is None:
            return super().prune_mask(
                sparsity, mask, values=weights, *args, **kwargs
            )
        else:  # Dynamic ablation
            sparsities = [sparsity, sparsity]
            # Need K, values for both magnitude scorers
            k = get_n_ones(sparsity, mask)
            calc_kwargs = [
                dict(
                    scorer_kwargs=[
                        dict(k=k, values=weights),
                        dict(k=k, values=dense_grads),
                    ]
                ),
                dict(values=weights),
            ]
            mask.data = HierarchicalMaskCalculator.calculate_mask(
                sparsities, mask, self.pruning_calcs, calc_kwargs
            )
            return mask

    def grow_mask(
        self,
        sparsity: float,
        mask: torch.Tensor,
        original_weights: torch.Tensor,
        dense_grads: torch.Tensor,
    ) -> torch.Tensor:
        if self.gamma_sal is not None:
            score_override = AblatedTileScorer.score(
                mask,
                tile_view="neuron",
            )
        else:
            score_override = None
        return super().grow_mask(
            sparsity,
            mask,
            original_weights,
            score_override=score_override,
            values=dense_grads,
        )

    def _step(self) -> bool:
        _topo_updated = False
        self._step_count += 1
        prune_ratio = self.scheduler(self._step_count)
        if prune_ratio is not None:
            self._logger.info(f"Updating topology at step {self._step_count}")
            self._distribute_sparsity(self.sparsity)
            for idx, config in enumerate(self.groups):
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
                config["gamma_sal"] = self.gamma_sal
                if idx == len(self.groups) - 1 and self.no_ablation_last_layer:
                    config["gamma_sal"] = None
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
            if self.random_mask_init:
                # Randomly prune for step 1
                scorer = RandomScorer
            else:
                # use magnitude pruning criterion
                scorer = MagnitudeScorer
            mask.data = FFIPruner(scorer=scorer).calculate_mask(
                config["sparsity"],
                mask,
                values=weights,
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
        gamma_sal: float,
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
            self.prune_mask(
                target_sparsity, mask, weights, dense_grads, gamma_sal=gamma_sal
            )
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


class NMSRigL(SRigL):
    # TODO: Don't inherit from SRigL

    def __init__(
        self,
        scheduler: BaseScheduler,
        distribution: BaseDistribution,
        optimizer: torch.optim.Optimizer,
        defaults: Optional[Dict[str, Any]] = None,
        sparsity: float | None = None,
        grown_weights_init: float = 0.0,
        init_method: Optional[str] = "grad_flow",
        random_mask_init: bool = False,
        n: int = 2,
        m: int = 4,
        pad: bool = False,
        padding_dim: int = 1,
        permute_conv_to_nhwc: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            scheduler,
            distribution,
            optimizer,
            defaults,
            sparsity,
            grown_weights_init,
            init_method,
            random_mask_init,
            *args,
            **kwargs,
        )
        self.n = n
        self.m = m
        self.pad = pad
        self.padding_dim = padding_dim
        self.permute_conv_to_nhwc = permute_conv_to_nhwc
        if self.gamma_sal is not None:
            self._logger.warning(
                f"Neuron ablation is not applicable for "
                "NMSRigL. Gamma sal was set to "
                f"{self.gamma_sal}. Setting to None"
            )
            self.gamma_sal = None
        if self.sparsity is None:
            self.sparsity = 1 - (self.n / self.m)
        if self.sparsity != 1 - (self.n / self.m):
            self._logger.warning(
                f"Must set sparsity to None or {1-self.n/self.m} for "
                f"{self.n}:{self.m} sparse training. Setting to "
                f" {1-self.n/self.m}"
            )
            self.sparsity = 1 - (self.n / self.m)
        if not self.permute_conv_to_nhwc:
            self._logger.warning(
                "permute_conv_to_nhwc is False. Typically 2:4"
                " kernels for conv require this option set to"
                " true."
            )
        self.pruner = UnstructuredPruner(scorer=MagnitudeScorer)
        self.grower = NMGrower(
            scorer=MagnitudeScorer,
            n=self.n,
            m=self.m,
            pad=self.pad,
            padding_dim=self.padding_dim,
            permute_conv_to_nhwc=self.permute_conv_to_nhwc,
        )

    def _initialize_masks(self) -> None:
        self._distribute_sparsity(self.sparsity)
        if self.random_mask_init:
            scorer = RandomScorer
        else:
            scorer = MagnitudeScorer
        pruner = NMPruner(
            scorer,
            self.n,
            self.m,
            self.pad,
            self.padding_dim,
            self.permute_conv_to_nhwc,
        )
        for config in self.groups:
            if config["sparsity"] == 0:
                continue
            # Prune to target sparsity for this step
            weights = getattr(config["module"], config["tensor_name"])
            mask = get_mask(config["module"], config["tensor_name"])
            mask.data = pruner.calculate_mask(
                self.sparsity, mask, values=weights
            )
            self._assert_structure(mask, config["tensor_fqn"])

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
            self._logger.error(f"fqn: {fqn:}")
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

    def _assert_sparsity_level(
        self, mask: torch.Tensor, sparsity_level: float
    ):  # noqa
        # For N:M we should end up with precise counts
        n_ones = mask.count_nonzero()
        target_n_ones = int(mask.numel() * (1 - sparsity_level))
        # We ignore off-by-one errors as these will be due to floor ops
        if n_ones != target_n_ones and abs(n_ones - target_n_ones) > 1:
            # With very large mask tensors, we may have some precision errors
            # with exact n_ones. Therefore, we simply log the warning instead of # noqa
            # raising.
            # Also naturally occurs in structured pruning
            # TODO: For structured pruning we may wish to calculate
            # actual_n_ones based on network topology
            self._logger.warning(
                f"n_ones actual {n_ones} != n_one target {target_n_ones}"
            )

    def __str__(self) -> str:
        s = DSTMixin.__str__(self)
        if self.prepared_:
            s += f"N:M: {self.n}:{self.m}\n"
        return s
