from typing import List, Optional
from functools import partial
import torch
from torch.ao.pruning.sparsifier.base_sparsifier import BaseSparsifier
from torch.nn.modules import Module

from sparsimony.dst.base import DSTMixin
from sparsimony.distributions.base import BaseDistribution
from sparsimony.schedulers.base import BaseScheduler
from sparsimony.parametrization.fake_sparsity import LOPParametrization
from sparsimony.utils import get_mask, get_parametrization, get_original_tensor
from sparsimony.pruners.unstructured import UnstructuredMagnitudePruner


class LocalOverParametrization(DSTMixin, BaseSparsifier):

    def __init__(
        self,
        scheduler: BaseScheduler,
        distribution: BaseDistribution,
        optimizer: torch.optim.Optimizer,
        sparsity: float = 0.5,
        grown_weights_init: float = 0.0,
        init_method: Optional[str] = "grad_flow",
    ):
        self.scheduler = scheduler
        self.distribution = distribution
        self.sparsity = sparsity
        self.grown_weights_init = grown_weights_init

        self.init_method = init_method
        parametrization = partial(
            LOPParametrization, history_len=scheduler.t_grow
        )
        defaults = dict(parametrization=parametrization)
        super().__init__(optimizer=optimizer, defaults=defaults)

    # @override
    def prepare(self, model, sparse_config):
        super().prepare(model, sparse_config)
        print(self.calculate_global_sparsity())
        for config in self.groups:
            mod = config["module"]
            tensor_name = config["tensor_name"]
            dense_weights = get_original_tensor(mod, tensor_name)
            parametrization = get_parametrization(mod, tensor_name)
            parametrization.register_dense_weights(dense_weights)
            assert (
                parametrization.dense_weights.data_ptr()
                == dense_weights.data_ptr()
            )
            assert (dense_weights * parametrization.mask == mod.weight).all()

    def _step(self) -> None:
        self._step_count += 1
        prune_ratio = self.scheduler(self._step_count)
        if prune_ratio is None:
            if self.scheduler.next_step_update(self._step_count):
                self._accumulate_history()
        else:
            self._distribute_sparsity(self.sparsity)
            if prune_ratio > 0:  # Prune mask by +prune_ratio
                self._prune_masks(prune_ratio)
            elif prune_ratio < 0:  # Grow mask by -prune_ratio
                self._grow_masks()
            self._broadcast_masks()

    def _prune_masks(self, prune_ratio: float):
        print("pruning masks")
        for config in self.groups:
            parametrization = get_parametrization(**config)
            mask = parametrization.mask
            sparse_weights = getattr(config["module"], config["tensor_name"])
            sparse_weight_history = parametrization.sparse_weight_history
            sparse_grad_history = parametrization.sparse_grad_history
            self.prune_mask(
                prune_ratio,
                mask,
                sparse_weights,
                sparse_weight_history,
                sparse_grad_history,
            )
            # reset buffers and turn off accumulation
            parametrization.reset_history(accumulate=False)

    def prune_mask(
        self,
        prune_ratio: float,
        mask: torch.Tensor,
        sparse_weights: torch.Tensor,
        sparse_weight_history: List[torch.Tensor],
        sparse_grad_history: List[torch.Tensor],
    ) -> torch.Tensor:
        print(f"weight history:\n {sparse_weight_history}")
        print(f"grad history:\n {sparse_grad_history}")

    def _grow_masks(self):
        print("growing masks")
        for config in self.groups:
            parametrization = get_parametrization(**config)
            mask = parametrization.mask
            original_weights = getattr(
                config["module"].parametrizations, config["tensor_name"]
            ).original
            sparse_weight_history = parametrization.sparse_weight_history
            sparse_grad_history = parametrization.sparse_grad_history
            self.grow_mask(
                config["sparsity"],
                mask,
                original_weights,
                sparse_weight_history,
                sparse_grad_history,
            )
            # Reset history now and let weights refill buffer
            parametrization.reset_history(accumulate=True)

    def grow_mask(
        self,
        sparsity: float,
        mask: torch.Tensor,
        original_weights: torch.Tensor,
        sparse_weight_history: List[torch.Tensor],
        sparse_grad_history: List[torch.Tensor],
    ) -> torch.Tensor:
        # print(f"weight history:\n {sparse_weight_history}")
        # print(f"grad history:\n {sparse_grad_history}")
        # for i in range(1, len(sparse_weight_history) - 1):
        #     print(
        #         torch.dist(
        #             sparse_weight_history[i], sparse_weight_history[i + 1]
        #         )
        #     )
        pass

    def _initialize_masks(self) -> None:
        self._distribute_sparsity(self.sparsity)
        for config in self.groups:
            # Prune to target sparsity for this step
            mask = get_mask(config["module"], config["tensor_name"])
            weights = getattr(config["module"], config["tensor_name"])
            mask.data = UnstructuredMagnitudePruner.calculate_mask(
                config["sparsity"], mask, weights
            )

    def _accumulate_history(self) -> None:
        for config in self.groups:
            get_parametrization(**config).accumulate = True

    def update_mask(self, module: Module, tensor_name: str, **kwargs):
        return super().update_mask(module, tensor_name, **kwargs)
