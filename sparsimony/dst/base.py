import collections
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import copy
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.ao.pruning.sparsifier.utils import get_arg_info_from_tensor_fqn

from sparsimony.utils import get_mask, get_original_tensor, get_parametrization
from sparsimony.nn_init import sparse_init
from sparsimony.mask_calculators import UnstructuredRandomPruner

_KEYS_NOT_IN_STATE_DICT = ["module", "module_fqn", "tensor_name"]


class DSTMixin(ABC):
    # TODO: Consider extracting additional base class for pruners / one shot
    _OPTIM_REG = {
        optim.SGD: ["momentum_buffer"],
        optim.AdamW: ["exp_avg", "exp_avg_sq"],
        optim.Adam: ["exp_avg", "exp_avg_sq"],
    }

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        random_mask_init: bool = True,
        global_pruning: bool = False,
        *args,
        **kwargs,
    ):
        """Mixin class to extend BaseSparisifer for DST algorithms

        Args:
            optimizer (torch.optim.Optimizer): Optimizer registered to model
                params
            random_mask_init (bool, optional): If True, randomly prune mask at
                initialization. Otherwise, use pruning criteria. Defaults to
                True.
        """
        if type(optimizer) not in self._OPTIM_REG:
            raise NotImplementedError(
                f"DSTMixin does not support optimizer type: {type(optimizer)}"
            )
        self.optimizer = optimizer
        self.random_mask_init = random_mask_init
        self.global_pruning = global_pruning
        self._step_count = 0
        self._logger = logging.getLogger(__name__)
        self.prepared_ = False
        super().__init__(*args, **kwargs)

    @abstractmethod
    def prune_mask(
        self, sparsity: float, mask: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor: ...

    @abstractmethod
    def grow_mask(
        self,
        sparsity: float,
        mask: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor: ...

    @abstractmethod
    def _step(self) -> bool: ...

    @abstractmethod
    def _initialize_masks(self) -> None: ...

    # @ override
    def state_dict(self) -> Dict[str, Any]:
        r"""Returns the state of the optimizer as a :class:`dict`.

        It contains:
        * groups - a list containing all sparsity configuration groups
            with the key 'tensor_fqn' specifying the path to the sparsified
            tensor within a model

        """

        groups: List[Dict[str, Any]] = [
            dict(
                filter(
                    lambda key_value: key_value[0]
                    not in _KEYS_NOT_IN_STATE_DICT,
                    mg.items(),
                )
            )
            for mg in self.groups
        ]

        return {"groups": groups}

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        groups = copy.deepcopy(state_dict["groups"])
        for config in groups:
            tensor_fqn = config["tensor_fqn"]
            arg_info = get_arg_info_from_tensor_fqn(self.model, tensor_fqn)
            module = arg_info["module"]
            if strict and module is None:
                raise RuntimeError(f"Error loading {tensor_fqn} into the model")
            config.update(arg_info)
        self.__setstate__({"groups": groups})

    def _broadcast_tensor(self, t: torch.Tensor, from_rank: int = 0) -> None:
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(t, from_rank)

    def adjust_init_for_sparsity(self) -> None:
        if not hasattr(self, "init_method") or self.init_method is None:
            return

        for config in self.groups:
            if config["sparsity"] == 0:
                continue
            mask = get_mask(**config)
            original_tensor = get_original_tensor(**config)
            orig = original_tensor.clone().detach()

            sparse_init(
                self.init_method, tensor=original_tensor, sparsity_mask=mask
            )
            self._broadcast_tensor(original_tensor)
            unequal_elements = torch.where(
                orig != original_tensor,
                torch.ones_like(mask, dtype=torch.bool),
                torch.zeros_like(mask, dtype=torch.bool),
            )
            # None of the reinit values should be the same
            # inactive values should remain as the same values
            try:
                assert unequal_elements[mask == 1].all()
                assert not unequal_elements[mask == 0].any()
            except AssertionError:
                self._logger.debug(
                    "Assertion checks failed on newly initialized values!\n"
                    f"Found {(unequal_elements[mask == 1]==False).sum().item()}"
                    " values that had the same value after reinit and "
                    f"{(unequal_elements[mask==0].any()==True).sum().item()}"
                    f" reinit masked values in layer: {config['tensor_fqn']}."
                )

    def _distribute_sparsity(self, sparsity: float, *args, **kwargs) -> None:
        self.distribution(sparsity, self.groups, *args, **kwargs)

    # @override
    @torch.no_grad()
    def prepare(
        self,
        model: nn.Module,
        sparse_config: Dict[str, Any],
    ):
        super().prepare(model, sparse_config)
        self._initialize_masks()
        self._broadcast_masks()
        self.adjust_init_for_sparsity()
        self.zero_inactive_param_momentum_buffers()
        self.prepared_ = True

    def _broadcast_masks(self) -> None:
        for config in self.groups:
            mask = get_mask(**config)
            self._broadcast_tensor(mask)

    def calculate_mask_sparsity(self, mask: torch.Tensor):
        return 1 - (mask.sum() / mask.numel())

    def calculate_global_sparsity(self):
        total_el = 0
        nnz_el = 0
        for config in self.groups:
            mask = get_mask(config["module"], config["tensor_name"])
            total_el += mask.numel()
            nnz_el += mask.sum()
        return 1 - (nnz_el / total_el)

    @classmethod
    def get_prune_ratio_from_sparsity(
        cls, mask: torch.Tensor, sparsity: float
    ) -> torch.Tensor:
        current_sparsity = (mask == 0).sum() / mask.numel()
        return round(
            ((sparsity - current_sparsity) / (1 - current_sparsity)).item(), 6
        )

    @classmethod
    def get_sparsity_from_prune_ratio(
        cls, mask: torch.Tensor, prune_ratio: float
    ) -> torch.Tensor:
        current_sparsity = (mask == 0).sum() / mask.numel()
        return round(
            (prune_ratio * (1 - current_sparsity) + current_sparsity).item(), 6
        )

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
                f"n_ones actual {n_ones} != n_one target {actual_n_ones}"
            )

    # @override
    def step(self) -> bool:
        if not self.enable_mask_update:
            return False
        with torch.no_grad():
            return self._step()

    def zero_inactive_param_momentum_buffers(self) -> None:

        _unwrapped_step = self.optimizer.step

        def _momentum_zero_wrapper():
            _unwrapped_step()
            for config in self.groups:
                if config["sparsity"] == 0:
                    continue
                original_param = get_original_tensor(**config)
                state_kw_list = self._OPTIM_REG[type(self.optimizer)]
                mask = get_mask(**config)
                for state_kw in state_kw_list:
                    if state_kw in self.optimizer.state[original_param]:
                        self.optimizer.state[original_param][state_kw] *= mask

        self.optimizer.step = _momentum_zero_wrapper

    def get_layerwise_sparsity(self) -> Dict[str, float]:
        layerwise_sparsity_actual = collections.defaultdict(float)
        for i, config in enumerate(self.groups):
            mask = get_mask(**config)
            layerwise_sparsity_actual[config["tensor_fqn"] + "_sparsity"] = (
                self.calculate_mask_sparsity(mask).item()
            )
        return layerwise_sparsity_actual

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
                active_neurons.append(
                    torch.vmap(neuron_is_active)(mask).sum().item()
                )
                total_neurons.append(len(mask))
            active_vs_total_neurons = []
            for a, t in list(zip(active_neurons, total_neurons)):
                active_vs_total_neurons.append(f"{a}/{t}")
            # TODO: Should list ignored_layers from distribution
            grown_weights_init = getattr(
                self, "grown_weights_init", "Disabled, no regrowth"
            )
            init_method = getattr(self, "init_method", "Disabled, no re-init")

        else:
            err_message = (
                "Error: Sparsifier's prepare() method must be called first."
            )
            global_sparsity = err_message
            layerwise_sparsity_actual = err_message
            layerwise_sparsity_target = err_message
            active_vs_total_neurons = err_message
        return (
            f"{self.__class__.__name__}\n"
            f"Scheduler: {self.scheduler.__class__.__name__}\n"
            f"Distribution: {self.distribution.__class__.__name__}\n"
            f"Grown weights init to: {grown_weights_init}\n"
            "Re-init method for sparse weights during .prepare(): "
            f"{init_method}\n"
            f"Step No.: {self._step_count}\n"
            f"Global Sparsity Target: {self.sparsity}\n"
            f"Global Sparsity Actual: {global_sparsity}\n"
            f"Layerwise Sparsity Targets: {layerwise_sparsity_target}\n"
            f"Layerwise Sparsity Actual: {layerwise_sparsity_actual}\n"
            f"Active/Total Neurons: {active_vs_total_neurons}\n"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def _is_replica(
        self, module: nn.Module, tensor_name: str, *args, **kwargs
    ) -> bool:
        parametrization = get_parametrization(module, tensor_name)
        if (
            hasattr(parametrization, "is_replica_")
            and parametrization.is_replica_
        ):
            return True
        return False

    # TODO: Move to global pruner Mixin?
    def _global_step(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "self.global_prune is True but _global_step has not been "
            f"implemented for {self.__class__.__name__}."
        )

    def _global_init_prune(self) -> None:
        global_data_helper = GlobalPruningDataHelper(self.groups)
        if self.random_mask_init:
            global_data_helper.masks.data = (
                UnstructuredRandomPruner.calculate_mask(
                    self.sparsity, global_data_helper.masks
                )
            )
        else:
            # use pruning criterion
            self.prune_mask(
                self.sparsity,
                global_data_helper.masks,
                global_data_helper.sparse_weights,
            )
        self._assert_sparsity_level(global_data_helper.masks, self.sparsity)
        global_data_helper.reshape_and_assign_masks()


class GlobalPruningDataHelper:

    def __init__(self, groups: List[Dict[str, Any]]):
        self.groups = groups
        original_weights = []
        sparse_weights = []
        original_shapes = []
        original_numels = []
        masks = []
        for config in self.groups:
            module = config["module"]
            tensor_name = config["tensor_name"]
            mask = get_mask(module, tensor_name)
            original_shapes.append(mask.shape)
            original_numels.append(mask.numel())
            masks.append(mask.flatten())
            original_weights.append(
                get_original_tensor(module, tensor_name).flatten()
            )
            sparse_weights.append(getattr(module, tensor_name).flatten())
        self.original_weights = torch.concat(original_weights)
        self.sparse_weights = torch.concat(sparse_weights)
        self.masks = torch.concat(masks)
        self.original_shapes = original_shapes
        self.original_numels = original_numels

    def reshape_and_assign_masks(
        self,
    ) -> None:
        for idx, config in enumerate(self.groups):
            module = config["module"]
            tensor_name = config["tensor_name"]
            mask = get_mask(module, tensor_name)
            stride_start = sum(self.original_numels[:idx])
            stride_end = sum(self.original_numels[: idx + 1])
            shape = self.original_shapes[idx]
            mask.data = self.masks[stride_start:stride_end].reshape(shape)
