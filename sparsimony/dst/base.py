from abc import ABC, abstractmethod
from typing import Dict, Any, List
import copy
import logging
import torch
import torch.nn as nn
from torch.ao.pruning.sparsifier.utils import get_arg_info_from_tensor_fqn

from sparsimony.utils import get_mask, get_original_tensor
from sparsimony.nn_init import sparse_init

_KEYS_NOT_IN_STATE_DICT = ["module", "module_fqn", "tensor_name"]


class DSTMixin(ABC):

    def __init__(self, optimizer: torch.optim.Optimizer, *args, **kwargs):
        self.optimizer = optimizer
        self._step_count = 0
        self._logger = logging.getLogger(__name__)
        super().__init__(*args, **kwargs)

    @abstractmethod
    def prune_mask(
        self, prune_ratio: float, mask: torch.Tensor, *args, **kwargs
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

        return {
            # "state": self.state,
            "groups": groups
        }

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
        if self.init_method is None:
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
            # inactive values should remain as the same values completeness
            try:
                assert unequal_elements[mask == 1].all()
                assert not unequal_elements[mask == 0].any()
            except AssertionError:
                self._logger.warning(
                    "Assertion checks failed on newly initialized values!\n"
                    f"Found {(unequal_elements[mask == 1]==False).sum().item()}"
                    " values that had the same value after reinit and "
                    f"{(unequal_elements[mask==0].any()==True).sum().item()}"
                    f" reinit masked values in layer: {config['tensor_fqn']}."
                )

    def _distribute_sparsity(self, sparsity: float, *args, **kwargs) -> None:
        self.distribution(sparsity, self.groups, *args, **kwargs)

    # @override
    @torch.no_grad
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

    def _assert_sparsity_level(self, mask: torch.Tensor, sparsity_level: float):
        n_ones = mask.sum()
        actual_n_ones = int(mask.numel() * (1 - sparsity_level))
        if n_ones != actual_n_ones:
            raise RuntimeError(f"Found sparsity of {n_ones} != {actual_n_ones}")

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
                if "momentum_buffer" in self.optimizer.state[original_param]:
                    mask = get_mask(**config)
                    self.optimizer.state[original_param][
                        "momentum_buffer"
                    ] *= mask

        self.optimizer.step = _momentum_zero_wrapper

    def __str__(self) -> str:
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
            f"Scheduler: {self.scheduler.__class__.__name__}\n"
            f"Distribution: {self.distribution.__class__.__name__}\n"
            f"Grown weights init to: {self.grown_weights_init}\n"
            "Re-init method for sparse weights during .prepare(): "
            f"{self.init_method}\n"
            f"Step No.: {self._step_count}\n"
            f"Global Sparsity Target: {self.sparsity}\n"
            f"Global Sparsity Actual: {global_sparsity}\n"
            f"Layerwise Sparsity Targets: {layerwise_sparsity_target}\n"
            f"Layerwise Sparsity Actual: {layerwise_sparsity_actual}\n"
            f"Active/Total Neurons: {active_vs_total_neurons}"
        )

    def __repr__(self) -> str:
        return self.__str__()
