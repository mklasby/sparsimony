from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.prune as prune

from sparsimony.schedulers.base import BaseScheduler
from sparsimony.distributions.base import BaseDistribution
from .type_registry import TYPE_REGISTRY


class OptimWrapper(object):

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        pruner: prune.BasePruningMethod,  # todo: dst mix-in?
        scheduler: BaseScheduler,
        distribution: BaseDistribution,
        exclude_tensor_names: Optional[List[str]] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.pruner = pruner
        if exclude_tensor_names is None:
            exclude_tensor_names = []
        self.exclude_tensor_names = exclude_tensor_names
        self.scheduler = scheduler
        self.distribution = distribution
        self._parse_model_params_and_mods()
        self._initialize_pruner()
        self._step = 0

    def step(self) -> None:
        self._step += 1
        self.scheduler.step()
        if self.scheduler._should_update_mask():
            self._update_masks()
        return

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        self.optimizer.add_param_group(param_group)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        for attr_name, attr_obj in self.__dict__.items():
            attr_obj.load_state_dict(state_dict[attr_name])

    def state_dict(self) -> Dict[str, Any]:
        state = {}
        for attr_name, attr_obj in self.__dict__.items():
            if attr_name[0] == "_":
                continue
            if attr_obj is not None and "state_dict" in attr_obj.__dir__():
                state.update({attr_name: attr_obj.state_dict()})
            else:
                state.update({attr_name: attr_obj})
        return state

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none)

    def register_buffer(
        self, name: str, buffer: torch.Tensor, param_group_idx: int = 0
    ) -> None:
        self.optimizer.param_groups[param_group_idx][name] = buffer

    def _parse_model_params_and_mods(self) -> None:
        self.param_tensor_names = {}
        for name, _ in self.model.named_parameters():
            mod_name = ".".join(name.split(".")[:-1])
            tensor_name = name.split(".")[-1]
            if tensor_name in self.exclude_tensor_names:
                continue
            if mod_name in self.param_tensor_names:
                self.param_tensor_names[mod_name].append(tensor_name)
            else:
                self.param_tensor_names[mod_name] = [tensor_name]

    def _initialize_pruner(self) -> None:
        sparsity = self.scheduler(0)
        sparsity_distribution = self.distribution(
            self.model, self.param_tensor_names, sparsity=sparsity
        )
        for mod_name, m in self.model.named_modules():
            if mod_name in self.param_tensor_names:
                for t_name in self.param_tensor_names[mod_name]:
                    self.pruner.apply(
                        m,
                        t_name,
                        amount=sparsity_distribution[mod_name],
                    ),
                    # m.register_buffer(
                    #     name=f"{n}_mask", buffer=getattr(m, f"{n}_mask")
                    # )

    def _update_masks(self) -> None:
        self._remove_masks()
        pass

    def _remove_masks(self) -> None:
        pass
