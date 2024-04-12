import torch
import torch.nn as nn


def get_mask(
    module: nn.Module, tensor_name: str = "weight", param_idx: int = 0, **kwargs
) -> torch.Tensor:
    return get_parametrization(module, tensor_name, param_idx).mask


def get_parametrization(
    module: nn.Module,
    tensor_name: str = "weight",
    param_idx: int = 0,
    **kwargs,
) -> torch.Tensor:
    return getattr(module.parametrizations, tensor_name)[param_idx]


def get_original_tensor(
    module: nn.Module,
    tensor_name: str = "weight",
    **kwargs,
) -> torch.Tensor:
    return getattr(module.parametrizations, tensor_name).original


def path_length(history: torch.Tensor):
    path_length = 0
    for i in range(len(history) - 1):
        path_length += torch.abs(history[i] - history[i + 1])
    return path_length
