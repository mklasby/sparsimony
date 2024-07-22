import torch
import torch.nn as nn

from sparsimony.parametrization.fake_sparsity import (
    FakeSparsity,
    FakeSparsityDenseGradBuffer,
)


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


def share_parametrizations(
    primary: nn.Module,
    replica: nn.Module,
    tensor_name: str = "weight",
    param_idx: int = 0,
):
    __SUPPORTED_PARAMETRIZATIONS = [FakeSparsity, FakeSparsityDenseGradBuffer]
    if not hasattr(primary, "parametrizations") or not hasattr(
        replica, "parametrizations"
    ):
        raise ValueError("Primary and replica modules must be parametrized!")
    primary_para = get_parametrization(primary, tensor_name, param_idx)
    replica_para = get_parametrization(replica, tensor_name, param_idx)
    if type(primary_para) not in __SUPPORTED_PARAMETRIZATIONS:
        raise ValueError(
            f"{type(primary_para)} is not supported for sharing."
            f"Use one of: {__SUPPORTED_PARAMETRIZATIONS}"
        )
    if type(primary_para) is not type(replica_para):
        raise ValueError(
            f"Primary is parametrized with type {type(primary_para)} but "
            f"replica is parametrized with type {type(replica_para)}"
        )
    for name, _ in primary_para.named_buffers():
        setattr(replica_para, name, getattr(primary_para, name))
