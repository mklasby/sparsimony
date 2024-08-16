from typing import List, Tuple, Callable, Dict, Any
import functools
from math import prod

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    if primary is replica:
        return  # no op
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
    if hasattr(primary_para, "is_replica_") and primary_para.is_replica_:
        raise ValueError(
            "Primary module parametrization has already been "
            "registered as a replica!"
        )
    for name, _ in primary_para.named_buffers():
        setattr(replica_para, name, getattr(primary_para, name))
    replica_para.is_replica_ = True  # state to track if this mod
    if hasattr(primary_para, "replicas_"):
        primary_para.replicas_.append(replica_para)
    else:
        primary_para.replicas_ = [replica_para]


def get_original_tensor_size(*args, **kwargs) -> Tuple[int]:
    original_sizes = [a.shape for a in args if isinstance(a, torch.Tensor)]
    original_sizes.extend(
        [v.shape for _, v in kwargs.items() if isinstance(v, torch.Tensor)]
    )
    for sizes in original_sizes:
        if sizes != original_sizes[0]:
            raise RuntimeError(
                "All tensors passed to mask calculators must be of same shape!"
            )
    if len(original_sizes) == 0:
        raise RuntimeError(
            "No tensors found in args/kwargs passed to mask calculator!"
        )
    return original_sizes[0]


def transform_args_kwargs_tensors(
    op: Callable, *args, **kwargs
) -> Tuple[Tuple[Any], Dict[str, Any]]:
    args = [op(a) if isinstance(a, torch.Tensor) else a for a in args]
    kwargs = {
        k: op(v) if isinstance(v, torch.Tensor) else v
        for k, v in kwargs.items()
    }
    return args, kwargs


@torch.no_grad()
def view_tensors_as(
    size: Tuple[int],
    pad: bool = False,
    padding_dim: int = 1,
) -> Callable:
    pad_value = float("nan")
    if not isinstance(size, Tuple):
        size = (1, size)

    def wrapper(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapped_fn(*args, **kwargs) -> torch.Tensor:
            original_size = get_original_tensor_size(*args, **kwargs)
            if pad:
                op = functools.partial(
                    pad_tensor_to_tile_view,
                    tile_view=size,
                    value=pad_value,
                    padding_dim=padding_dim,
                )
                args, kwargs = transform_args_kwargs_tensors(
                    op, *args, **kwargs
                )
            op = functools.partial(torch.Tensor.view, size=size)
            args, kwargs = transform_args_kwargs_tensors(op, *args, **kwargs)
            out = fn(*args, **kwargs)
            if pad:
                out = out.view(-1)
                indx = torch.argwhere(~torch.isnan(out))[:, 0]
                out = out[indx]
            return out.reshape(original_size)

        return wrapped_fn

    return wrapper


@torch.no_grad()
def view_tensors_as_neurons(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs) -> torch.Tensor:
        original_size = get_original_tensor_size(*args, **kwargs)
        if len(original_size) == 2:
            # linear and friends
            op = functools.partial(
                torch.Tensor.view,
                size=(
                    -1,
                    original_size[-1],
                ),
            )
        elif len(original_size) == 4:
            # conv
            op = functools.partial(
                torch.Tensor.view,
                size=(
                    -1,
                    prod(original_size[1:]),
                ),
            )
        else:
            raise NotImplementedError(
                "Sparsimony currently only support parameterized tensors of dim"
                " 2 or 4"
            )
        args, kwargs = transform_args_kwargs_tensors(op, *args, **kwargs)
        return fn(*args, **kwargs).reshape(original_size)

    return wrapped_fn


def calculate_per_tile_n_ones(
    mask: torch.Tensor,
    sparsity: float,
    candidate_tiles: List[int] | None = None,
):
    if candidate_tiles is not None:
        mask_slice = mask[candidate_tiles]
    else:
        mask_slice = mask
    n_ones_total = int(mask.numel() * (1 - sparsity))
    n_ones_per_tile = n_ones_total // mask_slice.shape[0]
    return n_ones_per_tile


def view_tensor_as_neuron(t: torch.Tensor):
    original_size = t.shape
    if len(original_size) == 2:
        return t.view(size=(-1, original_size[-1]))
    elif len(original_size) == 4:
        # conv
        return t.view(size=(-1, prod(original_size[1:])))
    else:
        raise NotImplementedError(
            "Sparsimony currently only support parameterized tensors of dim"
            " 2 or 4"
        )


def pad_tensor_to_tile_view(
    t: torch.Tensor,
    tile_view: Tuple[int],
    value: float = float("nan"),
    padding_dim: int = 1,
) -> torch.Tensor:
    if padding_dim == 1:
        # pad flattened tensor
        t = t.view(-1)
    elif padding_dim == 2:
        # pad as neurons
        t = view_tensor_as_neuron(t)
    else:
        raise NotImplementedError(
            "Padding is only implemented for padding_dim in [1,2]"
        )
    if t.shape[-1] % tile_view[-1] == 0:
        # no op, padding not required
        return t
    else:
        pad = (
            0,
            tile_view[-1] - (t.shape[-1] + tile_view[-1]) % tile_view[-1],
        )
    out = F.pad(t, pad, "constant", value)
    return out
