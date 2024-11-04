import logging
from typing import Tuple, Callable, Dict, Any, List, Generator  # noqa
import functools
from math import prod
from contextlib import contextmanager  # noqa
import math
from functools import wraps
from time import time


import torch
import torch.nn as nn
import torch.nn.functional as F

from sparsimony.parametrization.fake_sparsity import (
    FakeSparsity,
    FakeSparsityDenseGradBuffer,
)

_logger = logging.getLogger(__name__)
global __view_tensors_as_warning_logged
__view_tensors_as_warning_logged = False


def time_fn(f):
    _logger = logging.getLogger(__name__)

    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        _logger.debug(f"func: {f.__name__} took {te-ts:.4f} sec")
        # _logger.debug(
        #     "func:%r args:[%r, %r] took: %2.4f sec"
        #     % (f.__name__, args, kw, te - ts)
        # )
        return result

    return wrap


def get_mask(
    module: nn.Module, tensor_name: str = "weight", param_idx: int = 0, **kwargs
) -> torch.Tensor:
    return get_parametrization(module, tensor_name, param_idx).mask


def cast_mask(
    dtype: torch.dtype,
    module: nn.Module,
    tensor_name: str = "weight",
    param_idx: int = 0,
    **kwargs,
) -> torch.dtype:
    parametrization = get_parametrization(module, tensor_name, param_idx)
    original_dtype = parametrization.mask.dtype
    parametrization.mask = parametrization.mask.to(dtype=dtype)
    return original_dtype


def get_n_ones(sparsity: float, mask: torch.Tensor) -> int:
    return math.floor((1 - sparsity) * mask.numel())


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
    # TODO: Make recursive search more general, cover Tuples too
    for key in kwargs:  # recurse into kwargs looking for tensors
        if isinstance(kwargs[key], dict):
            _, kwargs[key] = transform_args_kwargs_tensors(op, **kwargs[key])
        if isinstance(kwargs[key], list):
            for idx, item in enumerate(kwargs[key]):
                _, kwargs[key][idx] = transform_args_kwargs_tensors(op, **item)
    return args, kwargs


@torch.no_grad()
def view_tensors_as(
    size: Tuple[int, int] | int,
    pad: bool = False,
    padding_dim: int = 1,
    permute_conv_to_nhwc: bool = False,
) -> Callable:
    pad_value = float("nan")
    if not isinstance(size, Tuple):
        size = (1, size)

    def wrapper(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapped_fn(*args, **kwargs) -> torch.Tensor:
            original_size = get_original_tensor_size(*args, **kwargs)
            _permuted = False
            if pad:
                raise NotImplementedError(
                    "Padding is being reimplemented for bool masks. "
                    "Try again soon :)"
                )
                op = functools.partial(
                    pad_tensor_to_tile_view,
                    tile_view=size,
                    value=pad_value,
                    padding_dim=padding_dim,
                )
                args, kwargs = transform_args_kwargs_tensors(
                    op, *args, **kwargs
                )
            if permute_conv_to_nhwc and len(original_size) == 4:
                # conv
                _permuted = True
                op = functools.partial(torch.permute, dims=(0, 2, 3, 1))
                args, kwargs = transform_args_kwargs_tensors(
                    op, *args, **kwargs
                )
                permuted_shape = get_original_tensor_size(*args, **kwargs)
                op = functools.partial(torch.Tensor.reshape, shape=size)
            else:
                op = functools.partial(torch.Tensor.view, size=size)
            args, kwargs = transform_args_kwargs_tensors(op, *args, **kwargs)
            out = fn(*args, **kwargs)
            if _permuted:
                out = out.view(permuted_shape).permute(0, 3, 1, 2).contiguous()
            if pad:
                out = out.view(-1)
                indx = torch.argwhere(~torch.isnan(out))[:, 0]
                out = out[indx]
            try:
                return out.view(original_size)
            except Exception as e:
                global __view_tensors_as_warning_logged
                if not __view_tensors_as_warning_logged:
                    _logger.warning(
                        "Had to reshape output from view_tensor_as. Please "
                        "check if your input tensors to calculate_mask() are "
                        "contiguous. If so please report this on GitHub. "
                        "Exception:"
                    )
                    _logger.warning(e)
                    __view_tensors_as_warning_logged = True
                pass
                return out.reshape(original_size)

        return wrapped_fn

    return wrapper


@torch.no_grad()
def view_tensors_as_neurons(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs) -> torch.Tensor:
        original_size = get_original_tensor_size(*args, **kwargs)
        _permuted = False
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
            _permuted = True
            op = functools.partial(torch.permute, dims=(0, 2, 3, 1))
            args, kwargs = transform_args_kwargs_tensors(op, *args, **kwargs)
            permuted_shape = get_original_tensor_size(*args, **kwargs)
            op = functools.partial(
                torch.Tensor.reshape,  # Must reshape after permutation
                shape=(
                    -1,
                    prod(original_size[1:]),
                ),
            )
        else:
            raise NotImplementedError(
                "Sparsimony currently only support parameterized tensors of dim"
                " 2 or 4"
            )
        # Reshaping
        args, kwargs = transform_args_kwargs_tensors(op, *args, **kwargs)
        if not _permuted:
            return fn(*args, **kwargs).reshape(original_size)
        else:
            return_val = fn(*args, **kwargs)
            return (
                return_val.view(permuted_shape)
                .permute(dims=(0, 3, 1, 2))
                .reshape(original_size)
                .contiguous()
            )

    return wrapped_fn


def calculate_per_tile_n_ones(
    mask: torch.Tensor,
    sparsity: float,
):
    n_ones_total = int(mask.numel() * (1 - sparsity))
    n_ones_per_tile = n_ones_total // mask.shape[0]
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


# TODO: Can use below to convert tensors in context manager vs. decorated func
# TODO: convert return to *tensors, *args, **kwargs
# TODO: Consider using AbstractContextManager to add some helpers to namespace
# def _get_neuron_tile_size(*tensors: torch.Tensor) -> Tuple[int]:
#     t = tensors[0]
#     original_size = t.shape
#     if len(original_size) == 2:
#         return t.view(size=(-1, original_size[-1]))
#     elif len(original_size) == 4:
#         # conv
#         return t.view(size=(-1, prod(original_size[1:])))
#     else:
#         raise NotImplementedError(
#             "Sparsimony currently only support parameterized tensors of dim"
#             " 2 or 4"
#         )


# def _verify_tensors_args(*tensors: torch.Tensor) -> None:
#     shapes = []
#     for t in tensors:
#         shapes.append(t.shape)
#     for shape in shapes:
#         if shape != shape[0]:
#             raise RuntimeError(
#                 "All tensors passed to mask_calculators.calculate_mask must be "  # noqa
#                 "of same shape!"
#             )


# @torch.no_grad()
# @contextmanager
# def view_tensors_as(
#     *tensors: torch.Tensor,
#     tile_size: Tuple[int] | int | str,
#     # pad: bool = False,
#     # padding_dim: int = 1,
#     # permute_conv_to_nhwc: bool = False,
# ) -> Generator[List[torch.Tensor], None, None]:
#     try:
#         _verify_tensors_args(*tensors)
#         _NAMED_TILE_SIZES = {"neuron": _get_neuron_tile_size}
#         if isinstance(tile_size, int):
#             tile_size = (1, tile_size)
#         elif isinstance(tile_size, str):
#             tile_size = _NAMED_TILE_SIZES[tile_size](*tensors)
#         original_shape = tensors[0].shape
#         reshaped_tensors = []
#         for t in tensors:
#             reshaped_tensors.append(t.view(tile_size))
#         yield reshaped_tensors

#     except RuntimeError as e:
#         if "shape" in str(e):
#             msg = (
#                 f"Invalid tile_size ({tile_size}) for tensor shape "
#                 f"({tensors[0].shape}) detected in "
#                 f"{view_tensors_as.__name__}"
#             )
#             _logger.error(msg)
#         raise e

#     finally:
#         for idx, t in enumerate(reshaped_tensors):
#             reshaped_tensors[idx] = t.view(original_shape)
