from typing import Optional, List
import torch

from sparsimony.distributions.base import (
    BaseDistribution,
    UniformDistribution,
    ERKDistribution,
)
from sparsimony.schedulers.base import (
    AcceleratedCubicScheduler,
    ConstantScheduler,
    CosineDecayScheduler,
)
from sparsimony.dst.rigl import RigL
from sparsimony.dst.srigl import SRigL, NMSRigL  # noqa
from sparsimony.dst.set import SET
from sparsimony.dst.gmp import GMP
from sparsimony.dst.static import StaticMagnitudeSparsifier


def rigl(
    optimizer: torch.optim.Optimizer,
    sparsity: float,
    t_end: int,
    delta_t: int = 100,
    pruning_ratio: float = 0.3,
) -> RigL:
    """Return RigL sparsifier.

    Args:
        optimizer (torch.optim.Optimizer): Previously initialized optimizer for
            training. Used to override the dense gradient buffers for
            sparse weights.
        sparsity (float): Sparsity level to prune network to.
        t_end (int): Step to freeze the sparse topology. Typically 75% of total
            training optimizer steps.
        delta_t (int, optional): Steps between topology update. Defaults to 100.
        pruning_ratio (float, optional): Fraction of nnz elements to prune each
            iteration. Defaults to 0.3.

    Returns:
        RigL: Initialized rigl sparsifier.
    """
    return RigL(
        scheduler=CosineDecayScheduler(
            quantity=pruning_ratio,
            t_end=t_end,
            delta_t=delta_t,
        ),
        distribution=ERKDistribution(),
        optimizer=optimizer,
        sparsity=sparsity,
    )


def set(
    optimizer: torch.optim.Optimizer,
    sparsity: float,
    t_end: int,
    delta_t: int = 390,
    pruning_ratio: float = 0.3,
) -> SET:
    """Return SET sparsifier.

    Args:
        optimizer (torch.optim.Optimizer): Previously initialized optimizer for
            training. Used to override the dense gradient buffers for
            sparse weights.
        sparsity (float): Sparsity level to prune network to.
        t_end (int): Step to freeze the sparse topology. Typically 75% of total
            training optimizer steps.
        delta_t (int, optional): Steps between topology update. Defaults to 100.
        pruning_ratio (float, optional): Fraction of nnz elements to prune each
            iteration. Defaults to 0.3.

    Returns:
        SET: Initialized SET sparsifier.
    """
    return SET(
        scheduler=ConstantScheduler(
            quantity=pruning_ratio,
            t_end=t_end,
            delta_t=delta_t,
        ),
        distribution=UniformDistribution(),
        optimizer=optimizer,
        sparsity=sparsity,
    )


def gmp(
    optimizer: torch.optim.Optimizer,
    t_accel: int,
    t_end: int,
    distribution: Optional[BaseDistribution] = None,
    delta_t: int = 100,
    initial_sparsity: float = 0.0,
    accelerated_sparsity: float = 0.7,
    final_sparsity: float = 0.9,
):
    """GMP* implementation by Kurtic et al.
    https://proceedings.mlr.press/v234/kurtic24a.html

    Args:
        optimizer (torch.optim.Optimizer): Previously initialized optimizer for
            training. Used to override the dense gradient buffers for
            sparse weights.
        t_accel (int): Step to jump to accelerated sparsity level
        t_end (int): Step to stop pruning model
        distribution (Optional[BaseDistribution], optional): Layerwise sparsity
            distribution. If None, uses uniform. Defaults to None.
        delta_t (int, optional): Steps between topology update. Defaults to 100.
        initial_sparsity (float, optional): Defaults to 0.0.
        accelerated_sparsity (float, optional): Sparsity to jump to at t_accel
            step. Defaults to 0.7.
        final_sparsity (float, optional): Final sparsity. Defaults to 0.9.

    Returns:
        GMP: GMP sparsifier
    """
    if distribution is None:
        distribution = UniformDistribution()
    return GMP(
        scheduler=AcceleratedCubicScheduler(
            t_end=t_end,
            delta_t=delta_t,
            t_accel=t_accel,
            initial_sparsity=initial_sparsity,
            accelerated_sparsity=accelerated_sparsity,
            final_sparsity=final_sparsity,
        ),
        distribution=distribution,
        optimizer=optimizer,
    )


def static(
    optimizer: torch.optim.Optimizer,
    sparsity: float,
) -> StaticMagnitudeSparsifier:
    """Return StaticMagnitude sparsifier.

    Args:
        optimizer (torch.optim.Optimizer): Previously initialized optimizer for
            training. Used to override the dense gradient buffers for
            sparse weights.
        sparsity (float): Sparsity level to prune network to.

    Returns:
        StaticMagnitudeSparsifier: Initialized StaticMagnitude sparsifier.
    """
    return StaticMagnitudeSparsifier(
        optimizer=optimizer,
        distribution=UniformDistribution(),
        sparsity=sparsity,
    )


def srigl_two_four(
    optimizer: torch.optim.Optimizer,
    t_end: int,
    delta_t: int = 100,
    pruning_ratio: float = 0.3,
    random_mask_init: bool = False,
    excluded_types: str | None | List[str] = "Conv2d",
    excluded_mod_name_regexs: str | None | List[str] = "classifier",
) -> RigL:
    """Return NMSRigL sparsifier.

    Args:
        optimizer (torch.optim.Optimizer): Previously initialized optimizer for
            training. Used to override the dense gradient buffers for
            sparse weights.
        sparsity (float): Sparsity level to prune network to.
        t_end (int): Step to freeze the sparse topology. Typically 75% of total
            training optimizer steps.
        delta_t (int, optional): Steps between topology update. Defaults to 100.
        pruning_ratio (float, optional): Fraction of nnz elements to prune each
            iteration. Defaults to 0.3.
        excluded_types Optional[Union[str, List[str]]]: String component of
            types to exclude. Defaults to Conv2d.
        excluded_mod_name_regex Optional[Union[str, List[str]]]: FQN module
            names to exclude.
        random_mask_init (bool, optional): If False, use magnitude pruning to
            initialize the mask. If True mask is randomly pruned. Defaults to
            False.

    Returns:
        RigL: Initialized rigl sparsifier.
    """
    return NMSRigL(
        scheduler=CosineDecayScheduler(
            quantity=pruning_ratio,
            t_end=t_end,
            delta_t=delta_t,
        ),
        distribution=UniformDistribution(
            excluded_types=excluded_types,
            excluded_mod_name_regexs=excluded_mod_name_regexs,
        ),
        optimizer=optimizer,
    )
