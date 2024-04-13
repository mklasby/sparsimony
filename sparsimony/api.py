import torch

from sparsimony.distributions.base import (
    UniformDistribution,
    ERKDistribution,
)
from sparsimony.schedulers.base import (
    ConstantScheduler,
    CosineDecayScheduler,
)
from sparsimony.dst.rigl import RigL
from sparsimony.dst.set import SET


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
            pruning_ratio=pruning_ratio,
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
            pruning_ratio=pruning_ratio,
            t_end=t_end,
            delta_t=delta_t,
        ),
        distribution=UniformDistribution(),
        optimizer=optimizer,
        sparsity=sparsity,
    )
