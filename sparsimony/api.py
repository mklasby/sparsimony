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
