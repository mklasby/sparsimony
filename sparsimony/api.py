from typing import Optional, List
import torch

from sparsimony.distributions.base import (
    BaseDistribution,
    UniformDistribution,
    ERKDistribution,
    UniformNMDistribution,
)
from sparsimony.schedulers.base import (
    AcceleratedCubicScheduler,
    BaseScheduler,
    ConstantScheduler,
    CosineDecayScheduler,
    AlwaysTrueScheduler,
)
from sparsimony.dst.rigl import RigL
from sparsimony.dst.srigl import SRigL, NMSRigL
from sparsimony.dst.set import SET
from sparsimony.dst.gmp import GMP
from sparsimony.dst.static import StaticMagnitudeSparsifier
from sparsimony.pruners import SRSTESparsifier


def rigl(
    optimizer: torch.optim.Optimizer,
    sparsity: float,
    t_end: int,
    delta_t: int = 100,
    pruning_ratio: float = 0.3,
    global_pruning: bool = False,
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
        global_pruning=global_pruning,
    )


def set(
    optimizer: torch.optim.Optimizer,
    sparsity: float,
    t_end: int,
    delta_t: int = 390,
    pruning_ratio: float = 0.3,
    global_pruning: bool = False,
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
        global_pruning=global_pruning,
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
    global_pruning: bool = False,
):
    """GMP* implementation by Kurtic et al.
    https://proceedings.mlr.press/v234/kurtic24a.html
    
    Documentation question:
    Does the implemented AcceleratedCubicScheduler() class object follows eq. (1) in reference [1] below? 
    [1] M. Zhu and S. Gupta, “To prune, or not to prune: exploring the efficacy of pruning for model compression,” Nov. 13, 2017, arXiv: arXiv:1710.01878. Accessed: Nov. 18, 2024. [Online]. Available: http://arxiv.org/abs/1710.01878

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
        global_pruning=global_pruning,
    )


def static(
    optimizer: torch.optim.Optimizer,
    sparsity: float,
    global_pruning: bool = False,
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
        global_pruning=global_pruning,
    )


def nm_srigl(
    optimizer: torch.optim.Optimizer,
    t_end: int,
    n: int = 2,
    m: int = 4,
    sparsity: None | float = 0.5,
    delta_t: int = 100,
    pruning_ratio: float = 0.3,
    random_mask_init: bool = False,
    excluded_types: str | None | List[str] = "Conv2d",
    excluded_mod_name_regexs: str | None | List[str] = "classifier",
    global_pruning: bool = False,
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
        random_mask_init=random_mask_init,
        global_pruning=global_pruning,
        n=n,
        m=m,
        sparsity=sparsity,
    )


def srigl(
    optimizer: torch.optim.Optimizer,
    sparsity: float,
    gamma_sal: None | float = 0.3,
    no_ablation_last_layer: bool = True,
    t_end: int | None = None,
    delta_t: int = 100,
    pruning_ratio: float = 0.3,
    scheduler: None | BaseScheduler = None,
    distribution: None | BaseDistribution = None,
    grown_weights_init: float = 0.0,
    init_method: Optional[str] = "grad_flow",
    random_mask_init: bool = False,
) -> SRigL:
    """Return SRigL sparsifier.

    Args:
        optimizer (torch.optim.Optimizer): Previously initialized optimizer for
            training. Used to override the dense gradient buffers for
            sparse weights.
        sparsity (float): Sparsity level to prune network to.
        gamma_sal (None | float, optional): Hyperparameter that controls
            dynamic neuron ablation. Defaults to 0.3. See paper for more info.
        no_ablation_last_layer (bool, optional): If True, do not ablate neurons
            last layer. This should only set to False in special circumstances.
            Defaults to True.
        t_end (int | None, optional):  Step to freeze the sparse topology.
            Typically 75% of total training optimizer steps. If None, must pass
            an initialized scheduler. Defaults to None.
        delta_t (int, optional): Steps between topology update. Defaults to 100
        pruning_ratio (float, optional):  Fraction of non-zero elements to prune
            each iteration. Defaults to 0.3.
        scheduler (None | BaseScheduler, optional): Scheduler to use to control
            mask updates. If None, use CosineDecayScheduler. Defaults to None.
        distribution (None | BaseDistribution, optional): Distribution to define
            layer-wise sparsity distribution. If None, ERKDistribution is used.
            Defaults to None.
        grown_weights_init (float, optional): Value to initialize newly grown
            weights to. Defaults to 0.0.
        init_method (Optional[str], optional): Optional reinitialization of
            weights after initializing masks. If None, default torch
            initialization is used. Defaults to "grad_flow".
        random_mask_init (bool, optional): If True, randomly initialize mask
            instead of magnitude pruning. Defaults to False.

    Raises:
        ValueError: If both t_end and scheduler are None.

    Returns:
        SRigL: Initialized SRigL sparsifier.
    """
    if t_end is None and scheduler is None:
        raise ValueError("Must pass t_end or an initialized scheduler")
    if scheduler is None:
        scheduler = (
            CosineDecayScheduler(
                quantity=pruning_ratio,
                t_end=t_end,
                delta_t=delta_t,
            ),
        )
    if distribution is None:
        distribution = ERKDistribution()
    return SRigL(
        scheduler,
        distribution,
        optimizer,
        sparsity=sparsity,
        grown_weights_init=grown_weights_init,
        init_method=init_method,
        random_mask_init=random_mask_init,
        gamma_sal=gamma_sal,
        no_ablation_last_layer=no_ablation_last_layer,
    )


def srste(
    scheduler: None | BaseScheduler = None,
    distribution: None | BaseDistribution = None,
    n: int = 2,
    m: int = 4,
    decay: None | float = 2e-4,
    *args,
    **kwargs,
) -> SRSTESparsifier:
    if scheduler is None:
        scheduler = AlwaysTrueScheduler()
    if distribution is None:
        distribution = UniformNMDistribution(n=n, m=m)
    return SRSTESparsifier(scheduler, distribution, n, m, decay)
