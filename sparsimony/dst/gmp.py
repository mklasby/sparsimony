from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.ao.pruning.sparsifier.base_sparsifier import BaseSparsifier
from torch.optim.optimizer import Optimizer as Optimizer

from sparsimony.distributions.base import BaseDistribution
from sparsimony.schedulers.base import BaseScheduler
from sparsimony.parametrization.fake_sparsity import FakeSparsity
from sparsimony.utils import get_mask
from sparsimony.dst.base import DSTMixin, GlobalPruningDataHelper
from sparsimony.mask_calculators import (
    UnstructuredPruner,
    MagnitudeScorer,
    NMStructureScorer,
)


class GMP(DSTMixin, BaseSparsifier):
    """GMP* implementation by Kurtic et al.
    https://proceedings.mlr.press/v234/kurtic24a.html
    """

    def __init__(
        self,
        scheduler: BaseScheduler,
        distribution: BaseDistribution,
        optimizer: torch.optim.Optimizer,
        defaults: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        self.scheduler = scheduler
        self.distribution = distribution
        self.sparsity = self.scheduler.initial_sparsity
        if defaults is None:
            defaults = dict(parametrization=FakeSparsity)
        super().__init__(
            optimizer=optimizer, defaults=defaults, *args, **kwargs
        )
        self.pruner = UnstructuredPruner(MagnitudeScorer)

    def grow_mask(
        self,
    ) -> torch.Tensor:
        pass

    def _step(self) -> bool:
        _topo_updated = False
        self._step_count += 1
        sparsity = self.scheduler(self._step_count)
        if sparsity is not None and sparsity != self.scheduler.initial_sparsity:
            self.sparsity = sparsity
            self._logger.info(
                f"Pruning to {self.sparsity*100:.2f}% sparsity at step "
                f"{self._step_count}"
            )
            if self.global_pruning:
                self._global_step()
            else:
                self._distribute_sparsity(sparsity)
                for config in self.groups:
                    if self._is_replica(**config):
                        continue
                    self.update_mask(**config)
            self._broadcast_masks()
            _topo_updated = True
        return _topo_updated

    def _initialize_masks(self) -> None:
        self._distribute_sparsity(self.scheduler.initial_sparsity)
        if self.global_pruning:
            self._global_init_prune()
            return
        for config in self.groups:
            # Prune to target sparsity for this step
            mask = get_mask(config["module"], config["tensor_name"])
            weights = weights = getattr(config["module"], config["tensor_name"])
            mask.data = self.pruner.calculate_mask(
                config["sparsity"], mask, values=weights
            )

    def update_mask(
        self,
        module: nn.Module,
        tensor_name: str,
        sparsity: float,
        **kwargs,
    ):
        mask = get_mask(module, tensor_name)
        if sparsity == 0:
            mask.data = torch.ones_like(mask)
        else:
            weights = getattr(module, tensor_name)
            self.prune_mask(sparsity, mask, values=weights)
            self._assert_sparsity_level(mask, sparsity)

    def _global_step(self) -> None:
        global_data_helper = GlobalPruningDataHelper(
            self.groups, self.global_buffers_cpu_offload
        )
        self.prune_mask(
            self.sparsity,
            global_data_helper.masks,
            values=global_data_helper.sparse_weights,
        )
        self._assert_sparsity_level(global_data_helper.masks, self.sparsity)
        global_data_helper.reshape_and_assign_masks()


class SGMP(GMP):
    def __init__(
        self,
        scheduler: BaseScheduler,
        distribution: BaseDistribution,
        optimizer: Optimizer,
        defaults: Dict[str, Any] | None = None,
        n: int = 2,
        m: int = 4,
        pad: bool = False,
        padding_dim: int = 1,
        permute_conv_to_nhwc: bool = False,
        *args,
        **kwargs,
    ):
        self.n = n
        self.m = m
        self.pad = pad
        self.padding_dim = padding_dim
        self.permute_conv_to_nhwc = permute_conv_to_nhwc
        super().__init__(
            scheduler, distribution, optimizer, defaults, *args, **kwargs
        )
        if self.global_pruning:
            raise ValueError("Cannot use global pruning with SGMP")
        if self.scheduler.final_sparsity > 1 - (self.n / self.m):
            raise ValueError(
                f"Final sparsity of {self.scheduler.final_sparsity} > "
                f"{self.n}/{self.m}. Check scheduler sparsities!"
            )
        if not self.permute_conv_to_nhwc:
            self._logger.warning(
                "permute_conv_to_nhwc is False. Typically 2:4"
                " kernels for conv require this option set to"
                " true."
            )
        self.pruner = UnstructuredPruner(MagnitudeScorer)

    def prune_mask(
        self,
        sparsity: float,
        mask: torch.Tensor,
        values: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        score_override = NMStructureScorer.score(
            MagnitudeScorer,
            mask,
            n=self.n,
            m=self.m,
            score_override=None,
            values=values.view(-1, 1),
        )
        return self.pruner.calculate_mask(
            sparsity, mask, score_override=score_override, values=values
        )

    def _assert_structure(self, mask, fqn: str) -> None:
        if self.n == 2 and self.m == 4:
            if mask.shape[1] % 64 != 0:
                self._logger.warning(
                    f"Mask shape is not a multiple of 64, this weight tensor "
                    "may not work with torch 2:4 semi-structured kernels!\n"
                    f"Mask shape: {mask.shape} found at {fqn}"
                )
        try:
            mask_view = mask.view(-1, self.m)
        except RuntimeError as e:
            self._logger.error(f"fqn: {fqn}")
            raise e
        ones = torch.count_nonzero(mask_view, dim=-1)
        if (ones < self.n).any():
            self._logger.warning(
                f"{fqn} mask has tiles with < {self.n} elements in tile! "
                f"Ones Tensor:\n {ones}"
            )

    def __str__(self):
        s = super().__str__()
        if self.prepared_:
            completed_tiles = []
            total_tiles = []
            for config in self.groups:
                mask = get_mask(**config)
                mask_view = mask.view(-1, self.m)
                completed_tiles.append(
                    (mask_view.sum(dim=-1) == self.n).sum().item()
                )
                total_tiles.append(mask_view.shape[0])
            completed_tile_strs = []
            for complete, total in list(zip(completed_tiles, total_tiles)):
                completed_tile_strs.append(f"{complete}/{total}")
            s += f"Pruned Tiles: {completed_tile_strs}"
        return s
