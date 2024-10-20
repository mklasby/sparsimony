from abc import ABC, abstractmethod
from typing import Callable, Optional
import math
import logging

import torch

from sparsimony.utils import calculate_per_tile_n_ones, view_tensors_as  # noqa
from .scorers import ABCScorer


class ABCMaskCalculator(ABC):

    # def __init__(self, scorer: ABCScorer, tensor_shape: tuple):
    def __init__(self, scorer: ABCScorer):
        self._logger = logging.getLogger(__name__)
        self.scorer = scorer
        # self.tensor_shape = tensor_shape  # TODO: move to context manager

    @abstractmethod
    def calculate_mask(
        self,
        sparsity: float,
        mask: torch.Tensor,
        score_override: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor: ...

    @abstractmethod
    def _override_scores(
        self,
        scores: torch.Tensor,
        score_overrides: torch.Tensor,
    ) -> torch.Tensor: ...

    @abstractmethod
    def _calculate_mask(
        self,
        n: int,
        sparsity: float,
        mask: torch.Tensor,
        scores: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor: ...

    @abstractmethod
    def _verify_mask_update(self, *args, **kwargs) -> None: ...


class BasePruner(ABCMaskCalculator):
    def calculate_n_drop(self, mask: torch.Tensor, sparsity: float) -> int:
        """Calculates the number of elements to be dropped from a mask
        tensor given a target sparsity.

        Args:
            mask (torch.Tensor): Mask to be applied to weight tensor
            sparsity (float): Target sparsity modification to elements.
                Should be a float between 0 and 1.

        Returns:
            int: The number of elements to be dropped from a mask
                tensor given a target sparsity
        """
        n_drop = math.ceil(mask.sum() - ((1 - sparsity) * mask.numel()))
        return n_drop

    def calculate_mask(
        self,
        sparsity: float,
        mask: torch.Tensor,
        score_override: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        score_override = self.scorer.init_score_override(mask, score_override)
        n_drop = self.calculate_n_drop(mask, sparsity)
        candidate_tiles = self.scorer.candidate_tiles(score_override)
        scores = self.scorer.score(*args, **kwargs)
        scores = self._override_scores(scores, score_override)
        self.scorer.all_reduce_scores(scores)  # TODO: Check what NaNs do
        mask[candidate_tiles] = self._calculate_mask(
            n_drop,
            sparsity,
            mask[candidate_tiles],
            scores[candidate_tiles],
        )
        return mask

    def _override_scores(
        self,
        scores: torch.Tensor,
        score_override: torch.Tensor,
    ):
        scores = self.scorer.override_inactive_scores(scores, score_override)
        return scores


class FineGrainedPruner(BasePruner):

    def _calculate_mask(
        self,
        n: int,
        sparsity: float,
        mask: torch.Tensor,
        scores: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        n_ones_per_tile_target = calculate_per_tile_n_ones(mask, sparsity)
        n_drop_per_tile = torch.tensor(
            [tile.sum().item() - n_ones_per_tile_target for tile in mask],
            dtype=torch.int,
        )
        if not self._verify_mask_update(
            n, n_ones_per_tile_target, n_drop_per_tile
        ):
            return mask
        if len(n_drop_per_tile.unique()) == 1:
            # We can do this in one shot
            k = n_drop_per_tile.unique().item()
            _, indices = torch.topk(scores, k=k, largest=False)
            mask = mask.scatter(
                dim=-1,
                index=indices,
                src=torch.zeros_like(mask),
            )
        else:
            # We drop a potentially unique value per tile
            # TODO: Vmap me bb
            for n_idx, (score, n_drop_this_tile) in enumerate(
                list(zip(scores, n_drop_per_tile.tolist()))
            ):
                if n_drop_this_tile <= 0:
                    continue
                _, indices = torch.topk(
                    score, k=n_drop_this_tile, largest=False
                )
                mask[n_idx] = mask[n_idx].scatter(
                    dim=-1,
                    index=indices,
                    src=torch.zeros_like(mask[n_idx]),
                )
        return mask

    def _verify_mask_update(
        self, n: int, n_ones_per_tile_target: int, n_drop_per_tile: int
    ) -> None:
        if n_ones_per_tile_target == 0:
            self._logger.warning(
                "Found a target nnz per tile of 0! All candidate tiles will be "
                "fully pruned by this pruner."
            )
        if not (n_drop_per_tile >= 0).all():
            self._logger.warning(
                f"n_drop_per_tile < 0 ({n_drop_per_tile}). Will skip tiles with"
                " nnz elements < n_drop"
            )
        if not (n_drop_per_tile.sum() >= n):
            self._logger.warning(
                "(n_drop_per_tile.sum() >= n_drop): "
                f"({n_drop_per_tile.sum()} >= {n}) "
                "Check sparsity level!"
            )
        if len(n_drop_per_tile.unique()) == 1:
            k = n_drop_per_tile.unique().item()
            if k < 0:
                self._logger.warning(
                    f"n_drop_per_tile == {k} for all tiles, skipping this "
                    "pruner."
                )
                return False
        return True


class StructuredPruner(BasePruner):
    def _verify_mask_update(self, n_tiles_to_drop: int) -> bool:
        if n_tiles_to_drop < 0:
            self._logger.warning(
                f"n_tiles_to_drop is <0 ({n_tiles_to_drop}), continuing..."
            )
            return False
        if n_tiles_to_drop == 0:
            self._logger.warning(
                f"n_tiles_to_drop == 0 ({n_tiles_to_drop}), continuing..."
            )
            return False
        return True

    def _calculate_mask(
        self,
        n: int,
        sparsity: float,
        mask: torch.Tensor,
        scores: torch.Tensor,
        aggregate_norm_ord: str | int = 2,
        agg_fn: Callable | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        n_tiles_to_drop = math.ceil(n / mask.shape[-1])
        if not self._verify_mask_update(n_tiles_to_drop):
            return mask
        if agg_fn is None:
            scores = torch.linalg.norm(scores, ord=aggregate_norm_ord, dim=1)
        else:
            scores = agg_fn(scores)
        _, indices = torch.topk(scores, k=n_tiles_to_drop, largest=False)
        # zero out all elements in tile for structured pruning
        mask[indices] = 0
        return mask


class BaseGrower(ABCMaskCalculator):

    def calculate_mask(
        self,
        sparsity: float,
        mask: torch.Tensor,
        score_override: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        score_override = self.scorer.init_score_override(mask, score_override)
        n_grow = self._calculate_n_grow(mask, sparsity)
        candidate_tiles = self.scorer.candidate_tiles(score_override)
        scores = self.scorer.score(*args, **kwargs)
        scores = self._override_scores(scores, score_override)
        self.scorer.all_reduce_scores(scores)  # TODO: Check what NaNs do
        mask[candidate_tiles] = self._calculate_mask(
            n_grow,
            sparsity,
            mask[candidate_tiles],
            scores[candidate_tiles],
        )
        return mask

    def _calculate_n_grow(cls, mask: torch.Tensor, sparsity: float) -> int:
        # target_nnz - current nnz
        n_grow = int(int(mask.numel() * (1 - sparsity)) - mask.sum())
        if n_grow < 0:
            current_n_ones = int(mask.sum().item())
            raise RuntimeError(
                f"Current sparsity > target in grow mask! Current n_ones "
                f"{current_n_ones} vs. Target n_ones "
                f"{int(mask.numel() * (1 - sparsity))}"
            )
        return n_grow

    def _override_scores(
        self,
        scores: torch.Tensor,
        score_override: torch.Tensor,
    ):
        scores = self.scorer.override_active_scores(scores, score_override)
        return scores


class FineGrainedGrower(BaseGrower):

    def _calculate_mask(
        self,
        n: int,
        sparsity: float,
        mask: torch.Tensor,
        scores: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        n_ones_per_tile_target = calculate_per_tile_n_ones(mask, sparsity)
        n_grow_per_tile = n_ones_per_tile_target - mask.sum(dim=-1)
        if not self._verify_mask_update(
            n, n_ones_per_tile_target, n_grow_per_tile
        ):
            return mask

        if len(n_grow_per_tile.unique()) == 1:  # every tile grows by same k
            k = n_grow_per_tile.unique().item()
            _, indices = torch.topk(scores, k=k, largest=True)
            mask = mask.scatter(
                dim=-1,
                index=indices,
                src=torch.ones_like(mask),
            )
        else:  # per tile growth
            # TODO: Vmap me
            for n_idx, (score, n_grow_this_tile) in enumerate(
                list(zip(scores, n_grow_per_tile.tolist()))
            ):
                if n_grow_this_tile <= 0:
                    continue
                if n_grow_this_tile > len(score):
                    self._logger.warning(
                        "n_grow_this_tile > len(score): "
                        f"{n_grow_this_tile} > {len(score)}"
                    )
                    n_grow_this_tile = len(score)
                _, indices = torch.topk(score, k=n_grow_this_tile, largest=True)
                mask[n_idx] = mask[n_idx].scatter(
                    dim=-1,
                    index=indices,
                    src=torch.ones_like(mask[n_idx]),
                )
        return mask

    def _verify_mask_update(
        self,
        n_grow: int,
        n_ones_per_tile_target: int,
        n_grow_per_tile,
        *args,
        **kwargs,
    ) -> bool:
        if n_ones_per_tile_target == 0:
            self._logger.warning(
                "Found a target nnz per tile of 0! All candidate tiles will be "
                "left fully pruned by this grower."
            )
        if not (n_grow_per_tile >= 0).all():
            self._logger.warning(
                f"n_grow_per_tile < 0 ({n_grow_per_tile}). Will skip tiles with"
                " nz elements > n_grow"
            )
        if not (n_grow_per_tile.sum(dtype=torch.int) <= n_grow):
            self._logger.warning(
                "(n_grow_per_tile.sum() <= n_grow): "
                f"({n_grow_per_tile.sum(dtype=torch.int)} <= {n_grow}) "
                "Check sparsity level and/or padding"
            )
        if len(n_grow_per_tile.unique()) == 1:
            k = n_grow_per_tile.unique().item()
            if k <= 0:
                self._logger.warning(
                    f"n_grow_per_tile == {k} for all tiles, skipping this "
                    "grower."
                )
                return False
        return True


class StructuredGrower(BaseGrower):

    def _calculate_mask(
        self,
        n: int,
        mask: torch.Tensor,
        scores: torch.Tensor,
        aggregate_norm_ord: str | int = 2,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        n_tiles_to_grow = n // mask.shape[-1]
        if not self._verify_mask_update(n_tiles_to_grow):
            return mask
        scores = torch.linalg.norm(scores, ord=aggregate_norm_ord, dim=1)
        _, indices = torch.topk(scores, k=n_tiles_to_grow, largest=True)
        mask[indices] = 1
        return mask

    def _verify_mask_update(
        self, n_tiles_to_grow: int, *args, **kwargs
    ) -> bool:
        if n_tiles_to_grow <= 0:
            self._logger.warning(
                f"n_tiles_to_grow is <=0 ({n_tiles_to_grow}), continuing..."
            )
            return False
        return True


class RandomPruner(BasePruner):
    """Pruning method that randomly prunes tensor."""

    @classmethod
    def get_scores(
        cls,
        mask: torch.Tensor,
        candidate_tiles: torch.Tensor,
        *args,
        **kwargs,
    ):
        return torch.where(
            torch.logical_and(mask == 1, mask != cls._SCORE_FILL_VALUE),
            torch.abs(torch.rand_like(mask)) + cls._EPS,
            torch.full_like(mask, cls._SCORE_FILL_VALUE),
        )
