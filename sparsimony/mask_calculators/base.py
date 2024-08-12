from abc import ABC, abstractmethod
from typing import Optional
import math
import logging

import torch
import torch.distributed as dist

from sparsimony.utils import calculate_per_tile_n_ones


class BaseMaskCalculator(ABC):
    _OVERRIDE_SENTINEL_VALUE: float = float("-inf")
    _EPS = 0.001

    @classmethod
    @abstractmethod
    def calculate_mask(
        cls,
        sparsity: float,
        mask: torch.Tensor,
        score_override: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor: ...

    @classmethod
    def _get_candidate_tiles(
        cls,
        mask: torch.Tensor,
        score_override: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if score_override is None:
            return torch.tensor([i for i in range(len(mask))])
        else:
            return torch.argwhere(
                score_override != cls._OVERRIDE_SENTINEL_VALUE
            )[:, 0].unique()


class BasePruner(BaseMaskCalculator):
    _SCORE_FILL_VALUE = float("inf")

    @classmethod
    def calculate_n_drop(cls, mask: torch.Tensor, sparsity: float) -> int:
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
        n_drop = math.ceil(
            mask.sum(dtype=torch.int) - ((1 - sparsity) * mask.numel())
        )
        return n_drop

    @classmethod
    @abstractmethod
    def get_scores(
        cls,
        mask: torch.Tensor,
        candidate_tiles: torch.Tensor,
        *args,
        **kwargs,
    ): ...


# class FineGrainedPruner(BasePruner):
#     @classmethod
#     def calculate_mask(
#         cls,
#         sparsity: float,
#         mask: torch.Tensor,
#         score_override: torch.Tensor | None = None,
#         *args,
#         **kwargs,
#     ):
#         n_drop = cls.calculate_n_drop(mask, sparsity)
#         candidate_tiles = cls._get_candidate_tiles(
#             mask,
#             score_override,
#         )
#         n_drop_per_tile = int(n_drop // len(candidate_tiles))
#         mask_slice = mask[candidate_tiles]
#         scores = cls.get_scores(
#             mask_slice, candidate_tiles, *args, **kwargs
#         )  # Logic defined by child
#         if dist.is_initialized():
#             dist.all_reduce(scores, dist.ReduceOp.AVG, async_op=False)
#         _, indices = torch.topk(scores, k=n_drop_per_tile, largest=False)
#         mask_slice = mask_slice.scatter(
#             dim=-1,
#             index=indices,
#             src=torch.zeros_like(mask_slice),
#         )
#         mask[candidate_tiles] = mask_slice
#         return mask


class FineGrainedPruner(BasePruner):
    @classmethod
    def calculate_mask(
        cls,
        sparsity: float,
        mask: torch.Tensor,
        score_override: torch.Tensor | None = None,
        *args,
        **kwargs,
    ):
        n_drop = cls.calculate_n_drop(mask, sparsity)
        candidate_tiles = cls._get_candidate_tiles(
            mask,
            score_override,
        )
        mask_slice = mask[candidate_tiles]
        mask_slice = mask[candidate_tiles]
        n_ones_per_tile_target = calculate_per_tile_n_ones(mask_slice, sparsity)
        n_drop_per_tile = torch.tensor(
            [n.sum().item() - n_ones_per_tile_target for n in mask_slice],
            dtype=torch.int,
        )
        assert (n_drop_per_tile >= 0).all()
        assert n_drop_per_tile.sum() >= n_drop
        scores = cls.get_scores(
            mask_slice, candidate_tiles, *args, **kwargs
        )  # Logic defined by child
        if dist.is_initialized():
            dist.all_reduce(scores, dist.ReduceOp.AVG, async_op=False)

        # TODO: Conditional flow on len(n_drop_per_tile.unique() == 1) to
        # unifiy with above class
        if len(n_drop_per_tile.unique()) == 1:
            _, indices = torch.topk(
                scores, k=n_drop_per_tile.unique().item(), largest=False
            )
            mask_slice = mask_slice.scatter(
                dim=-1,
                index=indices,
                src=torch.zeros_like(mask_slice),
            )
        else:  # per tile dropping
            # TODO: Vmap me
            for n_idx, (score, n_drop_this_tile) in enumerate(
                list(zip(scores, n_drop_per_tile.tolist()))
            ):
                _, indices = torch.topk(
                    score, k=n_drop_this_tile, largest=False
                )
                mask_slice[n_idx] = mask_slice[n_idx].scatter(
                    dim=-1,
                    index=indices,
                    src=torch.zeros_like(mask_slice[n_idx]),
                )
        mask[candidate_tiles] = mask_slice
        return mask


class StructuredPruner(BasePruner):
    _logger = logging.getLogger(__name__)

    @classmethod
    def calculate_mask(
        cls,
        sparsity: float,
        mask: torch.Tensor,
        score_override: torch.Tensor | None = None,
        aggregate_norm_ord: str | int = 2,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        n_drop = cls.calculate_n_drop(mask, sparsity)
        n_tiles_to_drop = math.ceil(n_drop / mask.shape[1])
        cls._logger.debug(
            f"Dropping {n_tiles_to_drop} * {mask.shape[-1]} == "
            f"{n_tiles_to_drop*mask.shape[-1]} actual vs. n_drop want: {n_drop}"
        )
        candidate_tiles = cls._get_candidate_tiles(
            mask,
            score_override,
        )
        mask_slice = mask[candidate_tiles]
        scores = cls.get_scores(
            mask_slice, candidate_tiles, *args, **kwargs
        )  # Logic defined by child
        scores = torch.linalg.norm(scores, ord=aggregate_norm_ord, dim=1)
        if dist.is_initialized():
            dist.all_reduce(scores, dist.ReduceOp.AVG, async_op=False)
        _, indices = torch.topk(scores, k=n_tiles_to_drop, largest=False)
        # zero out all elements in tile for structured pruning
        mask_slice[indices] = 0
        mask[candidate_tiles] = mask_slice
        return mask


class BaseGrower(BaseMaskCalculator):
    _SCORE_FILL_VALUE = -float("inf")

    @classmethod
    def get_n_grow(cls, mask: torch.Tensor, sparsity: float) -> int:
        # target_nnz - current nnz
        n_grow = int(mask.numel() * (1 - sparsity) - mask.sum(dtype=torch.int))
        if n_grow < 0:
            raise RuntimeError(
                f"Current sparsity > target in grow mask! Current n_ones "
                f"{int(mask.sum(dtype=torch.int).item())} vs. Target n_ones "
                f"{int(mask.numel() * (1 - sparsity))}"
            )
        return n_grow

    @classmethod
    @abstractmethod
    def get_scores(
        cls,
        mask: torch.Tensor,
        candidate_tiles: torch.Tensor,
        *args,
        **kwargs,
    ): ...


# class FineGrainedGrower(BaseGrower):

#     @classmethod
#     def calculate_mask(
#         cls,
#         sparsity: float,
#         mask: torch.Tensor,
#         score_override: torch.Tensor | None = None,
#         *args,
#         **kwargs,
#     ) -> torch.Tensor:
#         n_grow = cls.get_n_grow(mask, sparsity)
#         candidate_tiles = cls._get_candidate_tiles(mask, score_override)
#         n_grow_per_tile = int(n_grow // len(candidate_tiles))
#         mask_slice = mask[candidate_tiles]
#         scores = cls.get_scores(
#             mask_slice, candidate_tiles, *args, **kwargs
#         )  # Logic defined by child
#         if dist.is_initialized():
#             dist.all_reduce(scores, dist.ReduceOp.AVG, async_op=False)
#         _, indices = torch.topk(scores, k=n_grow_per_tile, largest=True)
#         mask_slice = mask_slice.scatter(
#             dim=-1,
#             index=indices,
#             src=torch.ones_like(mask_slice),
#         )
#         mask[candidate_tiles] = mask_slice
#         return mask


class FineGrainedGrower(BaseGrower):

    @classmethod
    def calculate_mask(
        cls,
        sparsity: float,
        mask: torch.Tensor,
        score_override: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        n_grow = cls.get_n_grow(mask, sparsity)
        candidate_tiles = cls._get_candidate_tiles(mask, score_override)
        mask_slice = mask[candidate_tiles]
        n_ones_per_tile_target = calculate_per_tile_n_ones(mask_slice, sparsity)
        n_grow_per_tile = torch.tensor(
            [n_ones_per_tile_target - n.sum().item() for n in mask_slice],
            dtype=torch.int,
        )
        assert (n_grow_per_tile >= 0).all()
        assert n_grow_per_tile.sum() <= n_grow
        scores = cls.get_scores(
            mask_slice, candidate_tiles, *args, **kwargs
        )  # Logic defined by child
        if dist.is_initialized():
            dist.all_reduce(scores, dist.ReduceOp.AVG, async_op=False)
        if len(n_grow_per_tile.unique()) == 1:
            _, indices = torch.topk(
                scores, k=n_grow_per_tile.unique().item(), largest=True
            )
            mask_slice = mask_slice.scatter(
                dim=-1,
                index=indices,
                src=torch.ones_like(mask_slice),
            )
        else:  # per tile growth
            # TODO: Vmap me
            for n_idx, (score, n_grow_this_tile) in enumerate(
                list(zip(scores, n_grow_per_tile.tolist()))
            ):
                _, indices = torch.topk(score, k=n_grow_this_tile, largest=True)
                mask_slice[n_idx] = mask_slice[n_idx].scatter(
                    dim=-1,
                    index=indices,
                    src=torch.ones_like(mask_slice[n_idx]),
                )
        mask[candidate_tiles] = mask_slice
        return mask


class StructuredGrower(BaseGrower):
    _logger = logging.getLogger(__name__)

    @classmethod
    def calculate_mask(
        cls,
        sparsity: float,
        mask: torch.Tensor,
        score_override: torch.Tensor | None = None,
        aggregate_norm_ord: str | int | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        n_grow = cls.get_n_grow(mask, sparsity)
        candidate_tiles = cls._get_candidate_tiles(mask, score_override)
        n_tiles_to_keep = n_grow // mask.shape[1]
        cls._logger.debug(
            f"Growing {n_tiles_to_keep} * {mask.shape[-1]} == "
            f"{n_tiles_to_keep*mask.shape[-1]} actual vs. n_grow want: {n_grow}"
        )
        mask_slice = mask[candidate_tiles]
        scores = cls.get_scores(
            mask_slice, candidate_tiles, *args, **kwargs
        )  # Logic defined by child
        scores = torch.linalg.norm(scores, ord=aggregate_norm_ord, dim=1)
        scores = torch.where(
            scores == float("inf"),
            torch.full_like(scores, cls._SCORE_FILL_VALUE),
            scores,
        )
        if dist.is_initialized():
            dist.all_reduce(scores, dist.ReduceOp.AVG, async_op=False)
        _, indices = torch.topk(scores, k=n_tiles_to_keep, largest=True)
        mask_slice[indices] = 1
        mask[candidate_tiles] = mask_slice
        return mask


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
            mask == 1,
            torch.abs(torch.rand_like(mask)) + cls._EPS,
            torch.full_like(mask, cls._SCORE_FILL_VALUE),
        )


class MagnitudePruner(BasePruner):

    @classmethod
    def get_scores(
        cls,
        mask: torch.Tensor,
        candidate_tiles: torch.Tensor,
        weights: torch.Tensor,
        *args,
        **kwargs,
    ):
        return torch.where(
            mask == 1,
            torch.abs(weights[candidate_tiles]),
            torch.full_like(weights[candidate_tiles], cls._SCORE_FILL_VALUE),
        )


class RandomGrower(BaseGrower):

    @classmethod
    def get_scores(
        cls, mask: torch.Tensor, candidate_tiles: torch.Tensor, *args, **kwargs
    ):
        return torch.where(
            mask == 0,
            torch.abs(torch.rand_like(mask))
            + cls._EPS,  # small eps for avoiding 0s
            torch.full_like(mask, cls._SCORE_FILL_VALUE),
        )


class GradientGrower(BaseGrower):

    @classmethod
    def get_scores(
        cls,
        mask: torch.Tensor,
        candidate_tiles: torch.Tensor,
        grads: torch.Tensor | None,
        *args,
        **kwargs,
    ):
        if grads is None:
            # Randomly grow
            return RandomGrower.get_scores(mask, candidate_tiles)
        return torch.where(
            mask == 0,
            torch.abs(grads[candidate_tiles]),
            torch.full_like(mask, cls._SCORE_FILL_VALUE),
        )
