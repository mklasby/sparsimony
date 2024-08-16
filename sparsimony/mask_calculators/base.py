from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple
import math
import logging

import torch
import torch.distributed as dist

from sparsimony.utils import calculate_per_tile_n_ones, view_tensor_as_neuron


class BaseMaskCalculator(ABC):
    _OVERRIDE_SENTINEL_VALUE: float = float("-inf")
    _EPS = 0.001
    _logger = logging.getLogger(__name__)

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

        n_drop = math.ceil(mask.nansum() - ((1 - sparsity) * mask.numel()))
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
        n_ones_per_tile_target = calculate_per_tile_n_ones(mask_slice, sparsity)
        if n_ones_per_tile_target == 0:
            cls._logger.warning(
                "Found a target nnz per tile of 0! All candidate tiles will be "
                "fully pruned by this pruner."
            )
        n_drop_per_tile = torch.tensor(
            [n.nansum().item() - n_ones_per_tile_target for n in mask_slice],
            dtype=torch.int,
        )
        if not (n_drop_per_tile >= 0).all():
            cls._logger.warning(
                f"n_drop_per_tile < 0 ({n_drop_per_tile}). Will skip tiles with"
                " nnz elements < n_drop"
            )
        if not (n_drop_per_tile.sum() >= n_drop):
            cls._logger.warning(
                "(n_drop_per_tile.sum() >= n_drop): "
                f"({(n_drop_per_tile.sum() >= n_drop)}) Check sparsity level "
                "and/or padding"
            )
        scores = cls.get_scores(
            mask_slice, candidate_tiles, *args, **kwargs
        )  # Logic defined by child
        if dist.is_initialized():
            dist.all_reduce(scores, dist.ReduceOp.AVG, async_op=False)

        if len(n_drop_per_tile.unique()) == 1:
            k = n_drop_per_tile.unique().item()
            if k < 0:
                cls._logger.warning(
                    f"n_drop_per_tile == {k} for all tiles, skipping this "
                    "pruner."
                )
                mask[candidate_tiles] = mask_slice
                return mask
            _, indices = torch.topk(scores, k=k, largest=False)
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
                if n_drop_this_tile <= 0:
                    continue
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

    @classmethod
    def _get_n_tiles_to_drop(cls, sparsity: float, mask: torch.Tensor) -> int:
        nnz_el_target = math.floor(mask.numel() * (1 - sparsity))
        nnz_tiles = (torch.count_nonzero(mask, dim=-1) != 0).sum()
        n_tiles_to_drop = math.ceil(
            (nnz_tiles * mask.shape[-1] - nnz_el_target) / mask.shape[-1]
        )
        return n_tiles_to_drop

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
        # nnz_tiles = (torch.count_nonzero(mask, dim=-1) == 0).sum()
        n_tiles_to_drop = cls._get_n_tiles_to_drop(sparsity, mask)
        # cls._logger.debug(
        #     f"Dropping {n_tiles_to_drop} * {mask.shape[-1]} == "
        #     f"{n_tiles_to_drop*mask.shape[-1]} actual vs. n_drop want: {n_drop}"  # noqa
        # )
        if n_tiles_to_drop < 0:
            cls._logger.warning(
                f"n_tiles_to_drop is <0 ({n_tiles_to_drop}), continuing..."
            )
            return mask
        if n_tiles_to_drop == 0:
            return mask
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


class HierarchicalMaskCalculator(BaseMaskCalculator):

    @classmethod
    def calculate_mask(
        cls,
        sparsities: List[float],
        mask: torch.Tensor,
        calculators: List[BaseMaskCalculator],
        calculator_kwargs: List[Dict[Any, Any]],
        score_override: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # TODO: Maybe just for pruners?
        for sparsity, calculator, calc_kwargs in list(
            zip(sparsities, calculators, calculator_kwargs)
        ):
            mask = calculator.calculate_mask(
                sparsity=sparsity,
                mask=mask,
                score_override=score_override,
                **calc_kwargs,
            )
            score_override = cls._get_score_override(
                mask,
                calculator._TILE_VIEW,
                score_override,
            )
        return mask

    @classmethod
    def _get_score_override(
        cls,
        mask: torch.Tensor,
        tile_view: str | Tuple[int],
        score_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if score_override is None:
            score_override = torch.zeros_like(mask)
        _orig_shape = mask.shape
        mask = cls._reshape_t_as_view(mask, tile_view)
        score_override = cls._reshape_t_as_view(score_override, tile_view)
        ablated_tile_idx = torch.argwhere(
            torch.count_nonzero(mask, dim=-1) == 0
        ).flatten()
        score_override[ablated_tile_idx] = cls._OVERRIDE_SENTINEL_VALUE
        mask = cls._reshape_t_as_view(mask, _orig_shape)
        score_override = cls._reshape_t_as_view(score_override, _orig_shape)
        return score_override

    @classmethod
    def _reshape_t_as_view(cls, t: torch.Tensor, view: str | Tuple[int]):
        if view == "neuron":
            return view_tensor_as_neuron(t)
        elif isinstance(view, Tuple):
            return t.view(view)
        else:
            raise NotImplementedError(f"Tile view {view} not supported!")


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
        n_ones_per_tile_target = calculate_per_tile_n_ones(
            mask, sparsity, candidate_tiles
        )
        n_grow_per_tile = n_ones_per_tile_target - torch.count_nonzero(
            mask_slice, dim=-1
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
                if n_grow_this_tile > len(score):
                    cls._logger.warning(
                        "n_grow_this_tile > len(score): "
                        f"{n_grow_this_tile} > {len(score)}"
                    )
                    n_grow_this_tile = len(score)
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
            torch.logical_and(mask == 1, mask != cls._SCORE_FILL_VALUE),
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
            torch.logical_and(mask == 1, mask != cls._SCORE_FILL_VALUE),
            torch.abs(weights[candidate_tiles]),
            torch.full_like(weights[candidate_tiles], cls._SCORE_FILL_VALUE),
        )


class ThresholdPruner(BasePruner):
    @classmethod
    def get_scores(
        cls,
        mask: torch.Tensor,
        candidate_tiles: torch.Tensor,
        scorer: BaseMaskCalculator,
        scoring_tensor: torch.Tensor,
        score_threshold: float,
        *args,
        **kwargs,
    ):
        scores = scorer.get_scores(mask, candidate_tiles, scoring_tensor)
        return torch.where(
            scores > score_threshold,
            torch.one_like(mask),
            torch.zeros_like(mask),
        )


class RandomGrower(BaseGrower):
    @classmethod
    def get_scores(
        cls, mask: torch.Tensor, candidate_tiles: torch.Tensor, *args, **kwargs
    ):
        return torch.where(
            torch.logical_and(mask == 0, mask != cls._SCORE_FILL_VALUE),
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
            torch.logical_and(mask == 0, mask != cls._SCORE_FILL_VALUE),
            torch.abs(grads[candidate_tiles]),
            torch.full_like(mask, cls._SCORE_FILL_VALUE),
        )
