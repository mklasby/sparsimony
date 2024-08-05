from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.distributed as dist


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
            return torch.tensor([i for i in range(mask.shape[0])])
        else:
            return torch.argwhere(
                score_override != cls._OVERRIDE_SENTINEL_VALUE
            )[:, 0].unique()


class BasePruner(BaseMaskCalculator):
    _SCORE_FILL_VALUE = -float("inf")

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
        n_drop = int(
            mask.sum(dtype=torch.int) - ((1 - sparsity) * mask.numel())
        )
        return n_drop

    @classmethod
    def calculate_mask(
        cls,
        sparsity: float,
        mask: torch.Tensor,
        score_override: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        n_drop = cls.calculate_n_drop(mask, sparsity)
        candidate_tiles = cls._get_candidate_tiles(
            mask,
            score_override,
        )
        n_drop_per_tile = int(n_drop // len(candidate_tiles))
        scores = cls.get_scores(
            mask, candidate_tiles, *args, **kwargs
        )  # Logic defined by child
        if dist.is_initialized():
            dist.all_reduce(scores, dist.ReduceOp.AVG, async_op=False)
        _, indices = torch.topk(scores, k=n_drop_per_tile, largest=False)
        mask[candidate_tiles] = mask[candidate_tiles].scatter(
            dim=-1, index=indices, src=torch.zeros_like(mask[candidate_tiles])
        )
        return mask

    @classmethod
    @abstractmethod
    def get_scores(
        cls,
        mask: torch.Tensor,
        candidate_tiles: torch.Tensor,
        *args,
        **kwargs,
    ): ...


class BaseGrower(BaseMaskCalculator):
    _SCORE_FILL_VALUE = -float("inf")

    @classmethod
    def get_n_grow(cls, mask: torch.Tensor, sparsity: float) -> int:
        # target_nnz - current nnz
        n_grow = int(mask.numel() * (1 - sparsity)) - int(
            mask.sum(dtype=torch.int).item()
        )
        if n_grow < 0:
            raise RuntimeError(
                f"Current sparsity > target in grow mask! Current n_ones "
                f"{int(mask.sum(dtype=torch.int).item())} vs. Target n_ones "
                f"{int(mask.numel() * (1 - sparsity))}"
            )
        return n_grow

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
        n_grow_per_tile = int(n_grow // len(candidate_tiles))
        scores = cls.get_scores(
            mask, candidate_tiles, *args, **kwargs
        )  # Logic defined by child
        if dist.is_initialized():
            dist.all_reduce(scores, dist.ReduceOp.AVG, async_op=False)
        _, indices = torch.topk(scores, k=n_grow_per_tile, largest=True)
        mask[candidate_tiles] = mask[candidate_tiles].scatter(
            dim=-1, index=indices, src=torch.ones_like(mask[candidate_tiles])
        )
        return mask

    @classmethod
    @abstractmethod
    def get_scores(
        cls,
        mask: torch.Tensor,
        candidate_tiles: torch.Tensor,
        *args,
        **kwargs,
    ): ...


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
            mask[candidate_tiles] == 1,
            torch.abs(torch.rand_like(mask[candidate_tiles])) + cls._EPS,
            torch.full_like(mask[candidate_tiles], cls._SCORE_FILL_VALUE),
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
            mask[candidate_tiles] == 1,
            torch.abs(weights[candidate_tiles]),
            torch.full_like(weights[candidate_tiles], cls._SCORE_FILL_VALUE),
        )


class RandomGrower(BaseGrower):

    @classmethod
    def get_scores(
        cls, mask: torch.Tensor, candidate_tiles: torch.Tensor, *args, **kwargs
    ):
        return torch.where(
            mask[candidate_tiles] == 0,
            torch.abs(torch.rand_like(mask[candidate_tiles]))
            + cls._EPS,  # small eps for avoiding 0s
            torch.full_like(mask[candidate_tiles], cls._SCORE_FILL_VALUE),
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
            mask[candidate_tiles] == 0,
            torch.abs(grads[candidate_tiles]),
            torch.full_like(mask[candidate_tiles], cls._SCORE_FILL_VALUE),
        )
