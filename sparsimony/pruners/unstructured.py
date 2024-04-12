import torch
import torch.distributed as dist
from typing import Optional
import numpy as np

from sparsimony.pruners.base import BasePruner, BaseGrower


class UnstructuredRandomPruner(BasePruner):
    """Pruning method that randomly prunes tensor."""

    @classmethod
    def calculate_mask(
        cls,
        prune_ratio: float,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Randomly prunes non-zero elements in mask by prune_ratio.

        eg., a 90% sparse mask with a 30% pruning ratio will return at 93%
        sparse mask. Caller should overwrite mask.data with returned value.

        Args:
            prune_ratio (float): % of currently inactive weights to prune.
            mask (torch.Tensor): Mask to be applied to weight tensor.

        Returns:
            torch.Tensor: mask with prune_ratio nnz elements set to 0 randomly.
        """
        n_drop = int(mask.sum() * prune_ratio)
        scores = torch.where(
            mask == 1, torch.abs(torch.rand_like(mask)), torch.zeros_like(mask)
        )
        if dist.is_initialized():
            dist.all_reduce(scores, dist.ReduceOp.AVG, async_op=False)
        _, indices = torch.topk(scores.view(-1), k=n_drop, largest=True)
        mask = (
            mask.view(-1)
            .scatter(dim=0, index=indices, src=torch.zeros_like(mask.view(-1)))
            .reshape(mask.shape)
        )
        return mask


class UnstructuredMagnitudePruner(BasePruner):

    @classmethod
    def calculate_mask(
        cls,
        prune_ratio: float,
        mask: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        n_drop = int(mask.sum() * prune_ratio)
        scores = torch.where(
            mask == 1, torch.abs(weights), torch.full_like(weights, np.inf)
        )
        if dist.is_initialized():
            dist.all_reduce(scores, dist.ReduceOp.AVG, async_op=False)
        _, indices = torch.topk(scores.view(-1), k=n_drop, largest=False)
        mask = (
            mask.view(-1)
            .scatter(dim=0, index=indices, src=torch.zeros_like(mask.view(-1)))
            .reshape(mask.shape)
        )
        return mask


class UnstructuredRandomGrower(BaseGrower):

    @classmethod
    def calculate_mask(
        cls, sparsity: float, mask: torch.Tensor
    ) -> torch.Tensor:
        n_grow = cls.get_n_grow(sparsity, mask)
        scores = torch.where(
            mask == 0, torch.abs(torch.rand_like(mask)), torch.zeros_like(mask)
        )
        if dist.is_initialized():
            dist.all_reduce(scores, dist.ReduceOp.AVG, async_op=False)
        _, indices = torch.topk(scores.view(-1), k=n_grow, dim=-1, largest=True)
        mask = (
            mask.view(-1)
            .scatter(dim=0, index=indices, src=torch.ones_like(mask.view(-1)))
            .reshape(mask.shape)
        )
        return mask


class UnstructuredGradientGrower(BaseGrower):

    @classmethod
    def calculate_mask(
        cls, sparsity: float, mask: torch.Tensor, grads: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if grads is None:
            # Randomly grow
            grads = torch.rand_like(mask)
        n_grow = cls.get_n_grow(sparsity, mask)

        # Set scores of active params to 0
        scores = torch.where(
            mask == 0, torch.abs(grads), torch.full_like(grads, -1)
        )
        if dist.is_initialized():
            dist.all_reduce(scores, dist.ReduceOp.AVG, async_op=False)
        _, indices = torch.topk(
            scores.view(-1),
            k=n_grow,
            dim=-1,
            largest=True,
        )
        mask = (
            mask.view(-1)
            .scatter(dim=0, index=indices, src=torch.ones_like(mask.view(-1)))
            .reshape(mask.shape)
        )
        return mask
