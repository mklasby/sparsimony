import torch
import torch.distributed as dist
from typing import Optional
import numpy as np

from sparsimony.pruners.base import BasePruner, BaseGrower

_EPS = 0.001


class UnstructuredRandomPruner(BasePruner):
    """Pruning method that randomly prunes tensor."""

    @classmethod
    def calculate_mask(
        cls,
        sparsity: float,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Randomly prunes non-zero elements in a mask by a target sparsity.

        eg., a 90% sparse mask, with target 93% sparsity will return a 93%
        sparse mask. Caller should overwrite mask.data with returned value.

        Args:
            sparsity (float): Target sparsity after pruning
                Should be a float between 0 and 1.

            mask (torch.Tensor): Mask to be applied to weight tensor.

        Returns:
            torch.Tensor: mask with prune_ratio nnz elements set to 0 randomly.
        """
        n_drop = BasePruner.calculate_n_drop(mask, sparsity)
        scores = torch.where(
            mask == 1,
            torch.abs(torch.rand_like(mask)) + _EPS,
            torch.zeros_like(mask),
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
        sparsity: float,
        mask: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        n_drop = BasePruner.calculate_n_drop(mask, sparsity)
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
            mask == 0,
            torch.abs(torch.rand_like(mask))
            + _EPS,  # small eps for avoiding 0s
            torch.zeros_like(mask),
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
            grads = torch.abs(torch.rand_like(mask)) + _EPS
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
