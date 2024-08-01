from abc import ABC, abstractmethod
import torch


class BasePruner(ABC):

    @classmethod
    @abstractmethod
    def calculate_mask(
        cls, sparsity: float, mask: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor: ...

    @staticmethod
    def calculate_n_drop(mask: torch.Tensor, sparsity: float) -> int:
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

    @staticmethod
    def get_prune_ratio_from_sparsity(
        mask: torch.Tensor, sparsity: float
    ) -> float:
        current_sparsity = (mask == 0).sum() / mask.numel()
        return (sparsity - current_sparsity) / (1 - current_sparsity)

    @staticmethod
    def get_sparsity_from_prune_ratio(
        mask: torch.Tensor, prune_ratio: float
    ) -> float:
        current_sparsity = (mask == 0).sum() / mask.numel()
        return (prune_ratio * (1 - current_sparsity)) + current_sparsity


class BaseGrower(ABC):

    @classmethod
    def get_n_grow(cls, sparsity: float, mask: torch.Tensor) -> int:
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
    @abstractmethod
    def calculate_mask(
        cls, sparsity: float, mask: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor: ...
