from abc import ABC, abstractmethod
import torch


class BasePruner(ABC):

    @classmethod
    @abstractmethod
    def calculate_mask(
        cls, sparsity: float, mask: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor: ...


class BaseGrower(ABC):

    @classmethod
    def get_n_grow(cls, sparsity: float, mask: torch.Tensor) -> int:
        # target_nnz - current nnz
        n_grow = int(mask.numel() * (1 - sparsity)) - int(mask.sum().item())
        if n_grow < 0:
            raise RuntimeError(
                f"Current sparsity > target in grow mask! Current n_ones "
                f"{int(mask.sum().item())} vs. Target n_ones "
                f"{int(mask.numel() * (1 - sparsity))}"
            )
        return n_grow

    @classmethod
    @abstractmethod
    def calculate_mask(
        cls, sparsity: float, mask: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor: ...
