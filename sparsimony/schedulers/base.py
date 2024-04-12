from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseScheduler(ABC):

    def __init__(
        self,
        pruning_ratio: float,
        t_end: int,
        delta_t: int,
    ):
        self.pruning_ratio = pruning_ratio
        self.t_end = t_end
        self.delta_t = delta_t

    def next_step_update(self, last_step: int) -> bool:
        if (last_step + 1) % self.delta_t == 0:
            return True
        return False

    @abstractmethod
    def __call__(self, step: int) -> Optional[float]: ...


class ConstantScheduler(BaseScheduler):

    def __init__(
        self, pruning_ratio: float, t_end: int, delta_t: int, *args, **kwargs
    ):
        super().__init__(pruning_ratio, t_end, delta_t)

    def __call__(self, step: int) -> Optional[float]:
        if step % self.delta_t != 0:
            return None
        if step > self.t_end:
            return None
        else:
            return self.pruning_ratio


class CosineDecayScheduler(BaseScheduler):

    def __init__(
        self, pruning_ratio: float, t_end: int, delta_t: int, *args, **kwargs
    ):
        super().__init__(pruning_ratio, t_end, delta_t)

    def __call__(self, step: int) -> Optional[float]:
        if step % self.delta_t != 0:
            return None
        if step > self.t_end:
            return None
        else:
            return (
                self.pruning_ratio
                / 2
                * (1 + np.cos((step * np.pi) / self.t_end))
            )


class SoftMemoryBoundScheduler(BaseScheduler):
    def __init__(
        self,
        pruning_ratio: float,
        t_end: int,
        delta_t: int,
        t_grow: int,
        *args,
        **kwargs
    ):
        super().__init__(pruning_ratio, t_end, delta_t)
        self.t_grow = t_grow
        assert t_grow < delta_t

    def next_step_update(self, last_step: int) -> bool:
        if last_step % self.delta_t == (self.delta_t - self.t_grow):
            # start filling buffers for grow step
            return True
        # elif (last_step + 1) % self.delta_t == self.t_grow:
        #     return True
        return False

    def __call__(self, step: int) -> Optional[float]:
        if step % self.delta_t == 0:
            return -self.pruning_ratio  # Grow by prune ratio
        elif step % (self.delta_t + self.t_grow) == 0:
            return self.pruning_ratio  # Prune by prune ratio
        else:
            return None
