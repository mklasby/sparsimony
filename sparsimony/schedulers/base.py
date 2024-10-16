from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseScheduler(ABC):

    def __init__(
        self,
        quantity: float,
        t_end: int,
        delta_t: int,
    ):
        self.quantity = quantity
        self.t_end = t_end
        self.delta_t = delta_t

    def next_step_update(self, last_step: int) -> bool:
        if (last_step + 1) % self.delta_t == 0:
            return True
        return False

    @abstractmethod
    def __call__(self, step: int) -> Optional[float]: ...


class StaticSchedule(BaseScheduler):
    def __init__(self, *args, **kwargs):
        return

    def __call__(self, *args, **kwargs):
        return None


class ConstantScheduler(BaseScheduler):

    def __init__(
        self, quantity: float, t_end: int, delta_t: int, *args, **kwargs
    ):
        super().__init__(quantity, t_end, delta_t)

    def __call__(self, step: int) -> Optional[float]:
        if step % self.delta_t != 0:
            return None
        if step > self.t_end:
            return None
        else:
            return self.quantity


class CosineDecayScheduler(BaseScheduler):

    def __init__(
        self, quantity: float, t_end: int, delta_t: int, *args, **kwargs
    ):
        super().__init__(quantity, t_end, delta_t)

    def __call__(self, step: int) -> Optional[float]:
        if step % self.delta_t != 0:
            return None
        if step > self.t_end:
            return None
        else:
            return self.quantity / 2 * (1 + np.cos((step * np.pi) / self.t_end))


class SoftMemoryBoundScheduler(BaseScheduler):
    def __init__(
        self,
        quantity: float,
        t_end: int,
        delta_t: int,
        t_grow: int,
        *args,
        **kwargs
    ):
        super().__init__(quantity, t_end, delta_t)
        self.t_grow = t_grow
        assert t_grow < delta_t

    def next_step_update(self, last_step: int) -> bool:
        if last_step + 1 > self.t_end:
            return False
        if last_step % self.delta_t == (self.delta_t - self.t_grow):
            # start filling buffers for grow step
            return True
        # elif last_step % self.delta_t == 0:
        #     return True
        # elif (
        #     last_step % self.delta_t == self.t_grow and last_step > self.delta_t  # noqa
        # ):
        #     # Prune next step (need plus one?)
        #     return True
        return False

    def __call__(self, step: int) -> Optional[float]:
        if step > self.t_end:
            return None
        if step % self.delta_t == 0:
            return -self.quantity  # Grow by prune ratio
        elif step % self.delta_t == self.t_grow and step > self.delta_t:
            return self.quantity  # Prune by prune ratio
        else:
            return None


class AcceleratedCubicScheduler(BaseScheduler):
    def __init__(
        self,
        t_end: int,
        delta_t: int,
        t_accel: int,
        initial_sparsity: float = 0.0,
        accelerated_sparsity: float = 0.7,
        final_sparsity: float = 0.9,
        *args,
        **kwargs
    ):
        super().__init__(None, t_end, delta_t)
        self.t_accel = t_accel
        self.initial_sparsity = initial_sparsity
        self.accelerated_sparsity = accelerated_sparsity
        self.final_sparsity = final_sparsity

    def __call__(self, step: int) -> Optional[float]:
        if step > self.t_end:
            return None
        elif step % self.delta_t != 0:
            return None
        else:  # Prune
            if step < self.t_accel:
                return self.initial_sparsity
            else:
                return (
                    self.final_sparsity
                    + (self.accelerated_sparsity - self.final_sparsity)
                    * (1 - (step - self.t_accel) / self.t_end) ** 3
                )
