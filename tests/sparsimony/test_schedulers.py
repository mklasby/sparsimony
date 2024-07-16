import pytest
import numpy as np

from sparsimony.schedulers.base import (
    ConstantScheduler,
    CosineDecayScheduler,
    SoftMemoryBoundScheduler,
)


class TestSchedulers:
    @pytest.fixture
    def constant_scheduler(self):
        return ConstantScheduler(pruning_ratio=0.5, t_end=10, delta_t=2)

    @pytest.fixture
    def cosine_decay_scheduler(self):
        return CosineDecayScheduler(pruning_ratio=0.5, t_end=10, delta_t=2)

    def test_constant_scheduler_call(self, constant_scheduler):
        # Test for step before t_end and divisible by delta_t
        assert constant_scheduler(2) == 0.5
        # Test for step not divisible by delta_t
        assert constant_scheduler(3) is None
        # Test for step after t_end
        assert constant_scheduler(11) is None

    def test_cosine_decay_scheduler_call(self, cosine_decay_scheduler):
        # Test for step before t_end and divisible by delta_t
        assert cosine_decay_scheduler(
            2
        ) == cosine_decay_scheduler.pruning_ratio / 2 * (
            1 + np.cos((2 * np.pi) / cosine_decay_scheduler.t_end)
        )
        # Test for step not divisible by delta_t
        assert cosine_decay_scheduler(3) is None
        # Test for step after t_end
        assert cosine_decay_scheduler(11) is None


def id_fn(args):
    pruning_ratio, t_end, delta_t, t_grow = args
    return (
        f"pruning_ratio: {pruning_ratio} t_end: {t_end} delta_t: {delta_t} "
        f"t_grow: {t_grow}"
    )


class TestSoftMemoryBound:

    @pytest.fixture(params=[(0.3, 200, 100, 20)], ids=id_fn)
    def scheduler(self, request):
        pruning_ratio, t_end, delta_t, t_grow = request.param
        _scheduler = SoftMemoryBoundScheduler(
            pruning_ratio=pruning_ratio,
            t_end=t_end,
            delta_t=delta_t,
            t_grow=t_grow,
        )
        yield _scheduler
        del _scheduler

    def test_next_step_update(self, scheduler):
        t_end = scheduler.t_end
        delta_t = scheduler.delta_t
        t_grow = scheduler.t_grow

        update_cycles = t_end // delta_t
        start_buffer_steps = [
            delta_t * n - t_grow for n in list(range(1, update_cycles + 1))
        ]
        # grow_next_steps = [
        #     (delta_t * n) - 1 for n in list(range(1, update_cycles + 1))
        # ]
        # prune_next_steps = [
        #     delta_t * n + t_grow for n in list(range(1, update_cycles + 1))
        # ]
        for step in range(1, t_end + 1):
            # Test cases for next_step_update
            if step in start_buffer_steps:
                assert (
                    scheduler.next_step_update(step) is True
                )  # Grow next step
            # elif step in grow_next_steps:
            #     print(step)
            #     assert scheduler.next_step_update(step) is True
            # elif step in prune_next_steps:
            #     assert scheduler.next_step_update(step) is True
            else:
                assert scheduler.next_step_update(step) is False

    def test_call(self, scheduler):
        pruning_ratio = scheduler.pruning_ratio
        t_end = scheduler.t_end
        delta_t = scheduler.delta_t
        t_grow = scheduler.t_grow

        update_cycles = t_end // delta_t
        grow_steps = [delta_t * n for n in list(range(1, update_cycles + 1))]
        prune_steps = [
            delta_t * n + t_grow for n in list(range(1, update_cycles + 1))
        ]
        for step in range(1, t_end + 1):
            # Test cases for next_step_update
            if step in grow_steps:
                assert scheduler(step) == -pruning_ratio  # Grow next step
            elif step in prune_steps:
                assert scheduler(step) == pruning_ratio
            else:
                assert scheduler(step) is None
