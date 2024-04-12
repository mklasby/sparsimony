import pytest
import numpy as np

from sparsimony.schedulers.base import ConstantScheduler, CosineDecayScheduler


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
