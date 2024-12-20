import pytest
import torch
import torch.nn as nn
from torch.nn.utils import parametrize

from sparsimony.parametrization.fake_sparsity import (
    FakeSparsity,
    FakeSparsityDenseGradBuffer,
)


class TestFakeSparsity:
    @pytest.fixture(
        params=[
            torch.ones((3, 3), dtype=torch.bool),
            torch.tensor([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=torch.bool),
        ],
        ids=["dense_mask", "50% sparse mask"],
    )
    def fake_sparsity(self, request):
        mask = request.param
        return FakeSparsity(mask)

    def test_fake_sparsity_forward(self, fake_sparsity):
        x = torch.ones((3, 3))
        output = fake_sparsity(x)
        assert torch.allclose(output, x * fake_sparsity.mask)

    def test_fake_sparsity_state_dict(self, fake_sparsity):
        state_dict = fake_sparsity.state_dict()
        assert "mask" in state_dict

    def test_fake_sparsity_grad(self, fake_sparsity):
        w1 = torch.ones((3, 3), dtype=torch.float, requires_grad=True)
        w2 = torch.ones((3, 3), dtype=torch.float, requires_grad=True)
        x = torch.arange(1, 10, dtype=torch.float).reshape(3, 3)
        out = x @ (w1 * fake_sparsity.mask) @ (w2)
        loss = (out - torch.rand((3, 3))).sum()
        loss.backward()
        assert (w1.grad[~fake_sparsity.mask] == 0).all()


class TestFakeSparsityDenseGradBuffer:
    @pytest.fixture(
        params=[
            torch.ones((3, 3)),
            torch.tensor(
                [[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=torch.float32
            ),
        ],
        ids=["dense_mask", "50% sparse mask"],
    )
    def fake_sparsity_dense_grad_buffer(self, request):
        mask = request.param
        return FakeSparsityDenseGradBuffer(mask)

    @pytest.fixture
    def linear(self, request, fake_sparsity_dense_grad_buffer):
        fake_sparsity_dense_grad_buffer.accumulate = True
        shape = fake_sparsity_dense_grad_buffer.mask.shape
        _linear = nn.Linear(
            in_features=shape[0], out_features=shape[1], bias=False
        )
        parametrize.register_parametrization(
            _linear, "weight", fake_sparsity_dense_grad_buffer
        )
        return _linear

    def test_fake_sparsity_dense_grad_buffer_forward(
        self, fake_sparsity_dense_grad_buffer, linear
    ):
        with torch.no_grad():
            linear_dense_clone = nn.Linear(
                in_features=linear.in_features,
                out_features=linear.out_features,
                bias=False,
            )
            linear_dense_clone.weight = torch.nn.Parameter(
                linear.parametrizations.weight.original.clone().detach()
            )
        target = torch.ones((3, 3))
        x = torch.ones((3, 3))
        x2 = x.clone().detach()
        out_sparse = linear(x)
        out_dense = linear_dense_clone(x2)
        loss1 = (target - out_sparse).sum()
        loss1.backward()
        loss2 = (target - out_dense).sum()
        loss2.backward()
        grad_target = linear_dense_clone.weight.grad
        assert (grad_target == fake_sparsity_dense_grad_buffer.dense_grad).all()

    def test_fake_sparsity_dense_grad_buffer_state_dict(
        self, fake_sparsity_dense_grad_buffer
    ):
        state_dict = fake_sparsity_dense_grad_buffer.state_dict()
        assert "mask" in state_dict
        assert "dense_grad" in state_dict
