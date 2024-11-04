import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from sparsimony.api import rigl
from sparsimony.dst.base import DSTMixin


@pytest.fixture(
    scope="function",
    params=[
        (10, 10),  # 10x10 mask and initial sparsity of 0%
        (5, 5),  # 5x5 mask and initial sparsity of 20%
        # (32, 3, 3),  # 32x3x3 mask and initial sparsity of 90% # TODO: Conv
        (768, 3072),  # 768x3072 mask and initial sparsity of 99%
        # (768, 670091),
    ],
    ids=[
        "10x10",
        "5x5",
        # "32x3x3",
        "768x3072",
        # "768x670091",
    ],
)
def model(request):
    shape = request.param
    if len(shape) == 2:
        return nn.Linear(
            in_features=shape[0], out_features=shape[1], bias=False
        )
    elif len(shape) == 3:
        return nn.Conv2d(in_channels=shape[0], kernel_size=shape[1], bias=False)
    else:
        raise RuntimeError()


@pytest.fixture(
    scope="function",
    params=[
        (  # 10x10 mask and initial sparsity of 0%
            (10, 10),
            0.0,
        ),
        (  # 5x5 mask and initial sparsity of 20%
            (5, 5),
            0.2,
        ),
        (  # 32x3x3 mask and initial sparsity of 90%
            (32, 3, 3),
            0.9,
        ),
        (  # 768x3072 mask and initial sparsity of 99%
            (768, 3072),
            0.99,
        ),
    ],
    ids=[
        "10x10_mask_0%_sparse ",
        "5x5_mask_20%_sparse ",
        "32x3x3_mask_90%_sparse ",
        "768x3072_mask_99%_sparse ",
    ],
)
def mask(request):
    mask_size, init_sparsity = request.param
    _mask = torch.zeros(size=mask_size, dtype=torch.float)
    n_ones = int(_mask.numel() * (1 - init_sparsity))
    scores = torch.rand(size=_mask.shape)
    _, idx = torch.topk(scores.view(-1), k=n_ones, largest=True)
    _mask.view(-1)[idx] = 1
    yield _mask
    del _mask


@pytest.fixture(
    scope="function",
    params=[0.0, 0.1, 0.5, 0.75, 0.83, 0.9, 0.99],
    ids=[f" sparsity:{p}" for p in [0.0, 0.1, 0.5, 0.75, 0.83, 0.9, 0.99]],
)
def sparsity(request):
    return request.param


def test_zero_inactive_param_momentum_buffers_sgd(model, sparsity):
    # Create a mock Linear layer and optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.1)

    # Create a DSTMixin instance
    sparsifier = rigl(optimizer=optimizer, sparsity=sparsity, t_end=100)

    sparse_config = [
        {"tensor_fqn": f"{fqn}.weight"}
        for fqn, module in model.named_modules()
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d)
    ]

    sparsifier.prepare(model, sparse_config)

    # Call the function under test
    sparsifier.zero_inactive_param_momentum_buffers()

    x = torch.ones(size=(model.in_features,))
    out = model(x)
    loss = (out - torch.zeros(size=(model.out_features,))).sum()
    loss.backward()
    optimizer.step()

    for config in sparsifier.groups:
        if config["sparsity"] == 0:
            continue
        original_param = getattr(
            config["module"].parametrizations, "weight"
        ).original
        mask = getattr(config["module"].parametrizations, "weight")[0].mask
        momentum_buffer = optimizer.state[original_param]["momentum_buffer"]
        assert (momentum_buffer[mask == 0] == 0).all()


def test_zero_inactive_param_momentum_buffers_adamw(model, sparsity):
    optimizer = optim.AdamW(model.parameters(), lr=0.1)
    sparsifier = rigl(optimizer=optimizer, sparsity=sparsity, t_end=100)

    sparse_config = [
        {"tensor_fqn": f"{fqn}.weight"}
        for fqn, module in model.named_modules()
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d)
    ]

    sparsifier.prepare(model, sparse_config)

    # Call the function under test
    sparsifier.zero_inactive_param_momentum_buffers()

    x = torch.ones(size=(model.in_features,))
    out = model(x)
    loss = (out - torch.zeros(size=(model.out_features,))).sum()
    loss.backward()
    optimizer.step()

    for config in sparsifier.groups:
        if config["sparsity"] == 0:
            continue
        original_param = getattr(
            config["module"].parametrizations, "weight"
        ).original
        mask = getattr(config["module"].parametrizations, "weight")[0].mask
        for state_kw in ["exp_avg"]:
            momentum_buffer = optimizer.state[original_param][state_kw]
            assert (momentum_buffer[mask == 0] == 0).all()


def test_prune_ratio_sparsity_conversion(mask, sparsity):
    current_sparsity = (mask == 0).sum() / mask.numel()
    expected_prune_ratio = round(
        ((sparsity - current_sparsity) / (1 - current_sparsity)).item(), 6
    )
    prune_ratio = DSTMixin.get_prune_ratio_from_sparsity(mask, sparsity)
    assert expected_prune_ratio == prune_ratio
    if sparsity > current_sparsity:
        assert expected_prune_ratio > 0
    else:
        assert expected_prune_ratio <= 0
    sparsity_test = DSTMixin.get_sparsity_from_prune_ratio(mask, prune_ratio)
    assert sparsity == round(sparsity_test, 2)


def test_non_contiguous_params(caplog):
    mod = nn.Linear(10, 10)
    mod.weight = nn.Parameter(
        mod.weight[:, :5]
    )  # convert to 10 out, 5 in, non-contiguous
    # Create a mock Linear layer and optimizer
    optimizer = optim.SGD(mod.parameters(), lr=0.1, momentum=0.1)
    # Create a DSTMixin instance
    sparsifier = rigl(optimizer=optimizer, sparsity=0.1, t_end=100)
    sparse_config = [
        {"tensor_fqn": f"{fqn}.weight"}
        for fqn, module in mod.named_modules()
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d)
    ]
    assert not mod.weight.is_contiguous()
    sparsifier.prepare(mod, sparse_config)
    assert mod.weight.is_contiguous()
    logged_warning = False
    for record in caplog.records:
        if (
            "Must pass contiguous parameters to sparsimony sparsifers!"
            in record.msg
        ):
            logged_warning = True
    assert logged_warning
