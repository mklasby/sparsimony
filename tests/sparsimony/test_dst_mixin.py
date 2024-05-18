import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from sparsimony import rigl


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


def id_fn(sparsity):
    return f"{sparsity}"


@pytest.mark.parametrize(
    "sparsity", [0.0, 0.1, 0.5, 0.75, 0.83, 0.9, 0.99], ids=id_fn
)
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


@pytest.mark.parametrize(
    "sparsity", [0.0, 0.1, 0.5, 0.75, 0.83, 0.9, 0.99], ids=id_fn
)
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
