import pytest
import torch

from sparsimony.mask_calculators import (
    UnstructuredGrower,
    MagnitudeScorer,
    RandomScorer,
)


@pytest.fixture(
    scope="function",
    params=[
        (  # 10x10 mask and initial sparsity of 0%
            (10, 10),
            0,
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
        "10x10_mask_0%_init_sparse",
        "5x5_mask_20%_init_sparse",
        "32x3x3_mask_90%_init_sparse",
        "768x3072_mask_99%_init_sparse",
    ],
)
def mask(request):
    mask_size, init_sparsity = request.param
    _mask = torch.zeros(size=mask_size, dtype=torch.bool)
    n_ones = int(_mask.numel() * (1 - init_sparsity))
    scores = torch.rand(size=_mask.shape)
    _, idx = torch.topk(scores.view(-1), k=n_ones, largest=True)
    _mask.view(-1)[idx] = 1
    yield _mask
    del _mask


@pytest.fixture(
    scope="function",
    params=[0.0, 0.2, 0.5, 0.9, 0.99],
    ids=[f"target_sparsity_{p*100}%" for p in [0.0, 0.2, 0.5, 0.89, 0.98]],
)
def sparsity(request):
    return request.param


@pytest.fixture
def dense_grads(mask):
    return torch.rand_like(mask, dtype=torch.float)


def test_unstructured_random_grower(mask, sparsity):
    grower = UnstructuredGrower(RandomScorer)
    n_grow = int(mask.numel() * (1 - sparsity)) - int(mask.sum().item())
    if n_grow < 0:
        with pytest.raises(RuntimeError) as excinfo:
            grown_mask = grower.calculate_mask(sparsity, mask, values=mask)
        assert f"{int(mask.sum(dtype=torch.int).item())}" in str(excinfo.value)
        assert str(int(mask.numel() * (1 - sparsity))) in str(excinfo.value)
        return
    else:
        grown_mask = grower.calculate_mask(sparsity, mask, values=mask)
    # Assertions
    assert grown_mask.shape == mask.shape

    expected_nonzero = int(mask.numel() * (1 - sparsity))
    assert torch.count_nonzero(grown_mask).item() == expected_nonzero


def test_unstructured_pruners(mask, sparsity, dense_grads):
    grower = UnstructuredGrower(MagnitudeScorer)
    n_grow = int(mask.numel() * (1 - sparsity)) - int(
        mask.sum(dtype=torch.int).item()
    )
    if n_grow < 0:
        with pytest.raises(RuntimeError) as excinfo:
            _ = grower.calculate_mask(sparsity, mask, values=dense_grads)
        assert f"{int(mask.sum(dtype=torch.int).item())}" in str(excinfo.value)
        assert str(int(mask.numel() * (1 - sparsity))) in str(excinfo.value)
        return

    # Seed weights with known values
    dense_grads = torch.where(
        mask == 0,
        torch.full_like(dense_grads, 1),
        torch.zeros_like(dense_grads),
    )
    _, idx = torch.topk(dense_grads.view(-1), k=n_grow, largest=True)
    dense_grads = torch.scatter(
        dense_grads.view(-1),
        dim=0,
        index=idx,
        src=torch.full_like(dense_grads.view(-1), 100),
    ).reshape(dense_grads.shape)
    # Call the method to be tested
    grown_mask = grower.calculate_mask(sparsity, mask, values=dense_grads)

    # Assertions
    assert grown_mask.shape == mask.shape

    # Calculate the expected number of non-zero elements after pruning
    expected_nonzero = int(mask.numel() * (1 - sparsity))
    assert torch.count_nonzero(grown_mask) == expected_nonzero

    # Assert correct values were grown!
    assert (grown_mask.view(-1)[idx] == 1).all()
