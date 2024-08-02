import pytest
import torch

from sparsimony.pruners.unstructured import (
    BasePruner,
    UnstructuredRandomPruner,
    UnstructuredMagnitudePruner,
)
from sparsimony.dst.base import DSTMixin


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
    params=[0.0, 0.2, 0.5, 0.9, 0.99],
    ids=[f" prune_ratio:{p*100}%" for p in [0.0, 0.2, 0.5, 0.9, 0.99]],
)
def prune_ratio(request):
    return request.param


@pytest.fixture
def weights(mask):
    return torch.rand_like(mask) * mask


def test_unstructured_random_pruner(mask, prune_ratio):
    # Call the method to be tested
    sparsity = DSTMixin.get_sparsity_from_prune_ratio(mask, prune_ratio)
    pruned_mask = UnstructuredRandomPruner.calculate_mask(sparsity, mask)

    # Assertions
    assert pruned_mask.shape == mask.shape
    assert pruned_mask.data_ptr() != mask.data_ptr()

    # Calculate the expected number of non-zero elements after pruning
    # Total elements minus number of zero elements.
    expected_nonzero = mask.sum() - int(mask.sum() * prune_ratio)
    assert torch.count_nonzero(pruned_mask) == expected_nonzero


def test_unstructured_pruners(mask, prune_ratio, weights):
    # Seed weights with known values
    weights = torch.where(
        mask == 1, torch.full_like(weights, 100), torch.zeros_like(weights)
    )
    sparsity = DSTMixin.get_sparsity_from_prune_ratio(mask, prune_ratio)
    n_drop = BasePruner.calculate_n_drop(mask, sparsity)
    _, idx = torch.topk(weights.view(-1), k=n_drop, largest=True)
    weights = torch.scatter(
        weights.view(-1),
        dim=0,
        index=idx,
        src=torch.full_like(weights.view(-1), 1),
    ).reshape(weights.shape)
    # Call the method to be tested
    pruned_mask = UnstructuredMagnitudePruner.calculate_mask(
        sparsity, mask, weights
    )

    # Assertions
    assert pruned_mask.shape == mask.shape
    assert pruned_mask.data_ptr() != mask.data_ptr()

    # Calculate the expected number of non-zero elements after pruning
    expected_nonzero = mask.sum() - int(mask.sum() * prune_ratio)
    assert torch.count_nonzero(pruned_mask) == expected_nonzero

    # Assert correct values were dropped!
    assert (pruned_mask.view(-1)[idx] == 0).all()
