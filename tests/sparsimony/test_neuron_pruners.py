import pytest
import torch
import math

from sparsimony.mask_calculators.neuron import (
    NeuronRandomPruner,
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
            (32, 3, 3, 3),
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
    neuron_view = _mask.view(-1, math.prod(_mask.shape[1:]))
    n_tiles = math.floor(n_ones / neuron_view.shape[-1])
    scores = torch.rand(size=(_mask.shape[0],))
    _, idx = torch.topk(scores, k=n_tiles, largest=True)
    _mask[idx] = 1
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


def test_neuron_random(mask, prune_ratio):
    # Call the method to be tested
    sparsity = DSTMixin.get_sparsity_from_prune_ratio(mask, prune_ratio)
    pruned_mask = NeuronRandomPruner.calculate_mask(sparsity, mask)

    # Assertions
    assert pruned_mask.shape == mask.shape
    mask.data = pruned_mask
    # Calculate the expected number of non-zero elements after pruning
    # Total elements minus number of zero elements.
    neuron_view_mask = mask.reshape(-1, math.prod(mask.shape[1:]))
    nnz_el_target = math.floor(mask.numel() * (1 - sparsity))
    nnz_tiles = math.floor(nnz_el_target / neuron_view_mask.shape[-1])
    expected_nonzero = nnz_tiles * neuron_view_mask.shape[-1]
    assert (
        torch.count_nonzero(neuron_view_mask, dim=1) != 0
    ).sum() == nnz_tiles
    max_n_ones = math.ceil(mask.numel() * (1 - sparsity))
    assert expected_nonzero <= max_n_ones
