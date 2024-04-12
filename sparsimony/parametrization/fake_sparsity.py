import torch
import torch.nn as nn


class FakeSparsity(nn.Module):
    r"""Parametrization for the weights. Should be attached to the 'weight' or
    any other parameter that requires a mask applied to it.

    Note::

        Once the mask is passed, the variable should not change the id. The
        contents of the mask can change, but the mask reference itself should
        not.
    """

    def __name__(self):
        return "FakeSparsity"

    def __init__(self, mask):
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(self, x):
        # print("in fake sparsity forward")
        assert self.mask.shape == x.shape
        return self.mask * x


class FakeSparsityDenseGradBuffer(FakeSparsity):
    r"""Parametrization for the weights. Should be attached to the 'weight' or
    any other parameter that requires a mask applied to it.

    Note::

        Once the mask is passed, the variable should not change the id. The
        contents of the mask can change, but the mask reference itself should
        not.
    """

    def __name__(self):
        return "FakeSparsityDenseGradBuffer"

    def __init__(self, mask):
        super().__init__(mask)
        self.register_full_backward_hook(_accumulate_grad_bhook)
        self.register_buffer("dense_grad", torch.zeros_like(self.mask))
        # Default true in case we need to update mask on step 1
        self.accumulate = True


def _accumulate_grad_bhook(self, grad_input, grad_output):
    if len(grad_input) > 1:
        raise ValueError("long input")
    if self.accumulate:
        self.dense_grad += grad_output[0]
        # TODO: fix unit test, output is correct.
        # self.dense_grad += grad_input[0]
        # TODO: Confirm we don't need to sync this manually
    else:
        self.dense_grad = torch.zeros_like(self.mask)


class LOPParametrization(FakeSparsity):
    r"""Parametrization for the weights. Should be attached to the 'weight' or
    any other parameter that requires a mask applied to it.

    Note::

        Once the mask is passed, the variable should not change the id. The
        contents of the mask can change, but the mask reference itself should
        not.
    """

    def __name__(self):
        return "LOPParametrization"

    def __init__(self, mask: torch.Tensor, history_len: int):
        super().__init__(mask)
        self.history_len = history_len
        self.sparse_grad_history = torch.empty(size=(history_len, *mask.shape))
        self.sparse_weight_history = torch.empty(
            size=(history_len, *mask.shape)
        )
        self.dense_grad_history = torch.empty(size=(history_len, *mask.shape))
        self.dense_weights = None
        # Default true in case we need to update mask on step 1
        self.accumulate = False
        self._accumulation_counter = 0
        self._register_hooks()

    def register_dense_weights(self, dense_weights: torch.Tensor):
        # We need to pass a reference to the dense weights since the weight
        # property of the parametrized module will change each time forward
        # is called!
        self.dense_weights = dense_weights

    def _register_hooks(self):
        self.register_full_backward_hook(_populate_history)

    @torch.no_grad
    @property
    def sparse_weight(self) -> torch.Tensor:
        if self.dense_weights is None:
            raise RuntimeError(
                "Need to assign dense_weight with register_dense_weight() after"
                "construction of parametrization"
            )
        return self.forward(self.mask * self.dense_weights)

    def reset_history(self, accumulate: bool = False):
        self.sparse_grad_history = torch.empty(
            size=(self.history_len, *self.mask.shape)
        )
        self.sparse_weight_history = torch.empty(
            size=(self.history_len, *self.mask.shape)
        )
        self.dense_grad_history = torch.empty(
            size=(self.history_len, *self.mask.shape)
        )
        self.accumulate = accumulate
        self._accumulation_counter = 0


def _populate_history(self, grad_input, grad_output):
    # TODO: Doesn't seem to be working as expected, getting mostly zeros
    if self.accumulate:
        self.dense_grad_history[self._accumulation_counter] = (
            grad_output[0].clone().detach()
        )
        self.sparse_grad_history[self._accumulation_counter] = (
            grad_input[0].clone().detach()
        )
        self.sparse_weight_history[self._accumulation_counter] = (
            self.sparse_weight.clone().detach()
        )
        self._accumulation_counter += 1
