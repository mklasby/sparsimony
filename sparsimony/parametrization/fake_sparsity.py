import torch
import torch.nn as nn
import deepspeed


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
        self.register_buffer("mask", mask.to(dtype=torch.bool))

    def forward(self, x):
        if len(x) == 0 and hasattr(x, "_z3_optimizer"):
            # sharded weight
            x = deepspeed.utils.safe_get_full_fp32_param(x)
        assert self.mask.shape == x.shape
        return self.mask * x

    def _broadcast_to_replicas(self):
        replicas = getattr(self, "replicas_", [])
        for r in replicas:
            for n, b in self.named_buffers():
                setattr(r, n, b)


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
        self.register_buffer(
            "dense_grad", torch.zeros(self.mask.shape, device=mask.device)
        )
        # Default true in case we need to update mask on step 1
        self.accumulate = True


def _accumulate_grad_bhook(self, grad_input, grad_output):
    if len(grad_input) > 1:
        raise ValueError("long input")
    if self.accumulate:
        self.dense_grad += grad_output[0]
    else:
        self.dense_grad = torch.zeros(self.mask.shape, device=self.mask.device)
