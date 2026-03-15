from __future__ import annotations

import torch
from torch import nn
from torch.autograd import Function


class _GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output.neg() * ctx.alpha, None


class GradientReversal(nn.Module):
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, alpha: float | None = None) -> torch.Tensor:
        return _GradientReversalFunction.apply(inputs, float(alpha if alpha is not None else self.alpha))
