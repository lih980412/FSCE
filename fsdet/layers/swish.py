from torch import nn
import torch

class Swish(nn.Module):
    """Swish Module.

    This module applies the swish function:

    .. math::
        Swish(x) = x * Sigmoid(x)

    Returns:
        Tensor: The output tensor.
    """

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)