import torch
import torch.nn as nn
from typing import Callable


def get_act(activation_name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if activation_name == 'elu':
        return nn.ELU()
    elif activation_name == 'relu':
        return nn.ReLU()
    elif activation_name == 'lrelu':
        return nn.LeakyReLU(negative_slope=0.2)
    elif activation_name =='selu':
        return nn.SELU()
    elif activation_name == 'gelu':
        return nn.GELU()
    elif activation_name =='silu':
        return nn.SiLU()
    elif activation_name == 'swish':
        def swish(x):
            return x * torch.sigmoid(x)
        return swish
    elif activation_name =='sigmoid':
        return nn.Sigmoid()
    elif activation_name == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError('activation function does not exist!')












