import torch
import torch.nn as nn
import random
import numpy as np
from typing import Callable


def get_act(activation_name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if activation_name == 'elu':
        return nn.ELU()
    elif activation_name == 'relu':
        return nn.ReLU()
    elif activation_name == 'lrelu':
        return nn.LeakyReLU(negative_slope=0.1)
    elif activation_name =='selu':
        return nn.SELU()
    elif activation_name == 'gelu':
        return nn.GELU()
    elif activation_name =='silu':
        return nn.SiLU()
    elif activation_name == 'swish':
        return lambda x: x * torch.sigmoid(x)
    elif activation_name =='mish':
        return nn.Mish()
    elif activation_name =='sigmoid':
        return nn.Sigmoid()
    elif activation_name == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError('activation function does not exist!')



def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    return





