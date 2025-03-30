import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import Optional, Tuple, Any, Union, List, Callable
from functools import partial
from torch.nn.utils import spectral_norm
from .conv_layers import (
    Conv2d_3x3, Conv2d_1x1, StridedConv2d_3x3, DilatedConv2d_3x3,
    Conv1d_3, Conv1d_1, StridedConv1d_3, DilatedConv1d_3
)



class CRPBlock(nn.Module):
    def __init__(self, features, n_stages, act=nn.ReLU(), maxpool=True, spec_norm=False, spatial_dim='2d'):
        super().__init__()
        if spatial_dim == '2d':
            conv_layer = Conv2d_3x3
            max_pool_layer = nn.MaxPool2d
            avg_pool_layer = nn.AvgPool2d
        elif spatial_dim == '1d':
            conv_layer = Conv1d_3
            max_pool_layer = nn.MaxPool1d
            avg_pool_layer = nn.AvgPool1d

        self.convs = nn.ModuleList()
        for i in range(n_stages):
            self.convs.append(conv_layer(features, features, stride=1, bias=False, spec_norm=spec_norm))
        self.n_stages = n_stages
        if maxpool:
            self.maxpool = max_pool_layer(kernel_size=5, stride=1, padding=2)
        else:
            self.maxpool = avg_pool_layer(kernel_size=5, stride=1, padding=2)

        self.act = act

    def forward(self, x):
        x = self.act(x)
        path = x
        for i in range(self.n_stages):
            path = self.maxpool(path)
            path = self.convs[i](path)
            x = path + x
        return x



class RCUBlock(nn.Module):
    """
    Residual Convolution Unit Block.

    Source: https://github.com/ermongroup/ncsnv2/blob/master/models/layers.py
    """
    def __init__(self, features, n_blocks, n_stages, act=nn.ReLU(), spec_norm=False, spatial_dim='2d'):
        super().__init__()
        if spatial_dim == '2d':
            conv_layer = Conv2d_3x3
        elif spatial_dim == '1d':
            conv_layer = Conv1d_3

        for i in range(n_blocks):
            for j in range(n_stages):
                setattr(self, '{}_{}_conv'.format(i + 1, j + 1), conv_layer(features, features, stride=1, bias=False,
                                                                         spec_norm=spec_norm))

        self.stride = 1
        self.n_blocks = n_blocks
        self.n_stages = n_stages
        self.act = act

    def forward(self, x):
        for i in range(self.n_blocks):
            residual = x
            for j in range(self.n_stages):
                x = self.act(x)
                x = getattr(self, '{}_{}_conv'.format(i + 1, j + 1))(x)

            x += residual
        return x


class MSFBlock(nn.Module):
    def __init__(self, in_planes, features, spec_norm=False, spatial_dim='2d'):
        """
        :param in_planes: tuples of input planes
        """
        if spatial_dim == '2d':
            conv_layer = Conv2d_3x3
        elif spatial_dim == '1d':
            conv_layer = Conv1d_3
        super().__init__()
        assert isinstance(in_planes, list) or isinstance(in_planes, tuple)
        self.convs = nn.ModuleList()
        self.features = features
        self.spatial_dim = spatial_dim

        for i in range(len(in_planes)):
            self.convs.append(conv_layer(in_planes[i], features, stride=1, bias=True, spec_norm=spec_norm))

    def forward(self, xs, shape):
        sums = torch.zeros(xs[0].shape[0], self.features, *shape, device=xs[0].device)
        for i in range(len(self.convs)):
            h = self.convs[i](xs[i])
            h = F.interpolate(h, size=shape, mode='bilinear' if self.spatial_dim == '2d' else 'linear', align_corners=True)
            sums += h
        return sums



class RefineBlock(nn.Module):
    def __init__(self, in_planes, features, act=nn.ReLU(), start=False, end=False, maxpool=True, spec_norm=False, spatial_dim='2d'):
        super().__init__()

        assert isinstance(in_planes, tuple) or isinstance(in_planes, list)
        self.n_blocks = n_blocks = len(in_planes)

        self.adapt_convs = nn.ModuleList()
        for i in range(n_blocks):
            self.adapt_convs.append(
                RCUBlock(in_planes[i], 2, 2, act, spec_norm=spec_norm, spatial_dim=spatial_dim)
            )

        self.output_convs = RCUBlock(features, 3 if end else 1, 2, act, spec_norm=spec_norm, spatial_dim=spatial_dim)

        if not start:
            self.msf = MSFBlock(in_planes, features, spec_norm=spec_norm, spatial_dim=spatial_dim)

        self.crp = CRPBlock(features, 2, act, maxpool=maxpool, spec_norm=spec_norm, spatial_dim=spatial_dim)

    def forward(self, xs, output_shape):
        assert isinstance(xs, tuple) or isinstance(xs, list)
        hs = []
        for i in range(len(xs)):
            h = self.adapt_convs[i](xs[i])
            hs.append(h)

        if self.n_blocks > 1:
            h = self.msf(hs, output_shape)
        else:
            h = hs[0]

        h = self.crp(h)
        h = self.output_convs(h)

        return h




class ConvMeanPool(nn.Module):
    """
    Source: https://github.com/ermongroup/ncsnv2/blob/master/models/layers.py
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int=3,
        biases: bool=True,
        spec_norm: bool=False,
        spatial_dim='2d',
    ):
        super().__init__()
        self.spatial_dim = spatial_dim
        if spatial_dim == '2d':
            conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
        elif spatial_dim == '1d':
            conv = nn.Conv1d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
        if spec_norm:
            conv = spectral_norm(conv)
        self.conv = conv
        

    def forward(self, inputs):
        output = self.conv(inputs)
        if self.spatial_dim == '2d':
            output = sum([output[:, :, ::2, ::2], output[:, :, 1::2, ::2],
                      output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
        elif self.spatial_dim == '1d':
            output = sum([output[:, :, ::2], output[:, :, 1::2]]) / 2.
        return output



class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, resample=None, act=nn.ELU(),
                 normalization=nn.BatchNorm2d, dilation=None, spec_norm=False, spatial_dim='2d'):
        super().__init__()
        self.non_linearity = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample
        self.normalization = normalization

        if spatial_dim == '2d':
            conv_layer = Conv2d_3x3
            dilated_conv_layer = DilatedConv2d_3x3
        elif spatial_dim == '1d':
            conv_layer = Conv1d_3
            dilated_conv_layer = DilatedConv1d_3
        if resample == 'down':
            if dilation is not None:
                self.conv1 = dilated_conv_layer(input_dim, input_dim, dilation=dilation, spec_norm=spec_norm)
                self.normalize2 = normalization(input_dim)
                self.conv2 = dilated_conv_layer(input_dim, output_dim, dilation=dilation, spec_norm=spec_norm)
                conv_shortcut = partial(dilated_conv_layer, dilation=dilation, spec_norm=spec_norm)
            else:
                self.conv1 = conv_layer(input_dim, input_dim, spec_norm=spec_norm)
                self.normalize2 = normalization(input_dim)
                self.conv2 = ConvMeanPool(input_dim, output_dim, 3, spec_norm=spec_norm, spatial_dim=spatial_dim)
                conv_shortcut = partial(ConvMeanPool, kernel_size=1, spec_norm=spec_norm, spatial_dim=spatial_dim)

        elif resample is None:
            if dilation is not None:
                conv_shortcut = partial(dilated_conv_layer, dilation=dilation, spec_norm=spec_norm)
                self.conv1 = dilated_conv_layer(input_dim, output_dim, dilation=dilation, spec_norm=spec_norm)
                self.normalize2 = normalization(output_dim)
                self.conv2 = dilated_conv_layer(output_dim, output_dim, dilation=dilation, spec_norm=spec_norm)
            else:
                conv_shortcut = partial(Conv2d_1x1, spec_norm=spec_norm)
                self.conv1 = conv_layer(input_dim, output_dim, spec_norm=spec_norm)
                self.normalize2 = normalization(output_dim)
                self.conv2 = conv_layer(output_dim, output_dim, spec_norm=spec_norm)
        else:
            raise Exception('invalid resample value')

        if output_dim != input_dim or resample is not None:
            self.shortcut = conv_shortcut(input_dim, output_dim)

        self.normalize1 = normalization(input_dim)


    def forward(self, x):
        output = self.normalize1(x)
        output = self.non_linearity(output)
        output = self.conv1(output)
        output = self.normalize2(output)
        output = self.non_linearity(output)
        output = self.conv2(output)

        if self.output_dim == self.input_dim and self.resample is None:
            shortcut = x
        else:
            shortcut = self.shortcut(x)

        return shortcut + output

#from .ncsn import NCSNv2
#
#if __name__ == '__main__':
#    ncsnv2=NCSNv2(3, nn.BatchNorm2d, nn.ELU(), 64, torch.tensor([0.9, 0.5, 0.2, 0.1]), 4)
#    x=torch.randn(1,3,32,32)
#    y=torch.tensor([0,1,2,3])
#    output=ncsnv2(x,y[0])
#    print(output.shape)