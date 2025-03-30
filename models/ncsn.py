import torch.nn as nn
import torch.nn.functional as F
import torch
from functools import partial
from .layers import ResidualBlock, RefineBlock


class NCSNv2(nn.Module):
    """
    Noise Conditional Score Network - Version 2
    
    Related Articles:
    - NCSN: Generative Modeling by Estimating Gradients of the Data Distribution, https://arxiv.org/abs/1907.05600
    - NCSNv2: Improved Techniques for Training Score-Based Generative Models, https://arxiv.org/abs/2006.09011
    """
    def __init__(
        self,
        data_channels: int,
        norm,
        act,
        ngf,
        sigmas,
        num_classes: int,
    ):
        """
        Args:
            data_channels (int): Number of channels in the input data.
            norm (nn.Module): Normalization layer.
            act (nn.Module): Activation function.
            ngf (int): Number of filters in the first layer.
            sigmas (Tensor): Standard deviations of the noises.
            num_classes (int): Number of classes (noise levels).
        """
        super().__init__()
        self.norm = norm
        self.ngf = ngf
        self.num_classes = num_classes
        self.act = act
        self.register_buffer('sigmas', sigmas)
        
        self.begin_conv = nn.Conv2d(data_channels, ngf, kernel_size=3, stride=1, padding=1)
        
        self.normalizer = self.norm(ngf, self.num_classes)
        self.end_conv = nn.Conv2d(ngf, data_channels, kernel_size=3, stride=1, padding=1)
        
        self.res1 = nn.Sequential(
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                          normalization=self.norm),
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                          normalization=self.norm)
        )
        
        self.res2 = nn.Sequential(
            ResidualBlock(self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm)
        )
        
        self.res3 = nn.Sequential(
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm, dilation=2),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm, dilation=2)
        )
        
        self.res4 = nn.Sequential(
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm, dilation=4),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm, dilation=4)
        )
        
        self.refine1 = RefineBlock([2 * self.ngf], 2 * self.ngf, act=act, start=True)
        self.refine2 = RefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, act=act)
        self.refine3 = RefineBlock([2 * self.ngf, 2 * self.ngf], self.ngf, act=act)
        self.refine4 = RefineBlock([self.ngf, self.ngf], self.ngf, act=act, end=True)

    def forward(self, x, y):
        """
        Args:
            x (Tensor): Input data. Shape: (N, C, H, W)
            y (Tensor): Condition. Shape: (N,).
                - Note that every entry in y must be an integer and be in the range [0, num_classes-1].
        Returns:
            output (Tensor): Output data. Shape: (N, C, H, W)
        """
        output = self.begin_conv(x)

        layer1 = self.res1(output)
        layer2 = self.res2(layer1)
        layer3 = self.res3(layer2)
        layer4 = self.res4(layer3)

        ref1 = self.refine1([layer4], layer4.shape[2:])
        ref2 = self.refine2([layer3, ref1], layer3.shape[2:])
        ref3 = self.refine3([layer2, ref2], layer2.shape[2:])
        output = self.refine4([layer1, ref3], layer1.shape[2:])

        output = self.normalizer(output)
        output = self.act(output)
        output = self.end_conv(output)

        used_sigmas = self.sigmas[y].view(x.shape[0], *([1] * len(x.shape[1:])))

        output = output / used_sigmas

        return output




class SimpleNet1d(nn.Module):
    def __init__(self, data_dim, hidden_dim, sigmas):
        super().__init__()
        self.fc1 = nn.Linear(data_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, data_dim)
        self.sigmas = sigmas


    def forward(self, x, y):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        used_sigmas = self.sigmas[y].view(x.shape[0], *([1] * len(x.shape[1:])))
        x = x / used_sigmas
        return x





class SimpleNet2d(nn.Module):
    def __init__(
        self,
        data_channels: int,
        norm,
        act,
        ngf,
        sigmas,
        num_classes: int,
    ):
        """
        Args:
            data_channels (int): Number of channels in the input data.
            norm (nn.Module): Normalization layer.
            act (nn.Module): Activation function.
            ngf (int): Number of filters in the first layer.
            sigmas (Tensor): Standard deviations of the noises.
            num_classes (int): Number of classes (noise levels).
        """
        super().__init__()
        self.norm = norm
        self.ngf = ngf
        self.num_classes = num_classes
        self.act = act
        self.register_buffer('sigmas', sigmas)
        
        self.begin_conv = nn.Conv1d(data_channels, ngf, kernel_size=3, stride=1, padding=1)
        
        self.normalizer = self.norm(ngf, self.num_classes)
        self.end_conv = nn.Conv1d(ngf, data_channels, kernel_size=3, stride=1, padding=1)

        self.res1 = nn.Sequential(
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                          normalization=self.norm, spatial_dim='1d'),
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                          normalization=self.norm, spatial_dim='1d'),
        )
        
        self.res2 = nn.Sequential(
            ResidualBlock(self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm, spatial_dim='1d'),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm, spatial_dim='1d'),
        )
        
        self.res3 = nn.Sequential(
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm, dilation=2, spatial_dim='1d'),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm, dilation=2, spatial_dim='1d')
        )
        
        self.res4 = nn.Sequential(
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm, dilation=4, spatial_dim='1d'),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm, dilation=4, spatial_dim='1d')
        )
        
        self.refine1 = RefineBlock([2 * self.ngf], 2 * self.ngf, act=act, start=True, spatial_dim='1d')
        self.refine2 = RefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, act=act, spatial_dim='1d')
        self.refine3 = RefineBlock([2 * self.ngf, 2 * self.ngf], self.ngf, act=act, spatial_dim='1d')
        self.refine4 = RefineBlock([self.ngf, self.ngf], self.ngf, act=act, end=True, spatial_dim='1d')

    def forward(self, x, y):
        """
        Args:
            x (Tensor): Input data. Shape: (N, C, L)
            y (Tensor): Condition. Shape: (N,).
                - Note that every entry in y must be an integer and be in the range [0, num_classes-1].
        Returns:
            output (Tensor): Output data. Shape: (N, C, L)
        """
        output = self.begin_conv(x)

        layer1 = self.res1(output)
        layer2 = self.res2(layer1)
        layer3 = self.res3(layer2)
        layer4 = self.res4(layer3)

        ref1 = self.refine1([layer4], layer4.shape[2:])
        ref2 = self.refine2([layer3, ref1], layer3.shape[2:])
        ref3 = self.refine3([layer2, ref2], layer2.shape[2:])
        output = self.refine4([layer1, ref3], layer1.shape[2:])

        output = self.normalizer(output)
        output = self.act(output)
        output = self.end_conv(output)

        used_sigmas = self.sigmas[y].view(x.shape[0], *([1] * len(x.shape[1:])))

        output = output / used_sigmas

        return output









