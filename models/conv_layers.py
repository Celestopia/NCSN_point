import torch.nn as nn
from torch.nn.utils import spectral_norm


class Conv2d_1x1(nn.Module):
    """
    1x1 convolutional layer.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int=1,
        bias: bool=True,
        spec_norm: bool=False
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1, 
            stride=stride,
            padding=0,
            bias=bias
        )
        if spec_norm:
            self.conv = spectral_norm(self.conv)
    
    def forward(self, x):
        return self.conv(x)


class Conv2d_3x3(nn.Module):
    """
    3x3 convolutional layer.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        bias: bool = True,
        spec_norm: bool = False
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias
        )
        if spec_norm:
            self.conv = spectral_norm(self.conv)
    
    def forward(self, x):
        return self.conv(x)


class StridedConv2d_3x3(nn.Module):
    """
    3x3 convolutional layer with stride 2.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias: bool = True,
        spec_norm: bool = False
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            bias=bias
        )
        if spec_norm:
            self.conv = spectral_norm(self.conv)
    
    def forward(self, x):
        return self.conv(x)


class DilatedConv2d_3x3(nn.Module):
    """
    3x3 convolutional layer.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
        bias: bool = True,
        spec_norm: bool = False
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=bias
        )
        if spec_norm:
            self.conv = spectral_norm(self.conv)
    
    def forward(self, x):
        return self.conv(x)



class Conv1d_1(nn.Module):
    """
    1d convolutional layer with kernel size 1.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int=1,
        bias: bool=True,
        spec_norm: bool=False
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1, 
            stride=stride,
            padding=0,
            bias=bias
        )
        if spec_norm:
            self.conv = spectral_norm(self.conv)
    
    def forward(self, x):
        return self.conv(x)


class Conv1d_3(nn.Module):
    """
    1d convolutional layer with kernel size 3.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        bias: bool = True,
        spec_norm: bool = False
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias
        )
        if spec_norm:
            self.conv = spectral_norm(self.conv)
    
    def forward(self, x):
        return self.conv(x)


class StridedConv1d_3(nn.Module):
    """
    1d convolutional layer with kernel size 3 and stride 2.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias: bool = True,
        spec_norm: bool = False
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            bias=bias
        )
        if spec_norm:
            self.conv = spectral_norm(self.conv)
    
    def forward(self, x):
        return self.conv(x)


class DilatedConv1d_3(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
        bias: bool = True,
        spec_norm: bool = False
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=bias
        )
        if spec_norm:
            self.conv = spectral_norm(self.conv)
    
    def forward(self, x):
        return self.conv(x)











