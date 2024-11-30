import monai
from monai.networks.nets import UNet
from monai.networks.layers.factories import Conv, LayerFactory, Pool
from monai.networks.layers import get_act_layer
from monai.networks.blocks import Convolution, ResidualUnit, ResidualSELayer, ADN
from monai.networks.layers.convutils import same_padding, stride_minus_kernel_padding
from monai.networks.layers import SkipConnection

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from typing import Sequence, Union, Tuple, Optional
from types import NoneType
from functools import partial

#######
# Basic Convolution unit
#######
class ConvolutionUnit(nn.Sequential):
    """
    Modified convolution block from monai.networks.blocks.Convolution for extra features.
    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        strides: convolution stride. Defaults to 1.
        kernel_size: convolution kernel size. Defaults to 3.
        adn_ordering: a string representing the ordering of activation, normalization, and dropout.
            Defaults to "NDA".
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        dropout: dropout ratio. Defaults to no dropout.
        dropout_dim: determine the spatial dimensions of dropout. Defaults to 1.

            - When dropout_dim = 1, randomly zeroes some of the elements for each channel.
            - When dropout_dim = 2, Randomly zeroes out entire channels (a channel is a 2D feature map).
            - When dropout_dim = 3, Randomly zeroes out entire channels (a channel is a 3D feature map).

            The value of dropout_dim should be no larger than the value of `spatial_dims`.
        dilation: dilation rate. Defaults to 1.
        bias: whether to have a bias term. Defaults to True.
        conv_only: whether to use the convolutional layer only. Defaults to False.
        padding: controls the amount of implicit zero-paddings on both sides for padding number of points
            for each dimension. Defaults to None.
        output_padding: controls the additional size added to one side of the output shape.
            Defaults to None.
        conv_op: type of convolution operation. "conv" for original monai.networks.blocks.Convolution. "dws" for depth-wise separable convolution. "spatially_separable" for spatially separable convolution.
    Notes:
        groups parameter removed.
    
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        strides: Sequence[int] | int = 1,
        kernel_size: Sequence[int] | int = 3,
        adn_ordering: str = 'NDA',
        act: tuple | str | None = 'PRELU',
        norm: tuple | str | None = 'INSTANCE',
        dropout: tuple | str | float | None = None,
        dropout_dim: int | None = 1,
        dilation: Sequence[int] | int = 1,
        bias: bool = True,
        conv_only: bool = False,
        padding: Sequence[int] | int | None = None,
        output_padding: Sequence[int] | int | None = None,
        conv_op: str = 'conv',
        conv_kwargs: dict = {},
    )->None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        if padding is None:
            padding = same_padding(kernel_size, dilation)
        conv_type = Conv[conv_op, self.spatial_dims]
        conv: nn.Module
        is_transposed = 'trans' in conv_op
        if is_transposed:
            if output_padding is None:
                output_padding = stride_minus_kernel_padding(1, strides)
            conv = conv_type(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
                output_padding=output_padding,
                bias=bias,
                dilation=dilation,
                **conv_kwargs
            )
        else:
            conv = conv_type(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
                bias=bias,
                dilation=dilation,
                **conv_kwargs
            )
        if norm.lower() == 'spectral':
            for mod in conv.modules():
                if hasattr(mod, 'weight'):
                    mod = nn.utils.spectral_norm(mod)
            # conv = nn.utils.spectral_norm(conv)
            norm = None
        self.add_module('conv', conv)
        if conv_only:
            return
        if act is None and norm is None and dropout is None:
            return
        self.add_module(
            "adn",
            ADN(
                ordering=adn_ordering,
                in_channels=out_channels,
                act=act,
                norm=norm,
                norm_dim=self.spatial_dims,
                dropout=dropout,
                dropout_dim=dropout_dim,
            ),
        )
    def forward(self, x):
        for module in self:
            x = module(x)
        return x
    
class PixelShuffle(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        scale_factor: int,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.scale_factor = scale_factor
    def forward(self, x):
        if self.spatial_dims == 2:
            b, inc, h, w = x.shape
            outc = inc // (self.scale_factor ** 2)
            return x.view(b,outc,self.scale_factor,self.scale_factor,h,w).permute(0,1,4,2,5,3).reshape(b,outc,h*self.scale_factor,w*self.scale_factor)
        elif self.spatial_dims == 3:
            b,inc,h,w,z = x.shape
            outc = inc // (self.scale_factor ** 3)
            return x.view(b,outc,self.scale_factor,self.scale_factor,self.scale_factor,h,w,z).permute(0,1,5,2,6,3,7,4).reshape(b,outc,h*self.scale_factor,w*self.scale_factor,z*self.scale_factor)

class PixelUnshuffle(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        scale_factor: int
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.scale_factor = scale_factor
    def forward(self, x):
        if self.spatial_dims == 2:
            b,inc,h,w = x.shape
            outc = inc * (self.scale_factor ** 2)
            return x.reshape(b,inc,h//self.scale_factor,self.scale_factor,w//self.scale_factor,self.scale_factor).permute(0,1,3,5,2,4).reshape(b,outc,h//self.scale_factor,w//self.scale_factor)
        elif self.spatial_dims == 3:
            b,inc,h,w,z = x.shape
            outc = inc * (self.scale_factor ** 3)
            return x.reshape(b,inc,h//self.scale_factor,self.scale_factor,w//self.scale_factor,self.scale_factor,z//self.scale_factor,self.scale_factor).permute(0,1,3,5,7,2,4,6).reshape(b,outc,h//self.scale_factor, w//self.scale_factor, z//self.scale_factor)

class PixelUpsample(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        scale_factors: Sequence[int] | int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int] = 3,
        adn_ordering: str = 'NDA',
        act: tuple | str | None = 'PRELU',
        norm: tuple | str | None = 'INSTANCE',
        dropout: tuple | str | float | None = None,
        dropout_dim: int | None = 1,
        dilation: Union[Sequence[int], int] = 1,
        bias: bool = True,
        conv_only: bool = True,
        padding: Union[Sequence[int], int, NoneType] = None,
        conv_op: str = 'conv',
        conv_kwargs: dict = {},
    ):
        super().__init__()
        scale_factors = [scale_factors for i in range(spatial_dims)] if isinstance(scale_factors, int) else scale_factors
        max_scale_factor = max(scale_factors)
        self.upsample_module = PixelShuffle(spatial_dims, max_scale_factor)
        inc = in_channels // (2 ** spatial_dims)
        conv_scale_factors = [max_scale_factor // scale_factor for scale_factor in scale_factors]
        if conv_op == 'convtrans':
            conv_op = 'conv'
        elif conv_op.endswith('_transpose'):
            conv_op = conv_op.split('_transpose')[0]
        self.conv = ConvolutionUnit(spatial_dims, inc, out_channels, conv_scale_factors, kernel_size, adn_ordering, act, norm, dropout, dropout_dim, dilation, bias, conv_only, padding, None, conv_op, conv_kwargs)
    def forward(self, x):
        return self.conv(self.upsample_module(x))

class PixelDownsample(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        scale_factors: Sequence[int] | int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int] = 3,
        adn_ordering: str = 'NDA',
        act: tuple | str | None = 'PRELU',
        norm: tuple | str | None = 'INSTANCE',
        dropout: tuple | str | float | None = None,
        dropout_dim: int | None = 1,
        dilation: Union[Sequence[int], int] = 1,
        bias: bool = True,
        conv_only: bool = True,
        padding: Union[Sequence[int], int, NoneType] = None,
        conv_op: str = 'conv',
        conv_kwargs: dict = {},
    ):
        super().__init__()
        scale_factors = [scale_factors for i in range(spatial_dims)] if isinstance(scale_factors, int) else scale_factors
        max_scale_factor = max(scale_factors)
        self.downsample_module = PixelUnshuffle(spatial_dims, max_scale_factor)
        inc = in_channels * (2 ** spatial_dims)
        conv_scale_factors = [max_scale_factor // scale_factor for scale_factor in scale_factors]
        output_padding = stride_minus_kernel_padding(1, conv_scale_factors) if any([stride > 1 for stride in conv_scale_factors]) else None
        if any([stride > 1 for stride in conv_scale_factors]):
            if conv_op == 'conv':
                conv_op = 'convtrans'
            elif not conv_op.endswith('_transpose'):
                conv_op = conv_op + '_transpose'
            kernel_size = stride
        self.conv = ConvolutionUnit(spatial_dims, inc, out_channels, conv_scale_factors, kernel_size, adn_ordering, act, norm, dropout, dropout_dim, dilation, bias, conv_only, padding, output_padding, conv_op, conv_kwargs)
    def forward(self, x):
        return self.conv(self.downsample_module(x))
    
def make_haar_wavelet_kernels(spatial_dims, include_ll):
    if spatial_dims == 3:
        num_filterbank = 8 if include_ll else 7
        harr_wav = torch.ones(1, 2, 2, 2) * 0.5
        harr_wav_L = harr_wav.clone()
        harr_wav_H1 = harr_wav.clone()
        harr_wav_H1[:,0] = -1 * harr_wav_H1[:,0]
        harr_wav_H2 = harr_wav.clone()
        harr_wav_H2[:,:,0] = -1 * harr_wav_H2[:,:,0]
        harr_wav_H3 = harr_wav.clone()
        harr_wav_H3[:,:,:,0] = -1 * harr_wav_H3[:,:,:,0]

        harr_wav_D1 = harr_wav.clone()
        harr_wav_D1[:,0,0] = -1 * harr_wav_D1[:,0,0]
        harr_wav_D1[:,1,1] = -1 * harr_wav_D1[:,1,1]
        harr_wav_D2 = harr_wav.clone()
        harr_wav_D2[:,0,:,0] = -1 * harr_wav_D2[:,0,:,0]
        harr_wav_D2[:,1,:,1] = -1 * harr_wav_D2[:,1,:,1]
        harr_wav_D3 = harr_wav.clone()
        harr_wav_D3[:,:,0,0] = -1 * harr_wav_D3[:,:,0,0]
        harr_wav_D3[:,:,1,1] = -1 * harr_wav_D3[:,:,1,1]
        harr_wav_D4 = harr_wav.clone()
        harr_wav_D4[:,0,0,0] = -1 * harr_wav_D4[:,0,0,0]
        harr_wav_D4[:,0,1,1] = -1 * harr_wav_D4[:,0,1,1]
        harr_wav_D4[:,1,0,1] = -1 * harr_wav_D4[:,1,0,1]
        harr_wav_D4[:,1,1,0] = -1 * harr_wav_D4[:,1,1,0]
        list_kernels = [harr_wav_L, harr_wav_H1, harr_wav_H2, harr_wav_H3, harr_wav_D1, harr_wav_D2, harr_wav_D3, harr_wav_D4] if num_filterbank == 8 else [harr_wav_H1, harr_wav_H2, harr_wav_H3, harr_wav_D1, harr_wav_D2, harr_wav_D3, harr_wav_D4]
    elif spatial_dims == 2:
        num_filterbank = 4 if include_ll else 3
        harr_wav = torch.ones(1, 2, 2) * 0.5
        harr_wav_L = harr_wav.clone()
        harr_wav_H1 = harr_wav.clone()
        harr_wav_H1[:,0] = -1 * harr_wav_H1[:,0]
        harr_wav_H2 = harr_wav.clone()
        harr_wav_H2[:,:,0] = -1 * harr_wav_H2[:,:,0]
        harr_wav_D = harr_wav.clone()
        harr_wav_D[:,0,0] = -1 * harr_wav_D[:,0,0]
        harr_wav_D[:,1,1] = -1 * harr_wav_D[:,1,1]
        
        list_kernels = [harr_wav_L, harr_wav_H1, harr_wav_H2, harr_wav_D] if include_ll else [harr_wav_H1, harr_wav_H2, harr_wav_D]
    return list_kernels

class WaveletUpsample(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        scale_factors: Sequence[int] | int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int] = 3,
        adn_ordering: str = 'NDA',
        act: tuple | str | None = 'PRELU',
        norm: tuple | str | None = 'INSTANCE',
        dropout: tuple | str | float | None = None,
        dropout_dim: int | None = 1,
        dilation: Union[Sequence[int], int] = 1,
        bias: bool = True,
        conv_only: bool = True,
        padding: Union[Sequence[int], int, NoneType] = None,
        conv_op: str = 'conv',
        conv_kwargs: dict = {'groups': 8},
        include_ll: bool = True
    ):
        super().__init__()
        scale_factors = [scale_factors for i in range(spatial_dims)] if isinstance(scale_factors, int) else scale_factors
        max_scale_factor = max(scale_factors)
        if spatial_dims == 3:
            num_filterbank = 8 if include_ll else 7
        else:
            num_filterbank = 4 if include_ll else 3
        wav_channels = in_channels * num_filterbank
        self.wavelet_kernels = Conv['convtrans', spatial_dims](in_channels, wav_channels, 2, max_scale_factor, 0, groups = in_channels, bias = False)
        for p in self.wavelet_kernels.parameters():
            p.requires_grad = False
        list_kernels = make_haar_wavelet_kernels(spatial_dims, include_ll)
        # print(self.wavelet_kernels.weight.data.shape, list_kernels[0].shape)
        for idx, kernel in enumerate(list_kernels):
            if spatial_dims == 3:
                kernel = kernel.unsqueeze(0).expand(in_channels, -1, -1, -1, -1)
            elif spatial_dims == 2:
                kernel = kernel.unsqueeze(0).expand(in_channels, -1, -1, -1)
            self.wavelet_kernels.weight.data[:, [idx]] = kernel
        conv_scale_factors = [max_scale_factor // scale_factor for scale_factor in scale_factors]
        padding = same_padding(kernel_size, dilation)
        self.conv = ConvolutionUnit(
            spatial_dims=spatial_dims,
            in_channels = wav_channels,
            out_channels = out_channels,
            strides=conv_scale_factors,
            kernel_size = kernel_size,
            adn_ordering=adn_ordering,
            act=act,
            norm=norm,
            dropout=dropout,
            dropout_dim=dropout_dim,
            dilation=dilation,
            bias=bias,
            conv_only=conv_only,
            padding=padding,
            conv_op=conv_op,
            conv_kwargs=conv_kwargs
        )
    def forward(self, x):
        return self.conv(self.wavelet_kernels(x))

class WaveletDownsample(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        scale_factors: Sequence[int] | int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int] = 3,
        adn_ordering: str = 'NDA',
        act: tuple | str | None = 'PRELU',
        norm: tuple | str | None = 'INSTANCE',
        dropout: tuple | str | float | None = None,
        dropout_dim: int | None = 1,
        dilation: Union[Sequence[int], int] = 1,
        bias: bool = True,
        conv_only: bool = True,
        padding: Union[Sequence[int], int, NoneType] = None,
        conv_op: str = 'convtrans',
        conv_kwargs: dict = {'groups': 8},
        include_ll: bool = True
    ):
        super().__init__()
        scale_factors = [scale_factors for i in range(spatial_dims)] if isinstance(scale_factors, int) else scale_factors
        max_scale_factor = max(scale_factors)
        if spatial_dims == 3:
            num_filterbank = 8 if include_ll else 7
        else:
            num_filterbank = 4 if include_ll else 3
        wav_channels = in_channels * num_filterbank
        self.wavelet_kernels = Conv['conv', spatial_dims](in_channels, wav_channels, 2, max_scale_factor, 0, groups = in_channels, bias = False)
        for p in self.wavelet_kernels.parameters():
            p.requires_grad = False
        list_kernels = make_haar_wavelet_kernels(spatial_dims, include_ll)
        # print(self.wavelet_kernels.weight.data.shape, list_kernels[0].shape)
        for idx, kernel in enumerate(list_kernels):
            if spatial_dims == 3:
                kernel = kernel.unsqueeze(0).expand(in_channels, -1, -1, -1, -1)
            elif spatial_dims == 2:
                kernel = kernel.unsqueeze(0).expand(in_channels, -1, -1, -1)
            self.wavelet_kernels.weight.data[idx*in_channels:(idx+1)*in_channels,:] = kernel
        conv_scale_factors = [max_scale_factor // scale_factor for scale_factor in scale_factors]
        padding = same_padding(kernel_size, dilation)
        self.conv = ConvolutionUnit(
            spatial_dims=spatial_dims,
            in_channels = wav_channels,
            out_channels = out_channels,
            strides=conv_scale_factors,
            kernel_size = kernel_size,
            adn_ordering=adn_ordering,
            act=act,
            norm=norm,
            dropout=dropout,
            dropout_dim=dropout_dim,
            dilation=dilation,
            bias=bias,
            conv_only=conv_only,
            padding=padding,
            conv_op=conv_op,
            conv_kwargs=conv_kwargs
        )
    def forward(self, x):
        return self.conv(self.wavelet_kernels(x))
    
class UpsampleLayer(nn.Module):
    """
    
    Args:
        spatial_dims: 
        scale_factors:
        in_channels:
        out_channels:
        kernel_size:
        adn_ordering:
        act:
        norm:
        dropout:
        dropout_dim:
        dilation:
        groups:
        bias:
        conv_only:
        
        mode:
            * pixel_shuffle: use pixel shuffle for upsampling. Convolution layer is followed by to match the output channel
            * conv_trans: use transposed convolution for upsampling.
            * haar_wavelet: use transposed convolution using haar wavelet kernels. Convolution layer is followed by to match the output channel
    """
    def __init__(
        self,
        spatial_dims: int,
        scale_factors: Sequence[int] | int,
        in_channels: int,
        out_channels: int,
        kernel_size: 'Sequence[int] | int' = 3,
        adn_ordering: 'str' = 'NDA',
        act: 'tuple | str | None' = 'PRELU',
        norm: 'tuple | str | None' = 'INSTANCE',
        dropout: 'tuple | str | float | None' = None,
        dropout_dim: 'int | None' = 1,
        dilation: 'Sequence[int] | int' = 1,
        bias: 'bool' = True,
        conv_only: 'bool' = True,
        mode: str = 'pixel_shuffle',
        conv_op: str = 'convtrans',
        conv_kwargs: dict = {},
        upsample_kwargs: dict = {}
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels,
        self.out_channels = out_channels
        self.scale_factors = scale_factors
        self.mode = mode
        padding = same_padding(kernel_size, dilation)
        if self.mode == 'pixel_shuffle':
            self.upsample = PixelUpsample(
                spatial_dims = spatial_dims, 
                scale_factors=scale_factors, 
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=kernel_size, 
                adn_ordering=adn_ordering,
                act=act, 
                norm=norm,
                dropout=dropout, 
                dropout_dim=dropout_dim, 
                dilation=dilation, 
                bias=bias, 
                conv_only=conv_only, 
                padding=padding, 
                conv_op = conv_op, 
                conv_kwargs=conv_kwargs,
                **upsample_kwargs
            )
        elif self.mode == 'convtrans':
            self.upsample = ConvolutionUnit(
                spatial_dims=spatial_dims, 
                in_channels=in_channels, 
                out_channels=out_channels, 
                strides = scale_factors, 
                kernel_size=kernel_size, 
                adn_ordering=adn_ordering, 
                act=act, 
                norm=norm, 
                dropout=dropout, 
                dropout_dim=dropout_dim, 
                dilation=dilation, 
                bias=bias, 
                conv_only=conv_only, 
                padding=padding, 
                output_padding=None, 
                conv_op = conv_op, 
                conv_kwargs=conv_kwargs,
                **upsample_kwargs
            )
        elif self.mode == 'haar_wavelet':
            if conv_op == 'convtrans':
                conv_op = 'conv'
            else:
                conv_op = conv_op.split('_transpose')[0]
            self.upsample = WaveletUpsample(
                spatial_dims = spatial_dims,
                scale_factors = scale_factors,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                adn_ordering=adn_ordering,
                act = act,
                norm = norm,
                dropout = dropout,
                dropout_dim = dropout_dim,
                dilation = dilation,
                bias = bias,
                conv_only = conv_only,
                padding = padding,
                conv_op = conv_op,
                conv_kwargs = conv_kwargs,
                **upsample_kwargs
            )
    def forward(self, x):
        return self.upsample(x)
    
class DownsampleLayer(nn.Module):
    """
    
    Args:
        spatial_dims: 
        scale_factors:
        in_channels:
        out_channels:
        kernel_size:
        adn_ordering:
        act:
        norm:
        dropout:
        dropout_dim:
        dilation:
        groups:
        bias:
        conv_only:
        
        mode:
            * pixel_shuffle: use pixel shuffle for upsampling. Convolution layer is followed by to match the output channel
            * conv_trans: use transposed convolution for upsampling.
            * haar_wavelet: use transposed convolution using haar wavelet kernels. Convolution layer is followed by to match the output channel
    """
    def __init__(
        self,
        spatial_dims: int,
        scale_factors: Sequence[int] | int,
        in_channels: int,
        out_channels: int,
        kernel_size: 'Sequence[int] | int' = 3,
        adn_ordering: 'str' = 'NDA',
        act: 'tuple | str | None' = 'PRELU',
        norm: 'tuple | str | None' = 'INSTANCE',
        dropout: 'tuple | str | float | None' = None,
        dropout_dim: 'int | None' = 1,
        dilation: 'Sequence[int] | int' = 1,
        bias: 'bool' = True,
        conv_only: 'bool' = True,
        mode: str = 'pixel_unshuffle',
        conv_op: str = 'conv',
        conv_kwargs: dict = {},
        downsample_kwargs: dict = {}
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels,
        self.out_channels = out_channels
        self.scale_factors = scale_factors
        self.mode = mode
        padding = same_padding(kernel_size, dilation)
        if self.mode == 'pixel_unshuffle':
            self.downsample = PixelDownsample(
                spatial_dims = spatial_dims, 
                scale_factors=scale_factors, 
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=kernel_size, 
                adn_ordering=adn_ordering,
                act=act, 
                norm=norm,
                dropout=dropout, 
                dropout_dim=dropout_dim, 
                dilation=dilation, 
                bias=bias, 
                conv_only=conv_only, 
                padding=padding, 
                conv_op = conv_op, 
                conv_kwargs=conv_kwargs,
                **downsample_kwargs
            )
        elif self.mode == 'conv':
            self.downsample = ConvolutionUnit(
                spatial_dims=spatial_dims, 
                in_channels=in_channels, 
                out_channels=out_channels, 
                strides = scale_factors, 
                kernel_size=kernel_size, 
                adn_ordering=adn_ordering, 
                act=act, 
                norm=norm, 
                dropout=dropout, 
                dropout_dim=dropout_dim, 
                dilation=dilation, 
                bias=bias, 
                conv_only=conv_only, 
                padding=padding, 
                output_padding=None, 
                conv_op = conv_op, 
                conv_kwargs=conv_kwargs,
                **downsample_kwargs
            )
        elif self.mode == 'haar_wavelet':
            if conv_op == 'conv':
                conv_op = 'convtrans'
            elif not conv_op.endswith('_transpose'):
                conv_op = conv_op + '_transpose'
            self.downsample = WaveletDownsample(
                spatial_dims = spatial_dims,
                scale_factors = scale_factors,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                adn_ordering=adn_ordering,
                act = act,
                norm = norm,
                dropout = dropout,
                dropout_dim = dropout_dim,
                dilation = dilation,
                bias = bias,
                conv_only = conv_only,
                padding = padding,
                conv_op = conv_op,
                conv_kwargs = conv_kwargs,
                **downsample_kwargs
            )
    def forward(self, x):
        return self.downsample(x)
    
#######
# Block Modules
#######
class ConvBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: 'int',
        in_channels: 'int',
        out_channels: 'int',
        strides: 'Sequence[int] | int' = 1,
        kernel_size: 'Sequence[int] | int' = 3,
        subunits: 'int' = 2,
        adn_ordering: 'str' = 'NDA',
        act: 'tuple | str | None' = 'PRELU',
        norm: 'tuple | str | None' = 'INSTANCE',
        dropout: 'tuple | str | float | None' = None,
        dropout_dim: 'int | None' = 1,
        dilation: 'Sequence[int] | int' = 1,
        bias: 'bool' = True,
        last_conv_only: 'bool' = False,
        padding: 'Sequence[int] | int | None' = None,
        conv_op: str = 'conv',
        conv_kwargs: dict = {}
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Sequential()
        if not padding:
            padding = same_padding(kernel_size, dilation)
        schannels = in_channels
        sstrides = strides
        subunits = max(1, subunits)
        
        for su in range(subunits):
            conv_only = last_conv_only and su == (subunits - 1)
            unit = ConvolutionUnit(
                self.spatial_dims,
                schannels,
                out_channels,
                strides=sstrides,
                kernel_size=kernel_size,
                adn_ordering=adn_ordering,
                act=act,
                norm=norm,
                dropout=dropout,
                dropout_dim=dropout_dim,
                dilation=dilation,
                bias=bias,
                conv_only=conv_only,
                padding=padding,
                conv_op = conv_op,
                conv_kwargs = conv_kwargs
            )

            self.conv.add_module(f"unit{su:d}", unit)

            # after first loop set channels and strides to what they should be for subsequent units
            schannels = out_channels
            sstrides = 1
#         # apply convolution to input to change number of output channels and size to match that coming from self.conv
#         if np.prod(strides) != 1 or in_channels != out_channels:
#             rkernel_size = kernel_size
#             rpadding = padding

#             if np.prod(strides) == 1:  # if only adapting number of channels a 1x1 kernel is used with no padding
#                 rkernel_size = 1
#                 rpadding = 0

#             conv_type = Conv[Conv.CONV, self.spatial_dims]
#             self.residual = conv_type(in_channels, out_channels, rkernel_size, strides, rpadding, bias=bias)
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class ResBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: 'int',
        in_channels: 'int',
        out_channels: 'int',
        strides: 'Sequence[int] | int' = 1,
        kernel_size: 'Sequence[int] | int' = 3,
        subunits: 'int' = 2,
        adn_ordering: 'str' = 'NDA',
        act: 'tuple | str | None' = 'PRELU',
        norm: 'tuple | str | None' = 'INSTANCE',
        dropout: 'tuple | str | float | None' = None,
        dropout_dim: 'int | None' = 1,
        dilation: 'Sequence[int] | int' = 1,
        bias: 'bool' = True,
        last_conv_only: 'bool' = False,
        padding: 'Sequence[int] | int | None' = None,
        conv_op: str = 'conv',
        conv_kwargs: dict = {}
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Sequential()
        self.residual = nn.Identity()
        if not padding:
            padding = same_padding(kernel_size, dilation)
        schannels = in_channels
        sstrides = strides
        subunits = max(1, subunits)
        
        for su in range(subunits):
            conv_only = last_conv_only and su == (subunits - 1)
            unit = ConvolutionUnit(
                self.spatial_dims,
                schannels,
                out_channels,
                strides=sstrides,
                kernel_size=kernel_size,
                adn_ordering=adn_ordering,
                act=act,
                norm=norm,
                dropout=dropout,
                dropout_dim=dropout_dim,
                dilation=dilation,
                bias=bias,
                conv_only=conv_only,
                padding=padding,
                conv_op = conv_op,
                conv_kwargs = conv_kwargs
            )

            self.conv.add_module(f"unit{su:d}", unit)

            # after first loop set channels and strides to what they should be for subsequent units
            schannels = out_channels
            sstrides = 1
        # apply convolution to input to change number of output channels and size to match that coming from self.conv
        if np.prod(strides) != 1 or in_channels != out_channels:
            rkernel_size = kernel_size
            rpadding = padding

            if np.prod(strides) == 1:  # if only adapting number of channels a 1x1 kernel is used with no padding
                rkernel_size = 1
                rpadding = 0

            conv_type = Conv[conv_op, self.spatial_dims]
            is_transposed = 'trans' in conv_op
            if is_transposed:
                output_padding = stride_minus_kernel_padding(1, strides)
                self.residual = conv_type(in_channels, out_channels, rkernel_size, strides, rpadding, bias=bias, output_padding = output_padding, **conv_kwargs)
            else:
                self.residual = conv_type(in_channels, out_channels, rkernel_size, strides, rpadding, bias=bias, **conv_kwargs)
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        res: torch.Tensor = self.residual(x)  # create the additive residual from x
        cx: torch.Tensor = self.conv(x)  # apply x to sequence of operations
        return cx + res  # add the residual to the output
    
class ConnectionBlock(nn.Module):
    def __init__(
        self,
        downblock:nn.Module,
        downsample:nn.Module,
        subblock:nn.Module,
        upsample:nn.Module,
        upblock:nn.Module,
        multires_block: nn.Module,
        level: str,
        skip_mode: str = 'cat',
        prefix: str = ''
    )->None:
        super().__init__()
        self.downblock = downblock
        self.downsample = downsample
        self.subblock = subblock
        self.upsample = upsample
        self.upblock = upblock
        self.prefix = prefix
        self.level = f"{prefix}level_{level}"
        self.skip_mode = skip_mode
        self.multires_block = multires_block
    def forward(self, x: torch.Tensor, list_features: list = None, x_multires: dict = None)->torch.Tensor | tuple:
        if (self.multires_block is not None) and (self.level in x_multires):
            multires_input = x_multires[self.level]
            feature_multires = self.multires_block(multires_input)
            x = torch.cat([x, feature_multires], dim = 1)
        x_skip = self.downblock(x)
        if isinstance(self.subblock, ConnectionBlock):
            x = self.upsample(self.subblock(self.downsample(x_skip), list_features, x_multires))
        else:
            x = self.upsample(self.subblock(self.downsample(x_skip)))
        if self.skip_mode == 'cat':
            x = torch.cat([x, x_skip], dim = 1)
        elif self.skip_mode == 'sum':
            x = x + x_skip
        elif self.skip_mode == 'mult':
            x = x * x_skip
        x = self.upblock(x)
        if list_features is not None:
            list_features[self.level.removeprefix(self.prefix)] = x
        return x
    
class UNetFeatureGenerator(nn.Module):
    def __init__(
        self,
        spatial_dims: 'int',
        in_channels: 'int',
        # out_channels: 'int',
        channels: 'Sequence[int]',
        strides: 'Sequence[int]',
        kernel_size: 'Sequence[int] | int' = 3,
        up_kernel_size: 'Sequence[int] | int' = 3,
        num_units: 'int' = 1,
        act: 'tuple | str' = ('RELU', {'inplace': True}),
        norm: 'tuple | str' = 'INSTANCE',
        dropout: 'float' = 0.0,
        bias: 'bool' = True,
        adn_ordering: 'str' = 'NDA',
        block_mode: str = 'base', # base, res
        block_conv_op: str = 'conv', # conv, dws, spatially_separable
        block_conv_kwargs: dict = {},
        downsample_mode: str = 'conv',
        downsample_conv_op: str = 'conv',
        downsample_conv_kwargs: dict = {},
        downsample_kwargs: dict = {},
        upsample_mode: str = 'convtrans',
        upsample_conv_op: str = 'convtrans',
        upsample_conv_kwargs: dict = {},
        upsample_kwargs: dict = {},
        deep_supervision: bool = False,
        multires_input: bool = False,
        base_feature_channels: int = None
    ):
        super().__init__()
        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if isinstance(kernel_size, Sequence) and len(kernel_size) != spatial_dims:
            raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence) and len(up_kernel_size) != spatial_dims:
            raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        # self.out_channels = out_channels
        self.channels = channels
        self.strides = []
        for stride in strides:
            if isinstance(stride, int):
                self.strides.append([stride] * self.spatial_dims)
            else:
                self.strides.append(stride)
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_units
        self.num_units = num_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering
        self.block_mode = block_mode
        self.block_conv_op = block_conv_op
        self.block_conv_kwargs = block_conv_kwargs
        self.downsample_mode = downsample_mode
        self.downsample_conv_op = downsample_conv_op
        self.downsample_conv_kwargs = downsample_conv_kwargs
        self.downsample_kwargs = downsample_kwargs
        self.upsample_mode = upsample_mode
        self.upsample_conv_op = upsample_conv_op
        self.upsample_conv_kwargs = upsample_conv_kwargs
        self.upsample_kwargs = upsample_kwargs
        self.deep_supervision = deep_supervision
        self.multires_input = multires_input
        self.network_depth = len(strides) - 1
        self.base_feature_channels = base_feature_channels
        
        def _create_block(
            inc: int, outc: int, channels: Sequence[int], strides: Sequence[int], level: int, multires_input: bool
        ) -> nn.Module:
            """
            Builds the UNet structure from the bottom up by recursing down to the bottom block, then creating sequential
            blocks containing the downsample path, a skip connection around the previous block, and the upsample path.

            Args:
                inc: number of input channels.
                outc: number of output channels.
                channels: sequence of channels. Top block first.
            """
            c = channels[0]
            s = strides[0]
            # multi-resolution block
            multires_block = None
            down_inc = inc
            if multires_input:
                multires_feat_channel = inc if self.base_feature_channels is None else self.base_feature_channels
                multires_block = ConvolutionUnit(
                    spatial_dims = self.spatial_dims,
                    in_channels = self.in_channels,
                    out_channels = multires_feat_channel,
                    strides = 1,
                    kernel_size = self.kernel_size,
                    conv_only = True,
                    bias = False,
                    padding = same_padding(self.kernel_size, 1),
                    conv_op = self.block_conv_op,
                    conv_kwargs = self.block_conv_kwargs
                )
                down_inc = inc + multires_feat_channel
            # down block
            downblock = self._get_down_layer(down_inc, c, 1)
            # downsample
            downsample = self._get_downsample(c, channels[1], s)
            # upsample
            upsample = self._get_upsample(channels[1], channels[1], s)
            # upblock
            upc = inc + channels[1]
            upblock = self._get_up_layer(upc, outc, 1)
            # middle block (subblock)
            subblock: nn.Module
            if len(channels) > 2:
                subblock = _create_block(channels[1], channels[1], channels[1:], strides[1:], level + 1, self.multires_input) # continue recursion down
            else:
                subblock = self._get_bottom_layer(channels[1], channels[1])
                upc = c + channels[1]
            return self._get_connection_block(downblock, downsample, subblock, upsample, upblock, multires_block, level)
        self.model = _create_block(channels[0], channels[0], self.channels, self.strides, 0, False)
        self.initial_conv = ConvolutionUnit(
            spatial_dims = self.spatial_dims,
            in_channels = self.in_channels,
            out_channels = self.channels[0],
            strides = 1,
            kernel_size = self.kernel_size,
            conv_only = True,
            bias = False,
            padding = same_padding(self.kernel_size, 1),
            conv_op = self.block_conv_op,
            conv_kwargs = self.block_conv_kwargs
        )
    
    def _get_connection_block(self, downblock:nn.Module, downsample:nn.Module, subblock:nn.Module, upsample:nn.Module, upblock:nn.Module, multires_block: nn.Module, level: int)->nn.Module:
        """
        Returns the block object defining a layer of the UNet structure including the implementation of the skip
        between encoding (down) and decoding (up) sides of the network.

        Args:
            down_path: encoding half of the layer
            downsample:
            subblock
            upsample
            upblock
            up_path: decoding half of the layer
            subblock: block defining the next layer in the network.
        Returns: block for this layer: `nn.Sequential(down_path, SkipConnection(subblock), up_path)`
        """
        return ConnectionBlock(downblock, downsample, subblock, upsample, upblock, multires_block, level)
    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int)->nn.Module:
        """
        Returns the encoding (down) part of a layer of the network. This typically will downsample data at some point
        in its structure. Its output is used as input to the next layer down and is concatenated with output from the
        next layer to form the input for the decode (up) part of the layer.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
        """
        mod = nn.Module
        if self.block_mode == 'base':
            mod = ConvBlock(
                self.spatial_dims,
                in_channels,
                out_channels,
                strides = strides,
                kernel_size = self.kernel_size,
                subunits = self.num_units,
                adn_ordering = self.adn_ordering,
                act = self.act,
                norm = self.norm,
                dropout = self.dropout,
                bias = self.bias,
                conv_op = self.block_conv_op,
                conv_kwargs = self.block_conv_kwargs
            )
        elif self.block_mode == 'res':
            mod = ResBlock(
                self.spatial_dims,
                in_channels,
                out_channels,
                strides = strides,
                kernel_size = self.kernel_size,
                subunits = self.num_units,
                adn_ordering = self.adn_ordering,
                act = self.act,
                norm = self.norm,
                dropout = self.dropout,
                bias = self.bias,
                conv_op = self.block_conv_op,
                conv_kwargs = self.block_conv_kwargs
            )
        return mod
    def _get_bottom_layer(self, in_channels: int, out_channels: int)->nn.Module:
        """
        Returns the bottom or bottleneck layer at the bottom of the network linking encode to decode halves.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
        """
        return self._get_down_layer(in_channels, out_channels, 1)
    def _get_downsample(self, in_channels:int, out_channels: int, strides: int):
        downsample = DownsampleLayer(
            spatial_dims = self.spatial_dims,
            scale_factors = strides,
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = self.kernel_size,
            adn_ordering=self.adn_ordering,
            act = self.act,
            norm = self.norm,
            mode = self.downsample_mode,
            conv_op = self.downsample_conv_op,
            conv_kwargs = self.downsample_conv_kwargs,
            downsample_kwargs = self.downsample_kwargs
        )
        return downsample
    def _get_upsample(self, in_channels: int, out_channels: int, strides: int):
        upsample = UpsampleLayer(
            spatial_dims = self.spatial_dims,
            scale_factors = strides,
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = self.kernel_size,
            adn_ordering=self.adn_ordering,
            act = self.act,
            norm = self.norm,
            mode = self.upsample_mode,
            conv_op = self.upsample_conv_op,
            conv_kwargs = self.upsample_conv_kwargs,
            upsample_kwargs = self.upsample_kwargs
        )
        return upsample
    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int) -> nn.Module:
        """
        Returns the decoding (up) part of a layer of the network. This typically will upsample data at some point
        in its structure. Its output is used as input to the next layer up.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
        """
        mod: nn.Module
        if self.block_mode == 'base':
            mod = ConvBlock(
                self.spatial_dims,
                in_channels,
                out_channels,
                strides = 1,
                kernel_size = self.kernel_size,
                subunits = self.num_units,
                adn_ordering = self.adn_ordering,
                act = self.act,
                norm = self.norm,
                dropout = self.dropout,
                bias = self.bias,
                conv_op = self.block_conv_op,
                conv_kwargs = self.block_conv_kwargs
            )
        elif self.block_mode == 'res':
            mod = ResBlock(
                self.spatial_dims,
                in_channels,
                out_channels,
                strides = 1,
                kernel_size = self.kernel_size,
                subunits = self.num_units,
                adn_ordering = self.adn_ordering,
                act = self.act,
                norm = self.norm,
                dropout = self.dropout,
                bias = self.bias,
                conv_op = self.block_conv_op,
                conv_kwargs = self.block_conv_kwargs
            )
        return mod
    def forward(self, x_input: (dict, torch.Tensor)):
        if isinstance(x_input, torch.Tensor):
            x_input = {'level_0': x_input}
        x = x_input['level_0']
        list_levels = [f"level_{level+1}" for level in range(self.network_depth)]
        if self.multires_input:
            scale_factors = [1 for _ in range(self.spatial_dims)]
            for idx, target_level in enumerate(list_levels):
                scale_factors = [s1 / s2 for s1, s2 in zip(scale_factors, self.strides[idx])]
                if target_level not in x_input:
                    mode = 'trilinear' if self.spatial_dims == 3 else 'bilinear'
                    x_input[target_level] = F.interpolate(x_input['level_0'], scale_factor = scale_factors, mode = mode)
        x_feature = self.initial_conv(x)
        out = {}
        if self.deep_supervision:
            self.model(x_feature, out, x_input) # inplace operations with variable out
        else:
            out['level_0'] = self.model(x_feature, None, x_input)
        return out
    
class OutputGenerator(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int = 3,
        num_units: int = 1,
        act: 'tuple | str' = ('RELU', {'inplace': True}),
        norm: 'tuple | str' = 'INSTANCE',
        dropout: 'float' = 0.0,
        bias: 'bool' = True,
        adn_ordering: 'str' = 'NDA',
        last_conv_only: bool = True,
        conv_op: str = 'conv',
        conv_kwargs: dict = {},
        middle_channels: int = None
    ):
        super().__init__()
        middle_channels = in_channels if middle_channels is None else middle_channels
        if num_units > 1:
            self.add_module('feature_decoder', ConvBlock(spatial_dims, in_channels, middle_channels, 1, kernel_size, num_units-1, adn_ordering, act, norm, dropout, bias = bias, last_conv_only = False, conv_op = conv_op, conv_kwargs = conv_kwargs))
        self.add_module('last', ConvBlock(spatial_dims, middle_channels,out_channels, 1, kernel_size, 1, last_conv_only = last_conv_only, bias = bias, conv_op = conv_op, conv_kwargs = conv_kwargs ))
    def forward(self, x):
        for module in self:
            x = module(x)
        return x

class MultiresOutputGenerator(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        channels: list,
        out_channels: int,
        kernel_size: Sequence[int] | int = 3,
        num_units: int = 1,
        act: 'tuple | str' = ('RELU', {'inplace': True}),
        norm: 'tuple | str' = 'INSTANCE',
        dropout: 'float' = 0.0,
        bias: 'bool' = True,
        adn_ordering: 'str' = 'NDA',
        last_conv_only: bool = True,
        conv_op: str = 'conv',
        conv_kwargs: dict = {},
        middle_channels: int = None
    ):
        super().__init__()
        self.levels = [f"level_{l}" for l in range(len(channels))]
        mod_dict = {}
        for inc, level in zip(channels, self.levels):
            mod_dict[level] = OutputGenerator(
                spatial_dims = spatial_dims,
                in_channels = inc,
                out_channels = out_channels,
                kernel_size = kernel_size,
                num_units = num_units,
                act = act,
                norm = norm,
                dropout = dropout,
                bias = bias,
                adn_ordering = adn_ordering,
                last_conv_only = last_conv_only,
                conv_op = conv_op,
                conv_kwargs = conv_kwargs,
                middle_channels = middle_channels
            )
        self.mod_dict = nn.ModuleDict(mod_dict)
    def forward(self, x_dict: dict):
        out = {}
        for level in self.levels:
            out[level] = self.mod_dict[level](x_dict[level])
        return out

class MultitaskOutputGenerator(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        channels: list,
        list_task_config: list,
    ):
        super().__init__()
        self.levels = [f"level_{l}" for l in range(len(channels))]
        mod_dict = {}
        for task_config in list_task_config:
            mod_dict[task_config['task_name']] = MultiresOutputGenerator(
                spatial_dims = spatial_dims,
                channels = channels,
                **task_config['params']
            )
        self.mod_dict = nn.ModuleDict(mod_dict)
    def forward(self, x_dict: dict):
        out = {}
        for task_name in self.mod_dict.keys():
            out[task_name] = self.mod_dict[task_name](x_dict)
        return out

class UNetWrapper(nn.Module):
    def __init__(
        self,
        net_feature: nn.Module,
        net_out: nn.Module
    ):
        super().__init__()
        self.net_feature = net_feature
        self.net_out = net_out
    def forward(self, x):
        if self.net_feature.spatial_dims == 2:
            x = x.squeeze(1).permute(0,3,1,2)
        features = self.net_feature(x)
        out = self.net_out(features)
        if self.net_feature.spatial_dims == 2:
            for level in out.keys():
                out[level] = out[level].unsqueeze(1).permute(0,1,3,4,2)
        return out
    def predict(self, x, outkey_name = None):
        if self.net_feature.spatial_dims == 2:
            x = x.squeeze(1).permute(0,3,1,2)
        features = self.net_feature(x)
        out = self.net_out(features)
        if self.net_feature.spatial_dims == 2:
            for level in out.keys():
                out[level] = out[level].unsqueeze(1).permute(0,1,3,4,2)
        if outkey_name is not None:
            return out[outkey_name]['level_0']
        return out['level_0']

##########################################
# Pre-defined unet


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)

class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, spatial_dims: int = 3, num_feat=64, num_grow_ch=32, num_layers = 5):
        super(ResidualDenseBlock, self).__init__()
        self.spatial_dims = spatial_dims
        conv_module = Conv['conv', spatial_dims]
        layers = []
        for i in range(num_layers-1):
            layers.append(conv_module(num_feat + num_grow_ch * i, num_grow_ch, 3, 1, 1))
        layers.append(conv_module(num_feat + num_grow_ch * (i+1), num_feat, 3, 1, 1))

        # self.conv1 = conv_module(num_feat, num_grow_ch, 3, 1, 1)
        # self.conv2 = conv_module(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        # self.conv3 = conv_module(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        # self.conv4 = conv_module(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        # self.conv5 = conv_module(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights(layers, 0.1)
        self.layers = nn.Sequential(*layers)
        
        # default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        res = x
        x = [x]
        for layer in self.layers:
            x_ = torch.cat(x, 1)
            x.append(self.lrelu(layer(x_)))
        x = x[-1]
        return x * 0.2 + res
        # x1 = self.lrelu(self.conv1(x))
        # x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        # x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        # x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        # x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # # Empirically, we use 0.2 to scale the residual for better performance
        # return x5 * 0.2 + x

class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, spatial_dims, num_feat, num_grow_ch=32, n_rdb = 3, num_layers_rdb = 5):
        super(RRDB, self).__init__()
        rdbs = []
        for i in range(n_rdb):
            rdbs.append(ResidualDenseBlock(spatial_dims, num_feat, num_grow_ch, num_layers_rdb))
        self.rdbs = nn.Sequential(*rdbs)
        # self.rdb1 = ResidualDenseBlock(spatial_dims, num_feat, num_grow_ch)
        # self.rdb2 = ResidualDenseBlock(spatial_dims, num_feat, num_grow_ch)
        # self.rdb3 = ResidualDenseBlock(spatial_dims, num_feat, num_grow_ch)

    def forward(self, x):
        res = x
        for rdb in self.rdbs:
            x = rdb(x)
        return x * 0.2 + res
        # out = self.rdb1(x)
        # out = self.rdb2(out)
        # out = self.rdb3(out)
        # # Empirically, we use 0.2 to scale the residual for better performance
        # return out * 0.2 + x

class ChannelAttention(nn.Module):
    def __init__(self, spatial_dims, in_channels, reduction_ratio=16, use_max = True):
        super(ChannelAttention, self).__init__()
        self.use_max = use_max
        self.avg_pool = Pool['ADAPTIVEAVG', spatial_dims](1)
        self.max_pool = Pool['ADAPTIVEMAX', spatial_dims](1)
        self.fc = nn.Sequential(
            Conv['conv', spatial_dims](in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            Conv['conv', spatial_dims](in_channels // reduction_ratio, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        if self.use_max:
            max_out = self.fc(self.max_pool(x))
            out = avg_out + max_out
        else:
            out = avg_out
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    def __init__(self, spatial_dims, kernel_size=7, use_max = True):
        super(SpatialAttention, self).__init__()
        self.use_max = use_max
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        inc = 2 if use_max else 1
        self.conv1 = Conv['conv', spatial_dims](inc, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        if self.use_max:
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            out = torch.cat([avg_out, max_out], dim=1)
        else:
            out = avg_out
        out = self.conv1(out)
        return self.sigmoid(out) * x

class CBAM(nn.Module):
    def __init__(self, spatial_dims, in_channels, reduction_ratio=8, attention_kernel_size=7, use_max = True):
        super(CBAM, self).__init__()
        self.use_max = use_max
        self.channel_attention = ChannelAttention(spatial_dims, in_channels, reduction_ratio, use_max)
        self.spatial_attention = SpatialAttention(spatial_dims, attention_kernel_size, use_max)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class AttentionDenseUNet(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels,
        out_channels,
        base_channels: int = 16,
        deep_supervision: bool = False,
        multires_input: bool = False,
        act: (str, tuple) = ('leakyrelu', {'inplace': True}),
        downsample_mode = 'pixel_unshuffle',
        upsample_mode = 'pixel_shuffle',
        last_pixelshuffle: bool = False
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.deep_supervision = deep_supervision
        self.multires_input = multires_input
        c = base_channels
        conv_mod = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        self.feature_conv1 = nn.utils.spectral_norm(conv_mod(in_channels, c, 3, 1, 1))
        self.out_conv1 = conv_mod(c, out_channels, 3, 1, 1)
        if self.deep_supervision:
            self.out_conv2 = conv_mod(2*c, out_channels, 3, 1, 1)
            self.out_conv3 = conv_mod(4*c, out_channels, 3, 1, 1)
            self.out_conv4 = conv_mod(8*c, out_channels, 3, 1, 1)
        inc1 = c
        if downsample_mode == 'pixel_unshuffle':
            inc2 = c * (2 ** spatial_dims)
            inc3 = 2*c * (2 ** spatial_dims)
            inc4 = 4*c * (2 ** spatial_dims)
        else:
            inc2, inc3, inc4 = c, 2*c, 4*c
        if upsample_mode == 'pixel_shuffle':
            upc1 = 8*c // (2 ** spatial_dims) + 4*c
            upc2 = 4*c // (2 ** spatial_dims) + 2*c
            upc3 = 2*c // (2 ** spatial_dims) + c
        else:
            upc1 = 8*c + 4*c
            upc2 = 4*c + 2*c
            upc3 = 2*c + c
            if last_pixelshuffle:
                upc3 = 2*c // (2 ** spatial_dims) + inc2
        
        if multires_input:
            inc2 += c
            inc3 += c 
            inc4 += c
            self.feature_conv2 = nn.utils.spectral_norm(conv_mod(in_channels, c, 3, 1, 1))
            self.feature_conv3 = nn.utils.spectral_norm(conv_mod(in_channels, c, 3, 1, 1))
            self.feature_conv4 = nn.utils.spectral_norm(conv_mod(in_channels, c, 3, 1, 1))

        self.down_conv1 = nn.Sequential(
            RRDB(spatial_dims, inc1, 8, 3, 5),
            conv_mod(inc1, c, 3, 1, 1),
            get_act_layer(act),
        )
        self.down_conv2 = nn.Sequential(
            RRDB(spatial_dims, inc2, 16, 3, 5),
            conv_mod(inc2, 2*c, 3, 1, 1),
            get_act_layer(act),
        )
        self.down_conv3 = nn.Sequential(
            RRDB(spatial_dims, inc3, 32, 3, 5),
            conv_mod(inc3, 4*c, 3, 1, 1),
            get_act_layer(act),
        )
        self.down_conv4 = nn.Sequential(
            RRDB(spatial_dims, inc4, 32, 3, 5),
            conv_mod(inc4, 8*c, 3, 1, 1),
            get_act_layer(act),
        )
        
        self.up_conv1 = nn.Sequential(
            nn.utils.spectral_norm(conv_mod(upc1, 4*c, 3, 1, 1)),
            get_act_layer(act),
            nn.utils.spectral_norm(conv_mod(4*c, 4*c, 3, 1, 1)),
            get_act_layer(act),
            nn.utils.spectral_norm(conv_mod(4*c, 4*c, 3, 1, 1)),
            get_act_layer(act),
        )
        self.up_conv2 = nn.Sequential(
            nn.utils.spectral_norm(conv_mod(upc2, 2*c, 3, 1, 1)),
            get_act_layer(act),
            nn.utils.spectral_norm(conv_mod(2*c, 2*c, 3, 1, 1)),
            get_act_layer(act),
            nn.utils.spectral_norm(conv_mod(2*c, 2*c, 3, 1, 1)),
            get_act_layer(act),
        )
        self.up_conv3 = nn.Sequential(
            nn.utils.spectral_norm(conv_mod(upc3, c, 3, 1, 1)),
            get_act_layer(act),
            nn.utils.spectral_norm(conv_mod(c, c, 3, 1, 1)),
            get_act_layer(act),
            nn.utils.spectral_norm(conv_mod(c, c, 3, 1, 1)),
            get_act_layer(act),
        )
        
        if downsample_mode == 'pixel_unshuffle':
            self.downsample = [PixelUnshuffle(spatial_dims, 2), PixelUnshuffle(spatial_dims, 2), PixelUnshuffle(spatial_dims, 2), PixelUnshuffle(spatial_dims, 2)]
        elif downsample_mode in ('nearest', 'trilinear'):
            self.downsample = [
                lambda x: F.interpolate(x, scale_factor = 1/2, mode = downsample_mode), 
                lambda x: F.interpolate(x, scale_factor = 1/2, mode = downsample_mode), 
                lambda x: F.interpolate(x, scale_factor = 1/2, mode = downsample_mode)
            ]
        elif downsample_mode == 'conv':
            self.downsample = [
                conv_mod(c, c, 3, 2, 1),
                conv_mod(2*c, 2*c, 3, 2, 1),
                conv_mod(4*c, 4*c, 3, 2, 1)
            ]
        if upsample_mode == 'pixel_shuffle':
            self.upsample = [PixelShuffle(spatial_dims, 2), PixelShuffle(spatial_dims, 2), PixelShuffle(spatial_dims, 2)]
        elif upsample_mode in ('nearest', 'trilinear'):
            self.upsample = [
                lambda x: F.interpolate(x, scale_factor = 2, mode = upsample_mode), 
                lambda x: F.interpolate(x, scale_factor = 2, mode = upsample_mode), 
                lambda x: F.interpolate(x, scale_factor = 2, mode = upsample_mode)
            ]
            if last_pixelshuffle:
                self.upsample[-1] = PixelShuffle(spatial_dims, 2)
        # self.pixel_unshuffle = PixelUnshuffle(spatial_dims, 2)
        # self.pixel_shuffle = PixelShuffle(spatial_dims, 2)
        self.perception_mode = False

        # merge layers
        self.merge1 = CBAM(spatial_dims, upc1, 8)
        self.merge2 = CBAM(spatial_dims, upc2, 4)
        self.merge3 = CBAM(spatial_dims, upc3, 2)

    def forward(self, x: (torch.Tensor, dict)):
        return_features = self.perception_mode
        out = {}
        features = {}
        # Get initial feature outputs
        if isinstance(x, dict):
            x_dict = x
        else:
            x_dict = {'level_0': x}
        if self.spatial_dims == 2:
            x_dict = {key: val.squeeze(1).permute(0,3,1,2) for key,val in x_dict.items()}
        x = x_dict['level_0']
        feat1 = self.feature_conv1(x)
        if self.multires_input:
            interp_mode = 'bilinear' if self.spatial_dims == 2 else 'trilinear'
            if 'level_1' not in x_dict.keys():
                feat2 = self.feature_conv2(F.interpolate(x, scale_factor = 1/2, mode = interp_mode))
            else:
                feat2 = self.feature_conv2(x_dict['level_1'])
            if 'level_2' not in x_dict.keys():
                feat3 = self.feature_conv3(F.interpolate(x, scale_factor = 1/4, mode = interp_mode))
            else:
                feat3 = self.feature_conv3(x_dict['level_2'])
            if 'level_3' not in x_dict.keys():
                feat4 = self.feature_conv4(F.interpolate(x, scale_factor = 1/8, mode = interp_mode))
            else:
                feat4 = self.feature_conv4(x_dict['level_3'])
            
        # Encoding outputs
        x1 = self.down_conv1(feat1)
        x = self.downsample[0](x1)
        if self.multires_input:
            x = torch.cat([feat2, x], dim = 1)
        x2 = self.down_conv2(x)
        x = self.downsample[1](x2)
        if self.multires_input:
            x = torch.cat([feat3, x], dim = 1)
        x3 = self.down_conv3(x)
        x = self.downsample[2](x3)
        if self.multires_input:
            x = torch.cat([feat4, x], dim = 1)
        x = self.down_conv4(x)
        # Decoding outputs
        if self.deep_supervision:
            out['level_3'] = self.out_conv4(x)
        if return_features:
            features['level_3'] = x
        x = self.upsample[0](x)
        x = self.up_conv1(self.merge1(torch.cat([x, x3], dim = 1)))
        if self.deep_supervision:
            out['level_2'] = self.out_conv3(x)
        if return_features:
            features['level_2'] = x
        x = self.upsample[1](x)
        x = self.up_conv2(self.merge2(torch.cat([x, x2], dim = 1)))
        if self.deep_supervision:
            out['level_1'] = self.out_conv2(x)
        if return_features:
            features['level_1'] = x
        x = self.upsample[2](x)
        x = self.up_conv3(self.merge3(torch.cat([x, x1], dim = 1)))
        out['level_0'] = self.out_conv1(x)
        if self.spatial_dims == 2:
            for level in out.keys():
                out[level] = out[level].unsqueeze(1).permute(0,1,3,4,2)
        if return_features:# perception mode not supported 2D version
            features['level_0'] = x
        if return_features:
            output = {
                f"{name}_{level}": o[level]
                for name, o in zip(['feature', 'out'], [features, out])
                for level in o.keys()
            }
            return output
        return out
