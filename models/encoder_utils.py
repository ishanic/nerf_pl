import cv2
import numpy as np
import torch
from torchvision import transforms
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import functools
import math
import warnings

def get_norm_layer(norm_type="instance", group_norm_groups=32):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == "batch":
        norm_layer = functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True
        )
    elif norm_type == "instance":
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    elif norm_type == "group":
        norm_layer = functools.partial(nn.GroupNorm, group_norm_groups)
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def calc_same_pad_conv2d(t_shape, kernel_size=3, stride=1):
    in_height, in_width = t_shape[-2:]
    out_height = math.ceil(in_height / stride)
    out_width = math.ceil(in_width / stride)

    pad_along_height = max((out_height - 1) * stride + kernel_size - in_height, 0)
    pad_along_width = max((out_width - 1) * stride + kernel_size - in_width, 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    return pad_left, pad_right, pad_top, pad_bottom


def same_pad_conv2d(t, padding_type="reflect", kernel_size=3, stride=1, layer=None):
    """
    Perform SAME padding on tensor, given kernel size/stride of conv operator
    assumes kernel/stride are equal in all dimensions.
    Use before conv called.
    Dilation not supported.
    :param t image tensor input (B, C, H, W)
    :param padding_type padding type constant | reflect | replicate | circular
    constant is 0-pad.
    :param kernel_size kernel size of conv
    :param stride stride of conv
    :param layer optionally, pass conv layer to automatically get kernel_size and stride
    (overrides these)
    """
    if layer is not None:
        if isinstance(layer, nn.Sequential):
            layer = next(layer.children())
        kernel_size = layer.kernel_size[0]
        stride = layer.stride[0]
    return F.pad(
        t, calc_same_pad_conv2d(t.shape, kernel_size, stride), mode=padding_type
    )


def same_unpad_deconv2d(t, kernel_size=3, stride=1, layer=None):
    """
    Perform SAME unpad on tensor, given kernel/stride of deconv operator.
    Use after deconv called.
    Dilation not supported.
    """
    if layer is not None:
        if isinstance(layer, nn.Sequential):
            layer = next(layer.children())
        kernel_size = layer.kernel_size[0]
        stride = layer.stride[0]
    h_scaled = (t.shape[-2] - 1) * stride
    w_scaled = (t.shape[-1] - 1) * stride
    pad_left, pad_right, pad_top, pad_bottom = calc_same_pad_conv2d(
        (h_scaled, w_scaled), kernel_size, stride
    )
    if pad_right == 0:
        pad_right = -10000
    if pad_bottom == 0:
        pad_bottom = -10000
    return t[..., pad_top:-pad_bottom, pad_left:-pad_right]
