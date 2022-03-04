import numpy as np
from scipy import ndimage
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

def conv3x3(in_planes, out_planes, stride=1, dilation=1, bias=False):
    "3x3 convolution with padding"

    kernel_size = np.asarray((3, 3))

    # Compute the size of the upsampled filter with
    # a specified dilation rate.
    upsampled_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size

    # Determine the padding that is necessary for full padding,
    # meaning the output spatial size is equal to input spatial size
    full_padding = (upsampled_kernel_size - 1) // 2

    # Conv2d doesn't accept numpy arrays as arguments
    full_padding, kernel_size = tuple(full_padding), tuple(kernel_size)

    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=full_padding,
        dilation=dilation,
        bias=bias,
    )

def conv1x1(in_planes, out_planes, stride=1, dilation=1, bias=False):
    "1x1 convolution with padding"

    kernel_size = np.asarray((1, 1))

    # Compute the size of the upsampled filter with
    # a specified dilation rate.
    upsampled_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size

    # Determine the padding that is necessary for full padding,
    # meaning the output spatial size is equal to input spatial size
    full_padding = (upsampled_kernel_size - 1) // 2

    # Conv2d doesn't accept numpy arrays as arguments
    full_padding, kernel_size = tuple(full_padding), tuple(kernel_size)

    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=full_padding,
        dilation=dilation,
        bias=bias,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()

        self.stride = stride
        self.dilation = dilation
        self.downsample = downsample

        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        # self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual

        out = self.relu(out)
        return out

class Interpolate(nn.Module):
    def __init__(
            self,
            size=None,
            scale_factor=None,
            mode="bilinear",
            align_corners=None,
    ):
        super().__init__()

        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(
            input, self.size, self.scale_factor, self.mode, self.align_corners
        )

class ResUBase:
    def __init__(self, n_input_channel=1):
        self.conv_down_1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "enc-conv0",
                        nn.Conv2d(
                            n_input_channel,
                            32,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                    ("enc-relu0", nn.ReLU(inplace=True)),
                    (
                        'enc-res1',
                        BasicBlock(
                            32, 32,
                            dilation=1
                        )
                    )
                ]
            )
        )
        self.conv_down_2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'enc-pool2',
                        nn.MaxPool2d(2)
                    ),
                    (
                        'enc-res2',
                        BasicBlock(
                            32, 64,
                            downsample=nn.Sequential(
                                nn.Conv2d(32, 64, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    )
                ]
            )
        )
        self.conv_down_4 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'enc-pool3',
                        nn.MaxPool2d(2)
                    ),
                    (
                        'enc-res3',
                        BasicBlock(
                            64, 128,
                            downsample=nn.Sequential(
                                nn.Conv2d(64, 128, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    )
                ]
            )
        )
        self.conv_down_8 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'enc-pool4',
                        nn.MaxPool2d(2)
                    ),
                    (
                        'enc-res4',
                        BasicBlock(
                            128, 256,
                            downsample=nn.Sequential(
                                nn.Conv2d(128, 256, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    )
                ]
            )
        )
        self.conv_down_16 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'enc-pool5',
                        nn.MaxPool2d(2)
                    ),
                    (
                        'enc-res5',
                        BasicBlock(
                            256, 512,
                            downsample=nn.Sequential(
                                nn.Conv2d(256, 512, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    ),
                    (
                        'enc-conv5',
                        nn.Conv2d(512, 256, kernel_size=1, bias=False)
                    )
                ]
            )
        )

        self.conv_up_8 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'dec-res1',
                        BasicBlock(
                            512, 256,
                            downsample=nn.Sequential(
                                nn.Conv2d(512, 256, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    ),
                    (
                        'dec-conv1',
                        nn.Conv2d(256, 128, kernel_size=1, bias=False)
                    )
                ]
            )
        )
        self.conv_up_4 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'dec-res2',
                        BasicBlock(
                            256, 128,
                            downsample=nn.Sequential(
                                nn.Conv2d(256, 128, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    ),
                    (
                        'dec-conv2',
                        nn.Conv2d(128, 64, kernel_size=1, bias=False)
                    )
                ]
            )
        )
        self.conv_up_2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'dec-res3',
                        BasicBlock(
                            128, 64,
                            downsample=nn.Sequential(
                                nn.Conv2d(128, 64, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    ),
                    (
                        'dec-conv3',
                        nn.Conv2d(64, 32, kernel_size=1, bias=False)
                    )
                ]
            )
        )
        self.conv_up_1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'dec-res1',
                        BasicBlock(
                            64, 32,
                            downsample=nn.Sequential(
                                nn.Conv2d(64, 32, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    )
                ]
            )
        )

class FCN(nn.Module, ResUBase):
    def __init__(self, n_input_channel=1, n_output_channel=1):
        super().__init__()
        ResUBase.__init__(self, n_input_channel)
        self.q_values = nn.Conv2d(32, n_output_channel, kernel_size=1, stride=1)

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

    def forward(self, obs):
        feature_map_1 = self.conv_down_1(obs)
        feature_map_2 = self.conv_down_2(feature_map_1)
        feature_map_4 = self.conv_down_4(feature_map_2)
        feature_map_8 = self.conv_down_8(feature_map_4)
        feature_map_16 = self.conv_down_16(feature_map_8)

        feature_map_up_8 = self.conv_up_8(torch.cat((feature_map_8,
                                                     F.interpolate(feature_map_16, size=feature_map_8.shape[-1],
                                                                   mode='bilinear', align_corners=False)), dim=1))
        feature_map_up_4 = self.conv_up_4(torch.cat((feature_map_4,
                                                     F.interpolate(feature_map_up_8, size=feature_map_4.shape[-1],
                                                                   mode='bilinear', align_corners=False)), dim=1))
        feature_map_up_2 = self.conv_up_2(torch.cat((feature_map_2,
                                                     F.interpolate(feature_map_up_4, size=feature_map_2.shape[-1],
                                                                   mode='bilinear', align_corners=False)), dim=1))
        feature_map_up_1 = self.conv_up_1(torch.cat((feature_map_1,
                                                     F.interpolate(feature_map_up_2, size=feature_map_1.shape[-1],
                                                                   mode='bilinear', align_corners=False)), dim=1))

        q_values = self.q_values(feature_map_up_1)
        return q_values
