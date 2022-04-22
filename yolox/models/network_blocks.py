#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Psigmoid(nn.Module):
    def __init__(self):
        super(Psigmoid, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(1, dtype=torch.float))
        self.beta = nn.Parameter(torch.tensor(-1, dtype=torch.float))

    def forward(self, x):
        x = self.alpha/(1+torch.exp(self.beta*x))
        return x


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == "sigmoid":
        module = nn.Sigmoid()
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

        # self.se = SELayer(out_channels, 16)
    def forward(self, x):
        y = self.conv2(self.conv1(x))
        # y = self.se(y)
        if self.use_add:
            y = y + x
        return y


class ResLayer(nn.Module):
    "Residual layer with `in_channels` inputs."

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, act="lrelu"
        )
        self.layer2 = BaseConv(
            mid_channels, in_channels, ksize=3, stride=1, act="lrelu"
        )
        self.se = SELayer(in_channels, 16)

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        # out = self.se(out)
        return x + out


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"

    ):
        # print("in_channels is :", in_channels) 512
        # print("out_channels is :", out_channels) 512
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self,x):

        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x
# class SPPBottleneck3(nn.Module):
#     """Spatial pyramid pooling layer used in YOLOv3-SPP"""
#
#     def __init__(
#         self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
#     ):
#         super().__init__()
#         hidden_channels = in_channels // 2
#         self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
#         self.max1_1 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
#         self.max1_2 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
#         self.max2_1 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
#         self.max2_2 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
#         self.max2_3 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
#         self.max2_4 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
#         self.max3_1 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
#         self.max3_2 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
#         self.max3_3 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
#         self.max3_4 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
#         self.max3_5 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
#         self.max3_6 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
#         conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
#         self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         max1_1_out = self.max1_1(x)
#         max1_2_out = self.max1_2(max1_1_out)
#         max2_1_out = self.max2_1(x)
#         max2_2_out = self.max2_2(max2_1_out)
#         max2_3_out = self.max2_3(max2_2_out)
#         max2_4_out = self.max2_4(max2_3_out)
#         max3_1_out = self.max3_1(x)
#         max3_2_out = self.max3_2(max3_1_out)
#         max3_3_out = self.max3_3(max3_2_out)
#         max3_4_out = self.max3_4(max3_3_out)
#         max3_5_out = self.max3_5(max3_4_out)
#         max3_6_out = self.max3_6(max3_5_out)
#         x = torch.cat([x]+[max1_2_out,max2_4_out,max3_6_out],dim=1)
#         x = self.conv2(x)
#         return x

class SPPBottleneck2(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.max1 = nn.MaxPool2d(kernel_size=5,stride=1,padding=2)
        self.max2 = nn.MaxPool2d(kernel_size=9,stride=1,padding=4)
        self.max3 = nn.MaxPool2d(kernel_size=13,stride=1,padding=6)
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        max1_out = self.max1(x)
        max2_out = self.max2(x)
        max3_out = self.max3(x)
        x = torch.cat([x]+[max1_out,max2_out,max3_out],dim=1)
        # x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x
class SPPBottleneck3(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)

        self.max1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
        )
        self.max2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        )
        self.max3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        )

        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):

        x = self.conv1(x)
        max1_out = self.max1(x)
        max2_out = self.max2(x)
        max3_out = self.max3(x)
        x = torch.cat([x]+[max1_out,max2_out,max3_out],dim=1)
        # x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)

        return x

class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
        input_w=None,
        input_h=None,
        use_attention=False
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.input_w = input_w,
        self.input_h = input_h,
        self.use_attention = use_attention
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

        if use_attention:
            # self.ca = CoordAttention(hidden_channels)
            # self.cbam = CBAM(hidden_channels)
            self.se = SELayer(hidden_channels, 1)
    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        if self.use_attention:
        #     self.ca(x_1)
            x_1 = self.se(x_1)
        #     x_1 = self.cbam(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"): #in 3 out 64
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act) #in 12 out 64

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)

class Focus2(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=3, stride=2, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels , out_channels, ksize, stride, act=act)# in 3 out 64

    def forward(self, x):

        return self.conv(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)





class ABPlus(nn.Module):
    def __init__(self, channel, reduction=None):
        super(ABPlus, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = BaseConv(channel, channel, 1, 1, act="sigmoid")# in 3 out 64

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.conv(y)
        return x * y.expand_as(x)

class AB(nn.Module):
    def __init__(self, channel, reduction=None):
        super(AB, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dconv = nn.Conv2d(in_channels=channel, out_channels=channel,kernel_size=1,stride=1,groups=channel)
        # self.wconv = nn.Conv2d(in_channels=channel, out_channels=channel,kernel_size=1,stride=1,groups=1)
        self.bn = nn.BatchNorm2d(channel)
        self.act = get_activation('sigmoid')
        # self.conv = DWConv(channel, channel, 1, 1, act="sigmoid")# in 3 out 64

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.act(self.bn(self.dconv(y)))
        return x * y.expand_as(x)

class DSELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(DSELayer, self).__init__()
        self.se = SELayer(channel=channel,reduction=reduction)

    def forward(self, x):
        pass


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        # 使用自适应池化缩减map的大小，保持通道不变
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # map尺寸不变，缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel,reduction=None):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAttention(nn.Module):

    def __init__(self, channel, reduction=16):
        super(CoordAttention, self).__init__()
        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        temp_c = max(8, channel // reduction)
        self.conv1 = nn.Conv2d(channel, temp_c, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(temp_c)
        self.act1 = h_swish()

        self.conv2 = nn.Conv2d(temp_c, channel, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(temp_c, channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        short = x
        n, c, H, W = x.shape
        x_h, x_w = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2)
        x_cat = torch.cat([x_h, x_w], dim=2)
        out = self.act1(self.bn1(self.conv1(x_cat)))
        x_h, x_w = torch.split(out, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        out_h = torch.sigmoid(self.conv2(x_h))
        out_w = torch.sigmoid(self.conv3(x_w))
        return short * out_w * out_h

class SCSEModule(nn.Module):
    def __init__(self, ch, re=16):
        super().__init__()
        self.cSE = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(ch,ch//re,1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(ch//re,ch,1),
                                 nn.Sigmoid())
        self.sSE = nn.Sequential(nn.Conv2d(ch,ch,1),
                                 nn.Sigmoid())

    class DoubleAtten(nn.Module):
        """
        A2-Nets: Double Attention Networks. NIPS 2018
        """

        def __init__(self, in_c):
            super(DoubleAtten, self).__init__()
            self.in_c = in_c
            """Convolve the same input feature map to produce three feature maps with the same scale, i.e., A, B, V (as shown in paper).
            """
            self.convA = nn.Conv2d(in_c, in_c, kernel_size=1)
            self.convB = nn.Conv2d(in_c, in_c, kernel_size=1)
            self.convV = nn.Conv2d(in_c, in_c, kernel_size=1)

        def forward(self, input):
            feature_maps = self.convA(input)
            atten_map = self.convB(input)
            b, _, h, w = feature_maps.shape

            feature_maps = feature_maps.view(b, 1, self.in_c, h * w)  # reshape A
            atten_map = atten_map.view(b, self.in_c, 1, h * w)  # reshape B to generate attention map
            global_descriptors = torch.mean((feature_maps * F.softmax(atten_map, dim=-1)),
                                            dim=-1)  # Multiply the feature map and the attention weight map to generate a global feature descriptor

            v = self.convV(input)
            atten_vectors = nn.F.softmax(v.view(b, self.in_c, h * w), dim=-1)  # 生成 attention_vectors
            out = torch.bmm(atten_vectors.permute(0, 2, 1), global_descriptors).permute(0, 2, 1)

            return out.view(b, _, h, w)
    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

#==============================NAM======================================
class Channel_Att(nn.Module):
    def __init__(self, channels, t=16):
        super(Channel_Att, self).__init__()
        self.channels = channels
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)

    def forward(self, x):
        residual = x
        x = self.bn2(x)
        # 式2的计算，即Mc的计算
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.sigmoid(x) * residual  #

        return x


class Att(nn.Module):

    def __init__(self, channels):
        super(Att, self).__init__()
        self.Channel_Att = Channel_Att(channels)


    def forward(self, x):
        x_out1 = self.Channel_Att(x)

        return x_out1

#==============================NAM======================================
if __name__ == '__main__':
    from torchsummary import summary
    c =16

    x = torch.randn(4, c, 128, 64)  # b, c, h, w
    print("↓↓↓↓↓↓↓↓↓↓↓↓↓||NAM||↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓")
    at = Att(c)
    summary(at, input_size=(c, 640, 640), device='cpu')

    x = torch.randn(4, c, 128, 64)  # b, c, h, w
    print("↓↓↓↓↓↓↓↓↓↓↓↓↓||CA||↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓")
    at = CoordAttention(c,  16)
    summary(at,input_size=(c,640,640),device='cpu')

    print("↓↓↓↓↓↓↓↓↓↓↓↓↓||CBAM||↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓")
    at = CBAM(c)
    summary(at, input_size=(c, 640, 640), device='cpu')

    print("↓↓↓↓↓↓↓↓↓↓↓↓↓||ABPLUS||↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓")
    at = ABPlus(c)
    summary(at, input_size=(c, 640, 640), device='cpu')

    print("↓↓↓↓↓↓↓↓↓↓↓↓↓||AB||↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓")
    at = AB(c)
    summary(at, input_size=(c, 640, 640), device='cpu')

    print("↓↓↓↓↓↓↓↓↓↓↓↓↓||SE16||↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓")
    at = SELayer(c,16)
    summary(at, input_size=(c, 640, 640), device='cpu')

    print("↓↓↓↓↓↓↓↓↓↓↓↓↓||SE8||↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓")
    at = SELayer(c, 8)
    summary(at, input_size=(c, 640, 640), device='cpu')

    print("↓↓↓↓↓↓↓↓↓↓↓↓↓||SE4||↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓")
    at = SELayer(c, 4)
    summary(at, input_size=(c, 640, 640), device='cpu')

    print("↓↓↓↓↓↓↓↓↓↓↓↓↓||SE2||↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓")
    at = SELayer(c, 2)
    summary(at, input_size=(c, 640, 640), device='cpu')

    print("↓↓↓↓↓↓↓↓↓↓↓↓↓||SE1||↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓")
    at = SELayer(c, 1)
    summary(at, input_size=(c, 640, 640), device='cpu')