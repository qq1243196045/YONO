#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv,SPPBottleneck3


class YOLOPAFPN4(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark2", "dark3", "dark4", "dark5"),
        in_channels=[128, 256, 512, 1024],
        depthwise=False,
        act="silu",
        at=None,
        use_attention=False

    ):
        super().__init__()

        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act,attention=at,use_attention=use_attention)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[3] * width), int(in_channels[2] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer( # 1024 ->512
            int(2 * in_channels[2] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv( # cat后：512 ->256
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.reduce_conv2 = BaseConv(  # cat后：256 ->128
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer( # 512 ->256
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        self.C3_p2 = CSPLayer(  # 256 ->128
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv( # 256->256
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.bu_conv3 = Conv(  # 128->128
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[2] * width), int(in_channels[2] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[2] * width),
            int(in_channels[3] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """
        # print(input.shape)
        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x3, x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16

        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        f_out2 = self.C3_p3(f_out1)  # 512->256/8
        fpn_out2 = self.reduce_conv2(f_out2)
        f_out2 = self.upsample(fpn_out2)
        f_out2 = torch.cat([f_out2, x3], 1)
        fpn_out3 = self.C3_p2(f_out2)
        pan_out3 = fpn_out3
        p_out3 = fpn_out3
        p_out2 = self.bu_conv3(p_out3)
        pan_out2 = torch.cat([p_out2, fpn_out2], 1)

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32

        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out3, pan_out2, pan_out1, pan_out0)
        return outputs
class YOLOPAFPN(nn.Module):
    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
        use_attention=False,
        num_class=None,

    ):
        super().__init__()

        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act,attention='abplus',use_attention=use_attention,out_features=in_features,num_class=num_class)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )

        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        # self.dark6 = nn.Sequential(
        #     Conv(int(in_channels[2] * width), int(2 * in_channels[2] * width), 3, 2, act=act),
        #     SPPBottleneck3(int(2 * in_channels[2] * width), int(2 * in_channels[2] * width), activation=act),
        #     CSPLayer(
        #         int(2 * in_channels[2] * width),
        #         int(2 * in_channels[2] * width),
        #         n=round(3 * depth),
        #         shortcut=False,
        #         depthwise=depthwise,
        #         act=act,
        #     ),
        #
        # )
        # self.dark7 = nn.Sequential(
        #     Conv(int(in_channels[2] * width * 2), int(4 * in_channels[2] * width), 3, 2, act=act),
        #     SPPBottleneck3(int(2 * in_channels[2] * width), int(2 * in_channels[2] * width), activation=act),
        #     CSPLayer(
        #         int(2 * in_channels[2] * width),
        #         int(2 * in_channels[2] * width),
        #         n=round(3 * depth),
        #         shortcut=False,
        #         depthwise=depthwise,
        #         act=act,
        #     ),
        #
        # )
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(int(2 * in_channels[2] * width), int(num_class))

    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """


    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """
        # print(input.shape)
        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]


        [x2, x1, x0, fc_out] = features


        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32
        # dark6_in = torch.cat([p_out0, x0], 1) # 1024,20
        # dark6_out = self.dark6(dark6_in)
        # dark6_out = self.dark6(x0)

        # fc_out = self.avgpool(dark6_out)
        # fc_out = self.fc(torch.flatten(fc_out, 1))

        # fc_out = torch.softmax(fc_out, 1)

        outputs = (pan_out2, pan_out1, pan_out0, fc_out)
        return outputs

class YOLOPAFPN5(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
        use_attention=False,
        input_w=640,
        input_h=640

    ):
        super().__init__()

        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act,attention='abplus',use_attention=use_attention)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """
        # print(input.shape)
        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs