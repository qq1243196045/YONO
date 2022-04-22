#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch.nn as nn

from .yolo_head import YOLOXHead,YOLOXHead4
from .yolo_pafpn import YOLOPAFPN,YOLOPAFPN4


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None, p=3):
        super().__init__()
        if backbone is None:
            if p==3:
                backbone = YOLOPAFPN(use_attention=False)
            elif p==4:
                backbone = YOLOPAFPN4(at='cbam',use_attention=True)

        if head is None:
            if p==3:
                head = YOLOXHead(80)
            elif p==4:
                head = YOLOXHead4(80)
        # self.training = False
        self.backbone = backbone
        self.head = head


    def forward(self, x, targets=None,class_id=None):

        # fpn output content features of [dark3, dark4, dark5] (pan_out2, pan_out1, pan_out0,fc)
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            assert class_id is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg,class_loss = self.head(
                fpn_outs, targets, x, class_id
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
                "class_loss": class_loss
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs

