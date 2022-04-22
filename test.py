from yolox.models import darknet
from yolox.models import  YOLOPAFPN
from yolox.models import YOLOX
import torch
out_features = ("dark2", "dark3", "dark4", "dark5", "avgpool", "fc")
self = darknet.CSPDarknet(1.0, 1.0, depthwise=False, act="relu",out_features=out_features)
x = torch.randn(1, 3, 640, 640)
# y= self(x)
# for i in range(len(y)):
#     print(y(i).shape)
self_fpn = YOLOPAFPN(1.0, 1.0, depthwise=False, act="relu", in_features=out_features)
yolo =YOLOX(backbone=self_fpn)
y= yolo(x)
for i in y:
    print(i.shape)

