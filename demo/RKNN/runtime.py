import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN

RKNN_MODEL = 'yolox.rknn'
rknn = RKNN()
rknn.config(mean_values=[123.675, 116.28, 103.53], std_values=[58.82, 58.82, 58.82])
ret = rknn.build(do_quantization=True, dataset='./dataset.txt')
img = cv2.imread('/home/white/PycharmProjects/yolox-pytorch-main/1716/VOCdevkitmon/VOC2007/JPEGImages/001182.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
ret = rknn.init_runtime()
print(ret)
outputs = rknn.inference(inputs=[img])
print(outputs.shape)