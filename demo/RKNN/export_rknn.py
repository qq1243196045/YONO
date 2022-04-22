import os
import urllib
import traceback
import time
import sys
import re
import numpy as np
import cv2
from yolox.data.data_augment import preproc as preprocess
from rknn.api import RKNN
import time
from yolox.utils import mkdir, multiclass_nms, demo_postprocess, vis
from config import Config

def run():
    cfg = Config()
    ONNX_MODEL = 'yolox.onnx'
    RKNN_MODEL = 'yolox.rknn'

    rknn = RKNN(verbose=True)
    rknn.list_devices()
    # rknn.config(target_platform='rk3566')
    rknn.config()
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret < 0:
        print("load faild")
    ret = rknn.build(do_quantization=False,dataset="/home/bona/Projects/python/YOLOX-main/demo/RKNN/dataset.txt")
    if ret < 0:
        print("build faild")
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret < 0:
        print("export faild")
    print("export down")