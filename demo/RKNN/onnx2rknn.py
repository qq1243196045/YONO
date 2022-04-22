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

    # rknn = RKNN(verbose=True)
    rknn = RKNN()

    rknn.list_devices()
    rknn.config()
    ret = rknn.load_onnx(model=ONNX_MODEL)
    ret = rknn.build(do_quantization=False,dataset="/home/bona/Projects/python/YOLOX-main/demo/RKNN/dataset.txt")
    ret = rknn.export_rknn(RKNN_MODEL)

    input_shape = tuple(map(int, "640,640".split(',')))
    origin_img = cv2.imread(cfg.image_path)

    # print(origin_img)



    img, ratio = preprocess(origin_img, input_shape,(0,1,2))


    # ret = rknn.init_runtime(target='rk3566',perf_debug=True, eval_mem=True)
    # ret = rknn.init_runtime(target='rk3566',perf_debug=True)
    ret = rknn.init_runtime(target='rk3566')
    rknn.get_sdk_version()

    # outputs = rknn.inference(inputs=[img],data_format='nchw')
    outputs = rknn.inference(inputs=[img])
    rknn.eval_perf()
    # rknn.eval_memory()



    # print(np.squeeze(outputs[0],axis=3).shape) # (1, 8400, 12, 1)


    predictions = demo_postprocess(np.squeeze(outputs[0],axis=3), input_shape)[0]
    # predictions = demo_postprocess(outputs[0], input_shape, p6=False)[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    boxes_xyxy /= ratio
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                         conf=cfg.score_thr, class_names=cfg.COCO_CLASSES)

    mkdir(cfg.output_dir)
    output_path = os.path.join(cfg.output_dir, cfg.image_path.split("/")[-1])
    print(output_path)
    cv2.imwrite(output_path, origin_img)
    cv2.imshow("show", origin_img)
    cv2.waitKey(0)

if __name__ == '__main__':
    run()


