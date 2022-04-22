import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from yolox.data.data_augment import preproc as preprocess
from rknn.api import RKNN
import time
from yolox.utils import mkdir, multiclass_nms, demo_postprocess, vis
from config import Config
from tqdm import tqdm


def rknn_init():
    RKNN_MODEL = 'yolox.rknn'
    rknn = RKNN(verbose=True)
    rknn.load_rknn(RKNN_MODEL)
    ret = rknn.init_runtime(target='rk3566')
    rknn.get_sdk_version()
    print("初始化完成")
    return rknn

def rknn_disroty(rknn):
    rknn.release()
    print("销毁完成")


if __name__ == '__main__':
    cfg = Config()
    input_shape=(640,640)
    rknn = rknn_init()
    image_ids = open('/mnt/data/datasets/1716/VOCdevkitmon/VOC2007/ImageSets/Main/test.txt').read().strip().split()
    if not os.path.exists("./input"):
        os.makedirs("./input")
    if not os.path.exists("./input/detection-results"):
        os.makedirs("./input/detection-results")
    if not os.path.exists("./input/images-optional"):
        os.makedirs("./input/images-optional")

    for img_id in tqdm(image_ids):
        image_id = img_id.strip()
        img_path = "/mnt/data/datasets/1716/VOCdevkitmon/VOC2007/JPEGImages/"+image_id+".jpg"

        f = open("./input/detection-results/" + image_id + ".txt", "w")
        origin_img = cv2.imread(img_path)
        img, ratio = preprocess(origin_img, input_shape, (0, 1, 2))
        outputs = rknn.inference(inputs=[img])
        predictions = demo_postprocess(np.squeeze(outputs[0], axis=3), input_shape)[0]

        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.50, score_thr=0.01)
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                             conf=cfg.score_thr, class_names=cfg.COCO_CLASSES)
            for x0,y0,x1,y1,score,index in dets:
                score*=100

                classname = cfg.COCO_CLASSES
                predicted_class = classname[int(index)]
                f.write("%s %s %s %s %s %s\n" % (predicted_class, str(score/100)[:6], str(int(x0)), str(int(y0)), str(int(x1)),str(int(y1))))
        f.close()

        output_path = os.path.join("./input/images-optional", image_id)
        cv2.imwrite(output_path+'.jpg', origin_img)

    rknn_disroty(rknn)
    print("Conversion completed!")