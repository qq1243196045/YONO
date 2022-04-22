#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger

import cv2
import numpy as np
import torch
from PIL import Image
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis, vis_clas

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser(image_path):
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "--demo", default="video", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default="yolo_s", help="model name")

    parser.add_argument(
        "--path", default=image_path, help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        default=True,
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        # default= r"D:\Projects\Python\YOLOX-main\exps\example\yolox_voc\yolox_voc_l.py",
        default="/home/white/projects/python/YOLOX-double/exps/example/yolox_voc/yolox_voc_s.py",
        type=str,
        help="pls input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt",
                        default="/home/white/projects/python/YOLOX-double/tools/YOLOX_outputs/yolox_voc_s/best_ckpt.pth",
                        type=str, help="ckpt for eval")
    # parser.add_argument("-c", "--ckpt", default=r"D:\Projects\Python\YOLOX-main\tools\YOLOX_outputs\yolox_voc_l\best_ckpt.pth", type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.5, type=float, help="test conf")
    parser.add_argument("--nms", default=0.5, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=640, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)


def clamp(x, min, max):
    if x < min:
        return min
    elif x > max:
        return max
    else:
        return x


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    map_text = []
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        cls = class_names[cls_id]
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])
        x0 = clamp(x0, 0, img.shape[1])
        x1 = clamp(x1, 0, img.shape[1])
        y0 = clamp(y0, 0, img.shape[0])
        y1 = clamp(y1, 0, img.shape[0])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        map_text.append(
            "%s %s %s %s %s %s\n" % (
            cls, str(float(score))[:6], str(int(x0)), str(int(y0)), str(int(x1)), str(int(y1))))
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img, map_text


def vis2(img, img_info):
    h = img_info['height']
    w = img_info['width']
    hh = img_info['lheight']
    img_info['text'] = 'debug'
    text = img_info['text']
    drow_line_height = int(h * hh)
    ptStart = (1, drow_line_height)
    ptEnd = (w - 1, drow_line_height)

    point_color = (0, 255, 0)  # BGR
    thickness = 2
    lineType = 3
    cv2.line(img, ptStart, ptEnd, point_color, thickness, lineType)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    black_img = Image.new(img.mode, (w, int(h * 1.2)))
    black_img.paste(img, (0, 0))
    black_img = np.array(black_img)

    txt_color = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    txt_size = cv2.getTextSize(text, font, 2, 2)[0]
    cv2.putText(black_img, text, (0, h + txt_size[1] + 3), font, 2, txt_color, thickness=4)
    return black_img


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            cls_names=COCO_CLASSES,
            trt_file=None,
            decoder=None,
            device="cpu",
            fp16=False,
            legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs, class_id = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )

            # logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info, class_id

    def visual(self, output, img_info, cls_conf=0.35, x_fc=None):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if x_fc is not None:
            pred, class_id = torch.max(x_fc,1)

            img = vis_clas(img, int(class_id), pred, self.cls_names, img_info)
        if output is None:
            return img, None
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res, text = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res, text


def get_star_by_buffer(buffer):
    w = buffer[0].shape[0]
    h = buffer[0].shape[1]
    length = len(buffer)
    frames = []
    M = np.zeros((w, h))
    Mx = np.zeros((w, h))
    My = np.zeros((w, h))
    for buf in buffer:
        frame = cv2.cvtColor(buf, cv2.COLOR_BGR2GRAY)
        frame = frame.astype(np.float64)
        frames.append(frame)

    for i in range(1, length // 3):
        flow_abs = abs(frames[i] - frames[i - 1])
        M += flow_abs * (i / length)

    for i in range(length // 3, 2 * (length // 3) + length % 3):
        flow_abs = abs(frames[i] - frames[i - 1])
        Mx += flow_abs * (i / length)
    for i in range(2 * (length // 3) + length % 3, length):
        flow_abs = abs(frames[i] - frames[i - 1])
        My += flow_abs * (i / length)

    star_rgb = np.array([M, Mx, My])
    star_rgb = star_rgb.transpose(1, 2, 0)
    min_star = np.min(star_rgb)
    max_star = np.max(star_rgb)
    star_rgb = ((star_rgb - min_star) * 255) / ((max_star - min_star) + 0.0001)
    star_rgb = star_rgb.astype(np.uint8)
    return star_rgb





def imageflow_demo(predictor, vis_folder, current_time, args):
    buffer = []

    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)

    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = os.path.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = os.path.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )

    # =============初始化======================
    dection_time = time.time()
    start_time = time.time()
    thrus = []
    ret_val, frame = cap.read()
    if not ret_val:
        print("video can not get ,please check path:{}".format(args.path))
        return
    while True:
        now_time = time.time()
        run_time = now_time - start_time
        if run_time > 3:
            break

        ret_val, frame = cap.read()

        if ret_val:
            # hand置为0，重新检测该帧
            hand = 0
            outputs, img_info, class_out = predictor.inference(frame)

            pred, class_id = torch.max(class_out, 1)
            class_name = predictor.cls_names[int(class_id)]

            # =====================分割算法============================================

            ratio = img_info['ratio']

            h = img_info['height']

            if outputs[0] is not None:

                for output in outputs[0]:

                    x1, y1, x2, y2, obj_conf, class_conf, class_pred = output
                    clas = predictor.cls_names[int(class_pred)]

                    # 如果目标是手的话，计算手的位置
                    if clas == "hand":
                        heigth = ((y1 / ratio) + (y2 / ratio)) / 2

                        # ddd为物体位于窗口的高度位置，介于0-1之间，0为顶端，1为低端
                        ddd = heigth / (h + 0.001)

                        thrus.append(ddd.item())
                img_info['lheight'] = ddd
                img_info[
                    'text'] = 'During initialization, please put your hands on both sides of your body secend:{}'.format(
                    str(3 - run_time).zfill(2))

            result_frame, text = predictor.visual(outputs[0], img_info, predictor.confthre,x_fc=class_out)


            if args.save_result:
                vid_writer.write(result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break

    if len(thrus)==0:
        ethrus = 0.6
        print("No detection hand ,use default,Warm!,default is very unuseful,please restart !")
    else:
        ethrus = np.mean(thrus)
    print("~~~~")
    print(ethrus)

    # =============初始化结束======================
    # =============循环检测======================

    hand = 0
    while True:
        now_time = time.time()
        time_after_detection = now_time - dection_time
        ret_val, frame = cap.read()
        if ret_val:
            # hand置为0，重新检测该帧
            hand = 0
            outputs, img_info, class_out = predictor.inference(frame)
            pred, class_id = torch.max(class_out, 1)
            img_info['lheight'] = ethrus
            # =====================分割算法============================================

            ratio = img_info['ratio']

            h = img_info['height']
            class_name = predictor.cls_names[int(class_id)]
            # if class_name == 'hand':
            if outputs[0] is not None:
                for output in outputs[0]:
                    x1, y1, x2, y2, obj_conf, class_conf, class_pred = output
                    clas = predictor.cls_names[int(class_pred)]
                    # 如果目标是手的话，计算手的位置
                    if clas == "hand":

                        heigth = ((y1 / ratio) + (y2 / ratio)) / 2
                        # ddd为物体位于窗口的高度位置，介于0-1之间，0为顶端，1为低端
                        ddd = heigth / h
                        print("ddd is ", ddd)
                        # 如果 高于阈值，认为手已经抬起来
                        if ddd < ethrus:
                            hand += 1

                    # 手抬起来了，处于动作阶段
                    if hand >= 1:
                        buffer.append(img_info["raw_img"])
                        if time_after_detection > 1:
                            img_info['text'] = 'In action, detecting...'
                    # 手落下了，动作结束
                    else:
                        # 动作刚结束的时候buffer有内容，执行手势检测
                        # 动作结束很久以后buffer没有内容，不需要执行手势检测，等待即可
                        if len(buffer) > 10:
                            star = get_star_by_buffer(buffer)
                            outputs, img_info, class_out = predictor.inference(star)
                            pred, class_id = torch.max(class_out, 1)
                            class_name = predictor.cls_names[int(class_id)]
                            img_info['lheight'] = ddd
                            dection_time = time.time()
                            img_info['text'] = 'detection result:{} confidence:{}'.format(class_name,
                                                                                          str(pred).zfill(
                                                                                              4))
                            result_frame, text = predictor.visual(outputs[0], img_info, predictor.confthre)
                            if args.save_result:
                                vid_writer.write(result_frame)
                            ch = cv2.waitKey(1)
                            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                                break
                        # 清空buffer
                        buffer = []

                result_frame, text = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_result:
                vid_writer.write(result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break


# =============循环检测结束======================

def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, COCO_CLASSES, trt_file, decoder, args.device, args.fp16, args.legacy)
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, args.path, current_time, args.save_result)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # img_id = "001308"
    # img_path = "/home/bona/Documents/datasets/1716/VOCdevkitmon/VOC2007/JPEGImages/" + img_id + ".jpg"
    # img_id = "010435"
    # img_path = "/data/datasets/BONA/VOCdevkitmon/VOC2007/JPEGImages/" + img_id + ".jpg"
    # img_path = r"C:\Users\white\Desktop\51fbe916-3877-4325-9d74-29382e20d72b.jpg"
    video_path = '/home/white/projects/python/YOLOX-double/tools/test2.mp4'
    # video_path = r'D:\Projects\Python\YOLOX-main\tools\demo\test6.avi'
    args = make_parser(video_path).parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
