#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger
from uuid import uuid4
import cv2
import numpy as np
import torch
from PIL import Image
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

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
    parser.add_argument(
        "--use_black",
        default=False,
        help="Fuse conv and bn for testing.",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        # default= r"D:\Projects\Python\YOLOX-main\exps\example\yolox_voc\yolox_voc_s.py",
        default="/home/white/projects/python/YOLOX-double/exps/example/yolox_voc/yolox_voc_s.py",
        type=str,
        help="pls input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default="/home/white/projects/python/YOLOX-double/tools/YOLOX_outputs/yolox_voc_s/best_ckpt.pth", type=str, help="ckpt for eval")
    # parser.add_argument("-c", "--ckpt", default=r"D:\pth\dgr\yolox-s\best_ckpt.pth", type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.5, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
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

def vis3(img,img_info):
    h = img_info['height']
    w = img_info['width']
    status = img_info['status']
    if 'lheight' not in img_info:
        img_info['lheight'] = 0.8
    drow_line_height = int(img_info['lheight'] * h)

    if 'text' not in img_info:
        img_info['text'] = 'debug'
    text = img_info['text']
    ptStart = (1, drow_line_height)
    ptEnd = (w-1, drow_line_height)


    point_color = (0, 255, 0)  # BGR
    thickness = 2
    lineType = 3
    cv2.line(img, ptStart, ptEnd, point_color, thickness, lineType)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    black_img = Image.new(img.mode, (w, int(h * 1.1)))
    black_img.paste(img, (0, 0))
    black_img = np.array(black_img)
    if status == 'detected':

        txt_color = (255, 0, 0)
    else:
        txt_color = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    txt_size = cv2.getTextSize(text, font, 0.6, 2)[0]
    cv2.putText(black_img, text, (0, h + txt_size[1] + 3), font, 0.6, txt_color, thickness=2)
    black_img = cv2.cvtColor(black_img,cv2.COLOR_RGB2BGR)
    return black_img


def vis2(img, img_info):
    h = img_info['height']
    w = img_info['width']
    status = img_info['status']
    if 'lheight' not in img_info:
        img_info['lheight'] = 0.8
    drow_line_height = int(img_info['lheight'] * h)

    if 'text' not in img_info:
        img_info['text'] = 'debug'
    text = img_info['text']
    ptStart = (1, drow_line_height)
    ptEnd = (w - 1, drow_line_height)

    point_color = (0, 255, 0)  # BGR
    thickness = 2
    lineType = 3
    cv2.line(img, ptStart, ptEnd, point_color, thickness, lineType)
    # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # black_img = Image.new(img.mode, (w, int(h * 1.1)))
    # black_img.paste(img, (0, 0))
    # black_img = np.array(black_img)
    black_img = img
    if status == 'detected':

        txt_color = (0, 0, 255)
    else:
        txt_color = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX

    txt_size = cv2.getTextSize(text, font, 1, 1)[0]
    txt_bk_color = [0, 0, 0]
    # 画黑框
    cv2.rectangle(
        black_img,
        (int((w - txt_size[0]) / 2), h - 1 - txt_size[1]),
        (int((w + txt_size[0]) / 2) + 1, h - 1),
        txt_bk_color,
        -1
    )
    # 写白字
    cv2.putText(black_img, text, (int((w - txt_size[0]) / 2), h - 1), font, 1, txt_color, thickness=1)
    # black_img = cv2.cvtColor(black_img, cv2.COLOR_RGB2BGR)
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
            outputs,class_out = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info, class_out

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]

        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize

        bboxes /= ratio




        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        # 不使用copy的话，会报错Layout of the output array img is incompatible with cv::Mat (step[ndims-1] !
        img = img.copy()

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        cv2.namedWindow('after vis',cv2.WINDOW_NORMAL)
        cv2.namedWindow('after vis2',cv2.WINDOW_NORMAL)
        cv2.imshow('after vis', vis_res)
        if args.use_black:
            vis_res = vis3(vis_res, img_info)
        else:
            vis_res = vis2(vis_res, img_info)
        cv2.imshow("after vis2", vis_res)
        cv2.waitKey(1)
        return vis_res

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

def init(cap, predictor,vid_writer):
    print("start debug")
    dection_time = time.time()

    thrus = []
    ret_val, frame = cap.read()
    assert ret_val, "video can not get ,please check path:{}".format(args.path)
    start_time = time.time()
    while True:
        now_time = time.time()
        run_time = now_time - start_time
        if run_time > 3:
            break

        ret_val, frame = cap.read()

        if ret_val:
            outputs, img_info,class_out = predictor.inference(frame)
            # =====================分割算法============================================

            ratio = img_info['ratio']

            h = img_info['height']
            img_info['status'] = 'init'
            if outputs[0] is not None:
                for output in outputs[0]:

                    x1, y1, x2, y2, obj_conf, class_conf, class_pred = output
                    clas = predictor.cls_names[int(class_pred)]

                    # 如果目标是手的话，计算手的位置
                    if clas == "hand":
                        # x1 /= ratio
                        # y1 /= ratio
                        # x2 /= ratio
                        # y2 /= ratio

                        ddd = (y1 + y2) / (2 * ratio)

                        # ddd为物体位于窗口的高度位置，介于0-1之间，0为顶端，1为低端

                        lheight = (ddd.item() / h) * 0.9
                        thrus.append(lheight)
                img_info['lheight'] = lheight
                # img_info['text'] = 'During initialization, please put your hands on both sides of your body secend:{}'.format(str(3-run_time).zfill(2))
                img_info['text'] = 'init'

            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)


            if args.save_result:
                vid_writer.write(result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break

    assert len(thrus)!=0,'erro,thrus is none'
    print("thrus is ",thrus)
    ethrus = np.mean(thrus)
    return ethrus



def detection(ethrus, cap, predictor, vid_writer):
    """
    以静止手的数量确定运动
    :param ethrus: 阈值，高于该阈值则认为手没有抬起来
    :param cap: cv摄像头
    :param predictor: yolox推理器
    :param vid_writer: cv保存视频
    :return:
    """
    hand = 0
    buffer = []
    before_is_2 = 1  # 初始静止
    dection_time = time.time()
    dection_text = None
    wait_time = 2
    while True:
        now_time = time.time()
        time_after_detection = now_time - dection_time
        ret_val, frame = cap.read()
        if ret_val:
            # hand置为0，重新检测该帧
            hand = 0
            outputs, img_info, class_out = predictor.inference(frame)

            img_info['lheight'] = ethrus
            img_info['status'] = 'detecting'
            # img_info['text'] = 'debug'
            # =====================分割算法============================================

            ratio = img_info['ratio']

            h = img_info['height']

            if outputs[0] is not None:
                for output in outputs[0]:

                    x1, y1, x2, y2, obj_conf, class_conf, class_pred = output
                    clas = predictor.cls_names[int(class_pred)]

                    # 如果目标是手的话，计算手的位置
                    if clas == "hand":

                        y11 = y1/ratio
                        y22 = y2/ratio

                        heigth = (y11 + y22) / 2
                        # ddd为物体位于窗口的高度位置，介于0-1之间，0为顶端，1为低端
                        ddd = heigth / h
                        # 如果 高于阈值，认为手落下
                        if ddd > ethrus:
                            hand += 1


                    # 手抬起来了，处于动作阶段
                if hand <= 1:
                    before_is_2 = 0
                    buffer.append(img_info["raw_img"])

                    if time_after_detection > wait_time :
                        img_info['text'] = 'Action progress, waiting for the end'
                        img_info['status'] = 'detecting'
                    elif dection_text is None:
                        img_info['text'] = 'Nodection'
                        img_info['status'] = 'detecting'
                    else:
                        img_info['status'] = 'detected'
                        img_info['text'] = dection_text

                # 手落下了，动作结束
                else:
                    if time_after_detection > wait_time :
                        img_info['text'] = 'NO Dection'
                        img_info['status'] = 'detecting'
                    elif dection_text is None:
                        img_info['text'] = 'NO Dection'
                        img_info['status'] = 'detecting'
                    else:
                        img_info['text'] = dection_text
                        img_info['status'] = 'detected'


                    # 如果上一帧也是没有手，说明连续2真手在下面，认为动作确实结束

                    if before_is_2:
                        if len(buffer) > 20:
                            star = get_star_by_buffer(buffer)

                            outputs, star_img_info, class_out = predictor.inference(star)
                            # 不保存star_img_info,改个名字,这样保存的还是img_info
                            img_info['lheight'] = ddd
                            img = star_img_info['raw_img']
                            uid = str(uuid4())


                            # cv2.imwrite(uid + '.jpg',img)
                            dection_time = time.time()
                            class_pred, class_id = torch.max(class_out,1)
                            clas = predictor.cls_names[int(class_id)]
                            dection_text = 'Result:{};   confidence:{:.2f}'.format(clas,
                                                                               float(obj_conf * class_conf))

                            img_info['text'] = dection_text
                            img_info['status'] = 'detected'
                            # if outputs[0] is not None:
                            #     for output in outputs[0]:
                            #         x1, y1, x2, y2, obj_conf, class_conf, class_pred = output
                            #         clas = predictor.cls_names[int(class_pred)]
                            #
                            #         dection_text = 'Detection result:{}||{}; confidence:{:.2f}'.format(int(class_pred) + 1, clas,
                            #                                                            float(obj_conf * class_conf))
                            #         img_info['text'] = dection_text

                            # result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
                            # if args.save_result:
                            #     vid_writer.write(result_frame)
                            # ch = cv2.waitKey(1)
                            # if ch == 27 or ch == ord("q") or ch == ord("Q"):
                            #     break
                            # 清空buffer
                            buffer = []
                        else:
                            pass
                            # 动作在极低的位置进行，即使双手都低于阈值，也应该给一个机会，而不是清空缓存
                        # 动作刚结束的时候buffer有内容，执行手势检测
                        # 动作结束很久以后buffer没有内容，不需要执行手势检测，等待即可


                    before_is_2 = 1


            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_result:
                vid_writer.write(result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break


def imageflow_demo(predictor, vis_folder, current_time, args):
    buffer = []
    cap = cv2.VideoCapture(args.path)
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
    if args.use_black:
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height * 1.1))
        )
    else:
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )

    #=============初始化确定阈值======================
    ethrus = init(cap,predictor,vid_writer)
    print("ethrus is ",ethrus)

    #=============循环检测======================
    detection(ethrus, cap, predictor, vid_writer)




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
    img_id = "010435"
    img_path = "/data/datasets/BONA/VOCdevkit/VOC2007/JPEGImages/" + img_id + ".jpg"
    video_path = '/home/white/projects/python/YOLOX-double/tools/test2.mp4'
    args = make_parser(video_path).parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
