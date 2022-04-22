class Config():
    model = "/home/white/PycharmProjects/YOLOX-main/tools/yolox.onnx"
    # image_path = '/home/white/PycharmProjects/YOLOX-main/demo/RKNN/pad.jpg'
    image_path = '/mnt/data/datasets/1716/VOCdevkitmon/VOC2007/JPEGImages/000002.jpg'
    output_dir = 'demo_output'
    score_thr = 0.3
    input_shape = "640,640"

    COCO_CLASSES = (
        "dog_shit",
        'cloth',
        'shoes',
        'wires',
        'desk',
        'chair',
        'fan',
    )

