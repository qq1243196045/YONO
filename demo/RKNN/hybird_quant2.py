import numpy as np
import cv2
from rknn.api import RKNN


if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()

    # Build model
    print('--> hybrid_quantization_step2')
    ret = rknn.hybrid_quantization_step2(model_input='./yolox.model',
                                         data_input='./yolox.data',
                                         model_quantization_cfg='./yolox.quantization.cfg')
    if ret != 0:
        print('hybrid_quantization_step2 failed!')
        exit(ret)
    print('done')

    rknn.accuracy_analysis(inputs=['/mnt/data/datasets/1716/VOCdevkitmon/VOC2007/JPEGImages/000927.jpg'], output_dir=None)

    # Set inputs
    img = cv2.imread('/mnt/data/datasets/1716/VOCdevkitmon/VOC2007/JPEGImages/000927.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    np.save('./functions_hybrid_quant_0.npy', outputs[0])
    print('done')


    # Export rknn model
    print('--> Export RKNN model')
    ret = rknn.export_rknn('./ssd_mobilenet_v2.rknn')
    if ret != 0:
        print('Export model failed!')
        exit(ret)
    print('done')

    rknn.release()

