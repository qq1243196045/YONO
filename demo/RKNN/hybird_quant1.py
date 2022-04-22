import numpy as np
import cv2
from rknn.api import RKNN

if __name__ == '__main__':
    ONNX_MODEL = 'yolox.onnx'
    RKNN_MODEL = 'yolox.rknn'
    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> config model')
    rknn.config(target_platform='rk3566')
    print('done')

    # Load tflite model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> hybrid_quantization_step1')
    ret = rknn.hybrid_quantization_step1(dataset="/home/bona/Projects/python/YOLOX-main/demo/RKNN/dataset.txt", proposal=False)
    if ret != 0:
        print('hybrid_quantization_step1 failed!')
        exit(ret)
    print('done')

    # Tips
    print('Please modify ssd_mobilenet_v2.quantization.cfg!')
    print('==================================================================================================')
    print('Modify Method: Fill the customized_quantize_layers with the output name & dtype of the custom layer.')
    print('')
    print('For example:')
    print('    custom_quantize_layers:')
    print('        FeatureExtractor/MobilenetV2/expanded_conv/depthwise/BatchNorm/batchnorm/add_1_rk:0: float16')
    print('        FeatureExtractor/MobilenetV2/expanded_conv/depthwise/Relu6:0: float16')
    print('Or:')
    print('    custom_quantize_layers: {')
    print('        FeatureExtractor/MobilenetV2/expanded_conv/depthwise/BatchNorm/batchnorm/add_1_rk:0: float16,')
    print('        FeatureExtractor/MobilenetV2/expanded_conv/depthwise/Relu6:0: float16,')
    print('    }')
    print('==================================================================================================')

    rknn.release()

