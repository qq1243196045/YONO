import os
import torch
from torchsummary import summary

from yolox.models.yolox import YOLOX
if __name__ == "__main__":
    # 需要使用device来指定网络在GPU还是CPU运行
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device  = torch.device('cuda' if torch.cuda.is_available()\
                               else 'cpu')
    m       = YOLOX().to(device)
    summary(m, input_size=(3, 416, 416))
