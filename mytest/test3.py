import torch
import cv2
img_path = '/data/datasets/BONA/VOCdevkit/VOC2007/JPEGImages/000002.jpg'
img=cv2.imread(img_path)
cv2.namedWindow("img",cv2.WINDOW_NORMAL)
cv2.rectangle(
            img,
            (0, 20),
            (500, 700),
            (0, 0, 0),
            -1
        )
cv2.imshow("img",img)
cv2.waitKey(0)