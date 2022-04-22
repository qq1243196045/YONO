import cv2
from PIL import Image
import numpy as np

def vis2(img,img_info):
    h = img_info['height']
    w = img_info['width']
    hh = img_info['lheight']
    drow_line_height = int(h * hh)
    ptStart = (1, drow_line_height)
    ptEnd = (w-1, drow_line_height)



    point_color = (0, 255, 0)  # BGR
    thickness = 2
    lineType = 3
    cv2.line(img, ptStart, ptEnd, point_color, thickness, lineType)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    black_img = Image.new(img.mode, (w, int(h * 1.2)))
    black_img.paste(img, (0, 0))
    black_img = np.array(black_img)
    text = 'detecting...'
    txt_color = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    txt_size = cv2.getTextSize(text, font, 2, 2)[0]
    cv2.putText(black_img, text, (0, h + txt_size[1] + 3), font, 2, txt_color, thickness=4)
    return black_img
if __name__ == '__main__':
    img_path = r'D:\data\bona\JPEGImages\test\010289.jpg'
    img = cv2.imread(img_path)
    cv2.namedWindow('test_orl', cv2.WINDOW_NORMAL)

    # print(img.shape)
    img_info = {}
    img_info['width'] = img.shape[1]
    img_info['height'] = img.shape[0]
    img_info['lheight'] = 0.8
    img_info['status'] = 'initing'
    img = vis2(img,img_info)
    cv2.namedWindow('test',cv2.WINDOW_NORMAL)
    cv2.imshow("test",img)
    cv2.waitKey(0)