#----------------------------------------------------#
#   获取测试集的ground-truth
#   具体视频教程可查看
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
import sys
import os
import glob
import xml.etree.ElementTree as ET
from yolox.data.datasets.voc_classes import VOC_CLASSES
'''
！！！！！！！！！！！！！注意事项！！！！！！！！！！！！！
# 这一部分是当xml有无关的类的时候，下方有代码可以进行筛选！
'''
#---------------------------------------------------#
#   获得类
#---------------------------------------------------#

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import cv2 as cv
import os
from tqdm import tqdm

def show_object_name(image: np.ndarray, name: str, p_tl,thickness=1):
    font = cv.FONT_HERSHEY_SIMPLEX
    # cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)
    return cv.putText(image, name, p_tl, font, 1, (255, 255, 255),thickness=thickness)

def show_object_rect(image: np.ndarray, bndbox):
    pt1 = bndbox[:2]
    pt2 = bndbox[2:]
    image_show = image
    return cv.rectangle(image_show, pt1, pt2, (0, 255, 255), 2)

def find(root, name):
    vas = root.find(name)
    return vas
def findall(root,name):
    vas = root.findall(name)
    return vas

def getxxyyns(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objs = (findall(root, 'object'))
    xxyyns = []
    for obj in objs:
        bndbox = find(obj, 'bndbox')
        name = find(obj, 'name').text
        xmin = int(find(bndbox, 'xmin').text)
        xmax = int(find(bndbox, 'xmax').text)
        ymin = int(find(bndbox, 'ymin').text)
        ymax = int(find(bndbox, 'ymax').text)
        xxyyns.append((xmin, xmax, ymin, ymax, name))
    return xxyyns

if __name__ == '__main__':
    import config
    base = config.datasets

    datas_dir = base
    save_dir = '/home/white/projects/python/YOLOX-double/tools/YOLOX_outputs/yolox_voc_s/vis_res/2022_04_22'
    img_dir = datas_dir + '/VOCdevkit/VOC2007/JPEGImages'
    xml_dir = datas_dir + '/VOCdevkit/VOC2007/Annotations'

    image_ids = open(datas_dir + "/VOCdevkit/VOC2007/ImageSets/Main/test.txt").read().strip().split()

    if not os.path.exists("./input"):
        os.makedirs("./input")
    if not os.path.exists("./input/ground-truth"):
        os.makedirs("./input/ground-truth")
    if not os.path.exists("./input/ground-truth-classification"):
        os.makedirs("./input/ground-truth-classification")

    for image_id in tqdm(image_ids):
        img_name = image_id + '.jpg'
        orl_img_path = os.path.join(img_dir, img_name)
        pre_img_path = os.path.join(save_dir, img_name)
        oral_img = cv.imread(orl_img_path)

        pre_img = cv.imread(pre_img_path)

        assert pre_img is not None, pre_img_path

        with open("./input/ground-truth/" + image_id + ".txt", "w") as new_f:
            with open("./input/ground-truth-classification/" + image_id + ".txt", "w") as new_fc:
                root = ET.parse(base + "/VOCdevkit/VOC2007/Annotations/" + image_id + ".xml").getroot()
                size = root.find('size')
                class_type = root.find('class_type').text

                width = int(size.find('width').text)
                height = int(size.find('height').text)
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult') != None:
                        difficult = obj.find('difficult').text
                        if int(difficult) == 1:
                            difficult_flag = True
                    obj_name = obj.find('name').text

                    '''
                    ！！！！！！！！！！！！注意事项！！！！！！！！！！！！
                    # 这一部分是当xml有无关的类的时候，可以取消下面代码的注释
                    # 利用对应的classes.txt来进行筛选！！！！！！！！！！！！
                    '''
                    class_names = VOC_CLASSES
                    if obj_name not in class_names:
                        print("obj is :{}".format(obj_name))
                        print("continue")
                        continue

                    bndbox = obj.find('bndbox')
                    name = obj.find(('name')).text
                    left = bndbox.find('xmin').text
                    top = bndbox.find('ymin').text
                    right = bndbox.find('xmax').text
                    bottom = bndbox.find('ymax').text


                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))



                    oral_img = show_object_rect(oral_img, (int(left), int(top), int(right), int(bottom)))
                    oral_img = show_object_name(oral_img, name, (int(left), int(top)))
                new_fc.write(str(class_type))

            show_img = show_object_name(oral_img, "groundtrue", (0, 20),thickness=2)

            pre_img = show_object_name(pre_img, "predict", (0, 20), thickness=2)


            orl_img = Image.fromarray(cv.cvtColor(show_img,cv.COLOR_BGR2RGB))

            pre_img = Image.fromarray(cv.cvtColor(pre_img,cv.COLOR_BGR2RGB))


            save_img = Image.new(pre_img.mode,(width*2,height))


            save_img.paste(orl_img,(0,0))
            save_img.paste(pre_img,(width,0))

            save_name = "(compare)" + img_name
            save_path = os.path.join(save_dir,save_name)

            save_img.save(save_path)


                # cv.imwrite(save_path,show_img)

    print("Conversion completed!")



















