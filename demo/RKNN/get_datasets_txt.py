datasets_path = "/mnt/data/datasets/1716/VOCdevkitmon/VOC2007/ImageSets/Main/trainval.txt"
imgs_dir =  "/mnt/data/datasets/1716/VOCdevkitmon/VOC2007/JPEGImages/"
num_dats =200
with open(datasets_path,"r")as f:
    ids = f.readlines()
    with open("dataset.txt",'w')as ff:
        for i in range(num_dats):
            img_id = ids[i].strip()
            img_path = imgs_dir + img_id + ".jpg\n"
            ff.write(img_path)
