import os
from yolox.data.datasets.voc_classes import VOC_CLASSES
import matplotlib.pyplot as plt
import classmap

if __name__ == '__main__':
    use_name_map = False
    VOC_CLASSES
    cmap = classmap.MAP
    cmap = {v:k for k,v in cmap.items()}

    clas = VOC_CLASSES
    clas = clas[:-1]


    gt_counter ={}
    for cla in clas:

        gt_counter[cla] = 0
    pr_counter = gt_counter.copy()
    acc_counter = gt_counter.copy()
    input_dir = 'input'
    gt_dir = os.path.join(input_dir, 'ground-truth-classification')

    pr_dir = os.path.join(input_dir, 'classification-results')
    txts = os.listdir(gt_dir)
    for txt in txts:
        gt_txt = os.path.join(gt_dir, txt)
        pr_txt = os.path.join(pr_dir, txt)

        with open(gt_txt) as fgt:
            gt_line = fgt.readline()

            gt_cla = gt_line
            if gt_cla in clas:
                gt_counter[gt_cla] += 1

        with open(pr_txt) as fpr:
            pr_line = fpr.readline()
            pr_cla =pr_line
            if pr_cla == gt_cla and pr_cla in clas:
                pr_counter[pr_cla] += 1

    for cla in clas:

        acc_counter[cla] = float(format(pr_counter[cla] / gt_counter[cla],'.4f'))
    sorted_acc = sorted(acc_counter.items(), key=lambda x: x[1])

    acc = sum(acc_counter.values())/len(acc_counter)
    acc = format(acc, '.4f')

    # fig = plt.figure(figsize=(10, 6))
    # # 设置bar底部占比
    # plt.gcf().subplots_adjust(bottom=0.35)
    # bar = plt.bar(acc_counter.keys(),acc_counter.values(),width=0.7)
    # plt.xticks(rotation=270)
    #
    # plt.bar_label(bar, label_type='edge')
    #
    # plt.show()
    keys = []
    values = []
    for k,v in sorted_acc:
        if use_name_map:
            keys.append(cmap[k])
        else:
            keys.append(k)
        values.append(v)
    fig = plt.figure(figsize=(16, 8))
    barh = plt.barh(keys, values)
    plt.bar_label(barh, label_type='edge')
    plt.title("acc of 20 classes without hand,mean:{}".format(str(acc).zfill(4)))
    plt.show()

