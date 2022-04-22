import os
from yolox.data.datasets.voc_classes import VOC_CLASSES
import matplotlib.pyplot as plt
import classmap
import numpy as np
from sklearn.metrics import confusion_matrix


def get_confusion(gt, pr, clas, gt_counter):
    gt_values = np.array(list(gt_counter.values()))

    guess = gt
    fact = pr
    classes = clas
    fig = plt.figure(figsize=(10, 8))
    plt.gcf().subplots_adjust(bottom=0.35)
    # 归一化，预测统计量是预测统计总量与真实值的比，
    confusion = (confusion_matrix(guess, fact,labels=clas) / np.expand_dims(gt_values,0)) * 100

    plt.imshow(confusion, cmap=plt.cm.Blues)
    indices = range(len(confusion))

    plt.xticks(indices, classes, rotation=270)
    plt.yticks(indices, classes)
    plt.colorbar()
    plt.xlabel('gt')
    plt.ylabel('pr')
    for first_index in range(len(confusion)):

        for second_index in range(len(confusion[first_index])):
            if not confusion[first_index][second_index] <0.1:
                # gt_counter是某一类的统计次数，例如{'Go away':110},clas是类别{'Go away','Come here'},且有序，second_index是列，列表示gt
                gt_c = gt_counter[clas[second_index]]
                color = 'w' if first_index ==second_index else 'black'

                plt.text(first_index, second_index, format(confusion[first_index][second_index], '.1f'), fontsize=7, horizontalalignment='center', color=color)

    plt.show()

if __name__ == '__main__':
    cmap = classmap.MAP
    cmap = {v:k for k,v in cmap.items()}

    clas = VOC_CLASSES
    clas = clas[:-1]

    gt_res = []
    pr_res = []
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
                gt_res.append(gt_cla)
                gt_counter[gt_cla] += 1


        with open(pr_txt) as fpr:
            pr_line = fpr.readline()
            pr_cla = pr_line

            if pr_cla in clas:
                pr_res.append(pr_cla)
    print(gt_counter)
    get_confusion(gt_res, pr_res, clas,gt_counter)