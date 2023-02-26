import numpy as np
import matplotlib.pyplot as plt
import itertools
import os

def draw_pic(datas, labs, name='1.jpg'):
    # 注释内容为加图例
    plt.figure()
    plt.cla()
    unque_labs = np.unique(labs)
    colors = [plt.cm.Spectral(each)
        for each in np.linspace(0, 1, len(unque_labs))
    ]
    # p = []
    # legends = []
    for i in range(len(unque_labs)):
        index = np.where(labs == unque_labs[i])
        pi = plt.scatter(
            datas[index, 0], datas[index, 1], c = [colors[i]]
        )
    #     p.append(pi)
    #     legends.append(unque_labs[i])
    # plt.legend(p, legends)
    plt.savefig(name)

def plot_confusion_matrix(cm, classes, normalize = False, 
    title = 'Confusion matrix', cmap = plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    else:
        cm = cm.astype('int')
    if len(classes) == 60:
        fig, ax = plt.subplots(figsize=(50, 50))
    elif len(classes) == 10:
        fig, ax = plt.subplots(figsize=(20, 15)) # [width, height]
    else:
        fig, ax = plt.subplots(figsize=(15, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title, fontsize= 25)
    # 坐标轴
    ax.figure.colorbar(im, ax=ax)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(np.arange(1, len(classes)+1), fontsize= 25)
    # ax.set_xticklabels(classes, fontsize= 25)
    ax.set_xlabel('Predicted Class', fontsize= 25)
    ax.set_ylabel('Actutal Class', fontsize= 25)
    # plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    ax.set_yticklabels(np.arange(1, len(classes)+1), fontsize= 25)
    # ax.set_yticklabels(classes, fontsize= 25)
    # 标数据
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max()/2
    # 因为shape[0]就是在y轴的数据，shape[1]就是在x轴的数据，所以要j,i
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        im.axes.text(j, i, format(cm[i,j], fmt),
            fontsize = 25,
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black"
        )
    return ax
    # fig = ax.get_figure()
    # fig.savefig(
    #     'test_result/conf_mat.png'
    # )