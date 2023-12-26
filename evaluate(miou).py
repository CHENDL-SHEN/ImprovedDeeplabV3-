#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
可计算miou，是我自己找的；可以计算出每个类的iou；
和继忠给的py计算结果一样，但是和原文相比都差 0.3%；

"""

import os
import numpy as np
import argparse
import json
from PIL import Image
from os.path import join


# 设标签宽W，长H
def fast_hist(a, b, n):  # a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的标签，形状(H×W,)；n是类别数目，实数（在这里为19）
    '''
	核心代码
	'''
    k = (a >= 0) & (a < n) & (a != 255)  # k是一个一维bool数组，形状(H×W,)；目的是找出标签中需要计算的类别（去掉了背景） k=0或1

    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)  # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)


def per_class_iu(hist):  # 分别为每个类别（在这里是19类）计算mIoU，hist的形状(n, n)
    '''
	核心代码
	'''
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))  # 矩阵的对角线上的值组成的一维数组/矩阵的所有元素之和，返回值形状(n,)

def get_tp_fp_tn_fn(hist):
    tp = np.diag(hist)# 标签为正，预测也为正，就是斜对角上的元素
    fp = hist.sum(0) - np.diag(hist)# 负类预测为正类，axis=0每列元素的和。
    fn = hist.sum(1) - np.diag(hist)# 正类预测为负类，axis=1每行元素求和。
    tn = hist.sum(1) + hist.sum(0) - np.diag(hist)# 负类预测为负类，sum()所有元素求和。
    return tp, fp, tn, fn
# hist.sum(0)=按列相加  hist.sum(1)按行相加
def F1ger(hist):
    tp, fp, tn, fn = get_tp_fp_tn_fn(hist)
    Precision = tp / (tp + fp)
    Recall = tp / (tp + fn)
    F1 = (2.0 * Precision * Recall) / (Precision + Recall)
    return F1

# def label_mapping(input, mapping):#主要是因为CityScapes标签里面原类别太多，这样做把其他类别转换成算法需要的类别（共19类）和背景（标注为255）
#    output = np.copy(input)#先复制一下输入图像
#    for ind in range(len(mapping)):
#        output[input == mapping[ind][0]] = mapping[ind][1]#进行类别映射，最终得到的标签里面之后0-18这19个数加255（背景）
#    return np.array(output, dtype=np.int64)#返回映射的标签
'''
  compute_mIoU函数原始以CityScapes图像分割验证集为例来计算mIoU值的（可以根据自己数据集的不同更改类别数num_classes及类别名称name_classes），本函数除了最主要的计算mIoU的代码之外，还完成了一些其他操作，比如进行数据读取，因为原文是做图像分割迁移方面的工作，因此还进行了标签映射的相关工作，在这里笔者都进行注释。大家在使用的时候，可以忽略原作者的数据读取过程，只需要注意计算mIoU的时候每张图片分割结果与标签要配对。主要留意mIoU指标的计算核心代码即可。
'''

def compute_mIoU(gt_dir, pred_dir):  # 计算mIoU的函数
# def compute_mIoU(gt_dir, pred_dir, devkit_dir):  # 计算mIoU的函数
    """
    Compute IoU given the predicted colorized images and
    """
    # with open('/home/ubuntu/DeepLab/datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/info.json', 'r') as fp:
    #     # 读取info.json，里面记录了类别数目，类别名称。（我们数据集是VOC2011，相应地改了josn文件）
    #     info = json.load(fp)
    # num_classes = np.int(info['classes'])  # 读取类别数目，这里是20类
    # print('Num classes', num_classes)  # 打印一下类别数目
    # name_classes = np.array(info['label'], dtype=np.str)  # 读取类别名称
    # # mapping = np.array(info['label2train'], dtype=np.int)#读取标签映射方式，详见博客中附加的info.json文件
    # hist = np.zeros((num_classes, num_classes))  # hist初始化为全零，在这里的hist的形状是[20, 20]
    '''
    原代码是有进行类别映射，所以通过json文件来存放类别数目、类别名称、 标签映射方式。而我们只需要读取类别数目和类别名称即可，可以按下面这段代码将其写死
    '''
    num_classes=6
    print('Num classes', num_classes)
    name_classes = ["ImSurf","Building","LowVeg","Tree","Car","Clutter"]
    # name_classes = ["ImSurf","Building","LowVeg","Tree","Car"]
    hist = np.zeros((num_classes, num_classes))

    # num_classes = 2
    # print('Num classes', num_classes)
    # name_classes = ["background", "fg"]
    # hist = np.zeros((num_classes, num_classes))
    image_data_list = os.listdir(gt_dir)  # 带后缀的所有文件名
    gt_imgs = [it[:-4] for it in image_data_list]  # 不带后缀的所有文件名
    # gt_imgs = open(devkit_dir, 'r').read().splitlines()  # 获得验证集标签名称列表
    gt_imgs = [join(gt_dir, x) for x in gt_imgs]  # 获得验证集标签路径列表，方便直接读取

    # pred_imgs = open(devkit_dir, 'r').read().splitlines()  # 获得验证集图像分割结果名称列表


    pred_data_list = os.listdir(pred_dir)  # 带后缀的所有文件名
    pred_imgs = [it[:-4] for it in pred_data_list]  # 不带后缀的所有文件名
    pred_imgs = [join(pred_dir, x) for x in pred_imgs]
    # pred_imgs = [join(pred_dir, x.split('/')[-1]) for x in pred_imgs]#获得验证集图像分割结果路径列表，方便直接读取

    for ind in range(len(gt_imgs)):  # 读取每一个（图片-标签）对
        pred = np.array(Image.open(pred_imgs[ind]+'.png'))  # 读取一张图像分割结果，转化成numpy数组
        label = np.array(Image.open(gt_imgs[ind]+'.png'))  # 读取一张对应的标签，转化成numpy数组
        # print pred.shape
        # print label.shape
        # label = label_mapping(label, mapping)#进行标签映射（因为没有用到全部类别，因此舍弃某些类别），可忽略
        if len(label.flatten()) != len(pred.flatten()):  # 如果图像分割结果与标签的大小不一样，这张图片就不计算
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()),
                                                                                  len(pred.flatten()), gt_imgs[ind],
                                                                                  pred_imgs[ind]))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)  # 对一张图片计算19×19的hist矩阵，并累加
        if ind > 0 and ind % 10 == 0:  # 每计算10张就输出一下目前已计算的图片中所有类别平均的mIoU值
            print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100 * np.mean(per_class_iu(hist))))
            print(per_class_iu(hist))

    mIoUs = per_class_iu(hist)  # 计算所有验证集图片的逐类别mIoU值
    F1 = F1ger(hist)  # 计算所有验证集图片的逐类别mIoU值
    for ind_class in range(num_classes):  # 逐类别输出一下mIoU值
        print('===>' + name_classes[ind_class] + 'IOU' + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
        print('===>' + name_classes[ind_class] + 'F1' + ':\t' + str(round(F1[ind_class] * 100, 2)))

    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))  # 在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
    print('===> AverageF1: ' + str(round(np.nanmean(F1) * 100, 2)))
    return mIoUs

compute_mIoU('/media/ders/Fjmnew/newdataPuse/test/labels/',
             '/media/ders/Fjmnew/DeeplabNUetCAM/DeeplabNUetCAM/Pexperiments4/pre/model@test@scale=0.5,1.0,1.5,2.0@iteration=0/',
             )  # 执行主函数 三个路径分别为 ‘ground truth’,'自己的实验分割结果'，‘分割图片名称txt文件’