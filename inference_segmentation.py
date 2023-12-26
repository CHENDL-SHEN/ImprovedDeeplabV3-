# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import sys
import copy
import shutil
import random
import argparse
import numpy as np

import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

from core.puzzle_utils import *
from core.networks import *
from core.datasets import *

from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *

from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *

from tools.ai.augment_utils import *
from tools.ai.randaugment import *

parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--data_dir', default='/media/ders/Fjmnew/newdataPuse/', type=str)

###############################################################################
# Network
###############################################################################
parser.add_argument('--architecture', default='DeepLabv3+', type=str)
parser.add_argument('--backbone', default='resnet50', type=str)
parser.add_argument('--mode', default='fix', type=str)
parser.add_argument('--use_gn', default=True, type=str2bool)

###############################################################################
# Inference parameters
###############################################################################
parser.add_argument('--tag', default='model', type=str)

parser.add_argument('--domain', default='test', type=str)

parser.add_argument('--scales', default='0.5,1.0,1.5,2.0', type=str)
parser.add_argument('--iteration', default=0, type=int)

if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()

    model_dir = create_directory('./Pexperiments4/models/')
    model_path = model_dir + f'{args.tag}.pth'

    if 'train' in args.domain:
        args.tag += '@train'
    else:
        args.tag += '@' + args.domain
    
    args.tag += '@scale=%s'%args.scales
    args.tag += '@iteration=%d'%args.iteration

    pred_dir = create_directory('./Pexperiments4/vis/{}/'.format(args.tag))
    
    set_seed(args.seed)
    log_func = lambda string='': print(string)
    
    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    normalize_fn = Normalize(imagenet_mean, imagenet_std)
    
    # for mIoU
    meta_dic = read_json('./data/VOC_2012.json')
    dataset = VOC_Dataset_For_Evaluation(args.data_dir, args.domain)
    
    ###################################################################################
    # Network
    ###################################################################################
    if args.architecture == 'DeepLabv3+':
        model = DeepLabv3_Plus(args.backbone, num_classes=meta_dic['classes'], mode=args.mode, use_group_norm=args.use_gn)
    elif args.architecture == 'Seg_Model':
        model = Seg_Model(args.backbone, num_classes=meta_dic['classes'] + 1)
    elif args.architecture == 'CSeg_Model':
        model = CSeg_Model(args.backbone, num_classes=meta_dic['classes'] + 1)
    
    model = model.cuda()
    model.eval()

    log_func('[i] Architecture is {}'.format(args.architecture))
    log_func('[i] Total Params: %.2fM'%(calculate_parameters(model)))
    log_func()

    load_model(model, model_path, parallel=False)
    
    #################################################################################################
    # Evaluation
    #################################################################################################
    eval_timer = Timer()
    scales = [float(scale) for scale in args.scales.split(',')]
    
    model.eval()
    eval_timer.tik()

    def inference(images, image_size):
        images = images.cuda()
        
        logits = model(images)
        logits = resize_for_tensors(logits, image_size)#上采样
        
        logits = logits[0] + logits[1].flip(-1)# logits是一个二维张量，那么logits[0]表示取第一个元素，logits[1]表示取第二个元素，然后对第二个元素进行水平翻转操作(.flip(-1))，最后将两个结果进行逐元素相加。
        logits = get_numpy_from_tensor(logits).transpose((1, 2, 0))# 将张量转换为数组并且改变通道次序为 H W C
        return logits

    with torch.no_grad():
        length = len(dataset)
        for step, (ori_image, image_id, gt_mask) in enumerate(dataset):
            ori_w, ori_h = ori_image.size

            cams_list = []

            for scale in scales:
                image = copy.deepcopy(ori_image)
                image = image.resize((round(ori_w*scale), round(ori_h*scale)), resample=PIL.Image.CUBIC)# 按照规模变换尺寸
                
                image = normalize_fn(image)# 正则化H  W  C
                image = image.transpose((2, 0, 1))# 变为C H W  一个通道一显示

                image = torch.from_numpy(image)# 变为tensor
                flipped_image = image.flip(-1)# 这行代码的含义是将变量image进行水平翻转操作，并将结果存储在flipped_image中。参数-1表示在水平方向上进行翻转操作
                
                images = torch.stack([image, flipped_image])# 2 3 256 256这行代码的含义是将image和flipped_image两个张量沿着新的维度进行堆叠，并将结果存储在images中

                cams = inference(images, (ori_h, ori_w))
                cams_list.append(cams)
            
            preds = np.sum(cams_list, axis=0)# cams_list中的所有元素进行按列求和操作，并将结果存储在preds中
            preds = F.softmax(torch.from_numpy(preds), dim=-1).numpy()
            
            if args.iteration > 0:
                # h, w, c -> c, h, w
                preds = crf_inference(np.asarray(ori_image), preds.transpose((2, 0, 1)), t=args.iteration)
                pred_mask = np.argmax(preds, axis=0)
            else:
                pred_mask = np.argmax(preds, axis=-1)

            ###############################################################################
            # cv2.imwrite('./demo.jpg', np.concatenate([np.asarray(ori_image)[..., ::-1], decode_from_colormap(pred_mask, dataset.colors)], axis=1))
            # input('write')
            #
            # cv2.imshow('Image', np.asarray(ori_image)[..., ::-1])
            # cv2.imshow('Prediction', decode_from_colormap(pred_mask, dataset.colors))
            # cv2.imshow('GT', decode_from_colormap(gt_mask, dataset.colors))
            # cv2.waitKey(0)
            ###############################################################################

            # if args.domain == 'test':
            #     # pred_mask = decode_from_colormap(pred_mask, dataset.colors)[..., ::-1]# 后边中括号是将BGR->RGB，遥感图像三通道 IR R G
                pred_mask = decode_from_colormap(pred_mask, dataset.colors)
                # # 定义标签颜色映射
                # label_colors = {
                #     0: (255, 255, 255),  # 背景
                #     1: (0, 0, 255),  # 类别1
                #     2: (0, 255, 255),  # 类别2
                #     3: (0, 255, 0),  # 类别3
                #     4: (255, 204, 0),  # 类别3
                #     5: (255, 0, 0)  # 类别3
                # }
                # # 创建一个彩色图像，与标签掩码图像大小相同
                # colored_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
                #
                # # 遍历标签掩码图像的每个像素，并根据标签颜色映射进行上色
                # for label in label_colors:
                #     colored_mask[pred_mask == label] = label_colors[label]
                #
                # # 显示彩色标签掩码图像
                # # cv2.imshow('Colored Mask', colored_mask)
                # # cv2.waitKey(0)
                # # cv2.destroyAllWindows()
            # imageio.imwrite(pred_dir + image_id + '.png', colored_mask.astype(np.uint8))
            imageio.imwrite(pred_dir + image_id + '.png', pred_mask.astype(np.uint8))

            sys.stdout.write('\r# Make CAM [{}/{}] = {:.2f}%'.format(step + 1, length, (step + 1) / length * 100))
            sys.stdout.flush()
        print()
    
    if args.domain == 'val':
        print("python3 evaluate.py --experiment_name {} --domain {} --mode png".format(args.tag, args.domain))