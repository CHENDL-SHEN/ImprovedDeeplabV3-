# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import sys
import copy
import shutil
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

from core.networks import *
from core.datasets import *

from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *
from tools.general.msloss import *

from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *

from tools.ai.augment_utils import *
from tools.ai.randaugment import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

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
# Hyperparameter
###############################################################################
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--max_epoch', default=50, type=int)

parser.add_argument('--lr', default=0.007, type=float)
parser.add_argument('--wd', default=4e-5, type=float)
parser.add_argument('--nesterov', default=True, type=str2bool)

parser.add_argument('--image_size', default=512, type=int)
parser.add_argument('--min_image_size', default=256, type=int)
parser.add_argument('--max_image_size', default=1024, type=int)

parser.add_argument('--print_ratio', default=0.1, type=float)

parser.add_argument('--tag', default='model', type=str)# 用这个命名存储的文件名

parser.add_argument('--label_name', default='resnet50@seed=0@aug=Affinity_ResNet50@ep=3@nesterov@train_aug@beta=10@exp_times=8@rw', type=str)

if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()
    
    log_dir = create_directory(f'./Pexperiments4/logs/')# 采用io_util文件创建文件夹
    data_dir = create_directory(f'./Pexperiments4/data/')
    model_dir = create_directory('./Pexperiments4/models/')
    tensorboard_dir = create_directory(f'./Pexperiments4/tensorboards/{args.tag}/')
    pred_dir = '/media/ders/dataset/VOC2012/SegmentationClassAug/'
    # pred_dir = './experiments/predictions/{}/'.format(args.label_name)

    log_path = log_dir + f'{args.tag}.txt'
    data_path = data_dir + f'{args.tag}.json'
    model_path = model_dir + f'{args.tag}.pth'
    
    set_seed(args.seed)
    log_func = lambda string='': log_print(string, log_path)# 这个lambda函数实际上是一个简单的封装，用于将log_print函数与特定的log_path绑定在一起，以便稍后调用。
    
    log_func('[i] {}'.format(args.tag))# 供打印用
    log_func()

    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    normalize_fn = Normalize(imagenet_mean, imagenet_std)
    
    train_transforms = [
        RandomResize_For_Segmentation(args.min_image_size, args.max_image_size),
        RandomHorizontalFlip_For_Segmentation(),
        
        Normalize_For_Segmentation(imagenet_mean, imagenet_std),
        RandomCrop_For_Segmentation(args.image_size),
    ]
    
    # if 'Seg' in args.architecture:
    #     if 'C' in args.architecture:
    #         train_transforms.append(Resize_For_Mask(args.image_size // 4))
    #     else:
    #         train_transforms.append(Resize_For_Mask(args.image_size // 8))

    train_transform = transforms.Compose(train_transforms + [Transpose_For_Segmentation()])
    
    test_transform = transforms.Compose([
        Normalize_For_Segmentation(imagenet_mean, imagenet_std),
        Top_Left_Crop_For_Segmentation(args.image_size),
        Transpose_For_Segmentation()# 该数据增强将 H W C->C H W
    ])
    
    meta_dic = read_json('./data/VOC_2012.json')
    class_names = np.asarray(meta_dic['class_names'])
    
    train_dataset = VOC_Dataset_For_WSSS(args.data_dir, 'train', pred_dir, train_transform)# 这里存放的训练的图片名
    valid_dataset = VOC_Dataset_For_Segmentation(args.data_dir, 'val', test_transform)# 存放的是验证集的图片名称

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=1, shuffle=False, drop_last=True)
    
    log_func('[i] mean values is {}'.format(imagenet_mean))
    log_func('[i] std values is {}'.format(imagenet_std))
    log_func('[i] The number of class is {}'.format(meta_dic['classes']))
    log_func('[i] train_transform is {}'.format(train_transform))
    log_func()

    val_iteration = len(train_loader)# train_loader总共有1320个batch
    log_iteration = int(val_iteration * args.print_ratio)# 1320*0.1就为132个
    max_iteration = args.max_epoch * val_iteration# 最大迭代次数：最大epoch*train_loader的数
    
    log_func('[i] log_iteration : {:,}'.format(log_iteration))
    log_func('[i] val_iteration : {:,}'.format(val_iteration))
    log_func('[i] max_iteration : {:,}'.format(max_iteration))
    
    ###################################################################################
    # Network
    ###################################################################################
    if args.architecture == 'DeepLabv3+':
        model = DeepLabv3_Plus(args.backbone, num_classes=meta_dic['classes'], mode=args.mode, use_group_norm=args.use_gn)
    elif args.architecture == 'Seg_Model':
        model = Seg_Model(args.backbone, num_classes=meta_dic['classes'] + 1)
    elif args.architecture == 'CSeg_Model':
        model = CSeg_Model(args.backbone, num_classes=meta_dic['classes'] + 1)

    param_groups = model.get_parameter_groups(None)
    params = [
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wd},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wd},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0},
    ]
    
    model = model.cuda()
    model.train()

    log_func('[i] Architecture is {}'.format(args.architecture))
    log_func('[i] Total Params: %.2fM'%(calculate_parameters(model)))
    log_func()

    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        use_gpu = '0'

    the_number_of_gpu = len(use_gpu.split(','))
    if the_number_of_gpu > 1:
        log_func('[i] the number of gpu : {}'.format(the_number_of_gpu))
        model = nn.DataParallel(model)# 多个GPU来跑

        # for sync bn
        # patch_replication_callback(model)

    load_model_fn = lambda: load_model(model, model_path, parallel=the_number_of_gpu > 1)
    save_model_fn = lambda: save_model(model, model_path, parallel=the_number_of_gpu > 1)
    save_model_fn_for_backup = lambda: save_model(model, model_path.replace('.pth', f'_backup.pth'), parallel=the_number_of_gpu > 1)
    
    ###################################################################################
    # Loss, Optimizer
    ###################################################################################
    class_loss_fn = nn.CrossEntropyLoss(ignore_index=255).cuda()

    # log_func('[i] The number of pretrained weights : {}'.format(len(param_groups[0])))
    # log_func('[i] The number of pretrained bias : {}'.format(len(param_groups[1])))
    # log_func('[i] The number of scratched weights : {}'.format(len(param_groups[2])))
    # log_func('[i] The number of scratched bias : {}'.format(len(param_groups[3])))
    
    optimizer = PolyOptimizer(params, lr=args.lr, momentum=0.9, weight_decay=args.wd, max_step=max_iteration, nesterov=args.nesterov)
    
    #################################################################################################
    # Train
    #################################################################################################
    data_dic = {
        'train' : [],
        'validation' : [],
    }# data_dic是一个字典，train和validation是键，用来存储训练和验证的数据；列表中的数据可能会随时间而变化，因此可以使用Average_Meter类来计算它们的平均值

    train_timer = Timer()
    eval_timer = Timer()

    train_meter = Average_Meter(['loss'])

    best_valid_mIoU = -1

    def evaluate(loader):
        model.eval()
        eval_timer.tik()

        meter = Calculator_For_mIoU('./data/VOC_2012.json') 

        with torch.no_grad():
            length = len(loader)
            for step, (images, labels) in enumerate(loader):
                images = images.cuda()
                labels = labels.cuda()

                logits = model(images)
                print(logits.shape)
                predictions = torch.argmax(logits, dim=1)
                
                # for visualization
                if step == 0:
                    for b in range(4):
                        image = get_numpy_from_tensor(images[b])
                        pred_mask = get_numpy_from_tensor(predictions[b])

                        image = denormalize(image, imagenet_mean, imagenet_std)[..., ::-1]# [..., ::-1]将BGR转化为RGB
                        h, w, c = image.shape

                        pred_mask = decode_from_colormap(pred_mask, train_dataset.colors)# 根据预测的掩码图像和颜色映射表，将预测的掩码转换为可视化的图像。
                        pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)# 将预测的掩码图像调整为与原始图像相同的尺寸。使用cv2.resize()函数进行调整，采用最近邻插值方法。
                        image = cv2.convertScaleAbs(image)
                        pred_mask = cv2.convertScaleAbs(pred_mask)
                        image = cv2.addWeighted(image, 0.5, pred_mask, 0.5, 0)[..., ::-1]# 将原始图像和预测的掩码图像按照一定的权重进行混合。这里使用cv2.addWeighted()函数，权重分别为0.5，表示两个图像的混合比例为1:1。最后，再将通道顺序从BGR转换为RGB。
                        image = image.astype(np.float32) / 255.

                        writer.add_image('Mask/{}'.format(b + 1), image, iteration, dataformats='HWC')
                
                for batch_index in range(images.size()[0]):
                    pred_mask = get_numpy_from_tensor(predictions[batch_index])
                    gt_mask = get_numpy_from_tensor(labels[batch_index])

                    h, w = pred_mask.shape
                    gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    meter.add(pred_mask, gt_mask)

                sys.stdout.write('\r# Evaluation [{}/{}] = {:.2f}%'.format(step + 1, length, (step + 1) / length * 100))
                sys.stdout.flush()
        
        print(' ')
        model.train()
        
        return meter.get(clear=True)
    
    writer = SummaryWriter(tensorboard_dir)# 向tensorboard_dir这个文件目录中写入信息，即存信息
    train_iterator = Iterator(train_loader)# 迭代器，遍布整个train_loader
    # torch.autograd.set_detect_anomaly(True)
    ##############one-hot独热编码#######
    # def mask2one_hot(label, out):
    #     """
    #     label: 标签图像 # （batch_size, 1, h, w)
    #     out: 网络的输出
    #     """
    #     num_classes = out  # 分类类别数
    #
    #     # current_label = label.squeeze(1)  # （batch_size, 1, h, w) ---> （batch_size, h, w)
    #
    #     batch_size, h, w = label.shape[0], label.shape[1], label.shape[2]
    #
    #     print(h, w, batch_size)
    #
    #     one_hots = []
    #     for i in range(num_classes):
    #         tmplate = torch.ones(batch_size, h, w)  # （batch_size, h, w)
    #         tmplate[label != i] = 0
    #         tmplate = tmplate.view(batch_size, 1, h, w)  # （batch_size, h, w) --> （batch_size, 1, h, w)
    #
    #         one_hots.append(tmplate)
    #
    #     onehot = torch.cat(one_hots, dim=1)
    #
    #     return onehot

    for iteration in range(max_iteration):# 总的batch 一个epoch有多少个batch*总的epoch
        images, labels = train_iterator.get()# 使trainloader可以迭代了
        images, labels = images.cuda(), labels.cuda()# image(8,3,512,512);labels(8,512,512)
        #################################################################################################
        # Inference
        #################################################################################################
        logits = model(images)# 4 6 512 512；若想改进则需要将logits进行softmax，然后再将标签 进行one-hot独热编码，即把标签按照通道数变为0-1模式；使得预测图和标签变为相同的维度

        # out = meta_dic['classes']
        # labels = mask2one_hot(labels, out)

        ###############################################################################
        # The part is to calculate losses.
        ###############################################################################
        if 'Seg' in args.architecture:
            labels = resize_for_tensors(labels.type(torch.FloatTensor).unsqueeze(1), logits.size()[2:], 'nearest', None)[:, 0, :, :]
            labels = labels.type(torch.LongTensor).cuda()

            # print(labels.size(), labels.min(), labels.max())

        alpha_CE = 1
        alpha_MS = 1
        loss_C = class_loss_fn(logits, labels)
        probs = F.softmax(logits, dim=1)
        loss_LS = MSloss()(probs, images)
        loss = alpha_CE * loss_C + alpha_MS * loss_LS
        # loss = class_loss_fn(logits, labels)# torch.nn.CrossEntropyLoss()在语义分割中计算损失时会对每个像素的loss求平均
        #################################################################################################
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_meter.add({
            'loss' : loss.item(), 
        })
        
        #################################################################################################
        # For Log
        #################################################################################################
        if (iteration + 1) % log_iteration == 0:# 一个epoch总数据有N个batch，每隔0.1*N个batch打印一次loss等等数据
            loss = train_meter.get(clear=True)
            learning_rate = float(get_learning_rate_from_optimizer(optimizer))
            
            data = {
                'iteration' : iteration + 1,
                'learning_rate' : learning_rate,
                'loss' : loss,
                'time' : train_timer.tok(clear=True),
            }
            data_dic['train'].append(data)# 字典里边的键存了一个data字典。这段代码将一个包含训练数据的字典data添加到data_dic字典中的'train'键对应的列表中
            write_json(data_path, data_dic)# 训练结果数据成json文件
            ## 每一个epoch后将下边这些数据存入到txt文件中
            log_func('[i] \
                iteration={iteration:,}, \
                learning_rate={learning_rate:.4f}, \
                loss={loss:.4f}, \
                time={time:.0f}sec'.format(**data)
            )

            writer.add_scalar('Train/loss', loss, iteration)# 这行代码的含义是将当前训练迭代的损失值记录到日志中，用于在训练过程中跟踪损失值的变化。其中，'Train/loss'是记录的标签名称，loss是当前迭代的损失值，iteration是当前的迭代次数。这段代码通常用于在训练过程中可视化和记录损失值的变化。
            writer.add_scalar('Train/learning_rate', learning_rate, iteration)
        
        #################################################################################################
        # Evaluation
        #################################################################################################
        if (iteration + 1) % val_iteration == 0:
            mIoU, _ = evaluate(valid_loader)

            if best_valid_mIoU == -1 or best_valid_mIoU < mIoU:
                best_valid_mIoU = mIoU

                save_model_fn()
                log_func('[i] save model')

            data = {
                'iteration' : iteration + 1,
                'mIoU' : mIoU,
                'best_valid_mIoU' : best_valid_mIoU,
                'time' : eval_timer.tok(clear=True),
            }
            data_dic['validation'].append(data)
            write_json(data_path, data_dic)
            
            log_func('[i] \
                iteration={iteration:,}, \
                mIoU={mIoU:.2f}%, \
                best_valid_mIoU={best_valid_mIoU:.2f}%, \
                time={time:.0f}sec'.format(**data)
            )
            
            writer.add_scalar('Evaluation/mIoU', mIoU, iteration)
            writer.add_scalar('Evaluation/best_valid_mIoU', best_valid_mIoU, iteration)
    
    write_json(data_path, data_dic)
    writer.close()

    print(args.tag)