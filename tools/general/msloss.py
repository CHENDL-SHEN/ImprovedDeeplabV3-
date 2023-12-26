import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function, Variable
from itertools import repeat
import numpy as np

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


class MSloss(nn.Module):
    def __init__(self):
        super(MSloss, self).__init__()
        self.beta = 1e-7
        self.lambdaTV = 0.001
        self.penalty = 'l1'

    def levelsetLoss(self, output, target):
        outshape = output.shape
        tarshape = target.shape  # [b, 3, 512 , 512]
        loss = 0.0
        for ich in range(tarshape[1]):  # 每个通道处

            target_ = torch.unsqueeze(target[:, ich], 1)  # [b,h,w] [b,1,h,w]
            target_ = target_.expand(tarshape[0], outshape[1], tarshape[2], tarshape[3])  # # [b,21,h,w]
            with torch.no_grad():
                pcentroid = torch.sum(target_ * output, (2, 3)) / torch.sum(output, (2, 3))
            pcentroid = pcentroid.view(tarshape[0], outshape[1], 1, 1)
            plevel = target_ - pcentroid.expand(tarshape[0], outshape[1], tarshape[2], tarshape[3])
            pLoss = plevel * plevel * output
            loss += torch.sum(pLoss) / 4

        return loss

    def gradientLoss2d(self, output):
        dH = torch.abs(output[:, :, 1:, :] - output[:, :, :-1, :])
        dW = torch.abs(output[:, :, :, 1:] - output[:, :, :, :-1])
        if (self.penalty == "l2"):
            dH = dH * dH
            dW = dW * dW

        loss = (torch.sum(dH) + torch.sum(dW)) / 4
        return loss

    def forward(self, output, target):
        loss_L = self.levelsetLoss(output, target)  # 水平集LS（levelset）；和原图中的颜色有关
        loss_A = self.gradientLoss2d(output) * self.lambdaTV  # 全变分项TV；梯度
        loss_LS = (loss_L + loss_A) * self.beta

        # print("loss_L={},loss_A={},loss_LS={}".format(loss_L * self.beta,loss_A * self.beta,loss_LS))

        return loss_LS
# class levelsetLoss(nn.Module):
#     def __init__(self):
#         super(levelsetLoss, self).__init__()
#
#     def forward(self, output, target):
#         # input size = batch x 1 (channel) x height x width
#         outshape = output.shape
#         tarshape = target.shape
#         loss = 0.0
#         for ich in range(tarshape[1]):
#             target_ = torch.unsqueeze(target[:, ich], 1)
#             target_ = target_.expand(tarshape[0], outshape[1], tarshape[2], tarshape[3])
#             pcentroid = torch.sum(target_ * output, (2, 3)) / torch.sum(output, (2, 3))
#             pcentroid = pcentroid.view(tarshape[0], outshape[1], 1, 1)
#             plevel = target_ - pcentroid.expand(tarshape[0], outshape[1], tarshape[2], tarshape[3])
#             pLoss = plevel * plevel * output
#             loss += torch.sum(pLoss)
#         return loss
#
#
# class gradientLoss2d(nn.Module):
#     def __init__(self, penalty='l1'):
#         super(gradientLoss2d, self).__init__()
#         self.penalty = penalty
#
#     def forward(self, input):
#         dH = torch.abs(input[:, :, 1:, :] - input[:, :, :-1, :])
#         dW = torch.abs(input[:, :, :, 1:] - input[:, :, :, :-1])
#         if (self.penalty == "l2"):
#             dH = dH * dH
#             dW = dW * dW
#
#         loss = torch.sum(dH) + torch.sum(dW)
#         return loss

