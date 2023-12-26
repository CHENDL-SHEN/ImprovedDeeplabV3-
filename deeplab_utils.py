# Copyright (C) 2021 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):  # 通道注意力机制
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):# 空间注意力机制
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, norm_fn=None):
        super().__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = norm_fn(planes)
        self.relu = nn.ReLU(inplace=True)

        self.initialize([self.atrous_conv, self.bn])

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)

    def initialize(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, output_stride, norm_fn):
        super().__init__()

        inplanes = 2048

        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        
        self.aspp1 = ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], norm_fn=norm_fn)
        self.aspp2 = ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], norm_fn=norm_fn)
        self.aspp3 = ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], norm_fn=norm_fn)
        self.aspp4 = ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], norm_fn=norm_fn)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
            norm_fn(256),
            nn.ReLU(inplace=True),
        )

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = norm_fn(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        
        self.initialize([self.conv1, self.bn1] + list(self.global_avg_pool.modules()))
    
    def forward(self, x):
        x1 = self.aspp1(x)#(8,256,32,32)
        x2 = self.aspp2(x)#(8,256,32,32)
        x3 = self.aspp3(x)#(8,256,32,32)
        x4 = self.aspp4(x)#(8,256,32,32)

        x5 = self.global_avg_pool(x)#(8,256,1,1)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)# 拼接5个分支产生的特征图(8,1280,32,32)

        x = self.conv1(x)# 经过1*1卷积(8,256,32,32)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x

    def initialize(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Decoder(nn.Module):
    def __init__(self, num_classes, low_level_inplanes, norm_fn):
        super().__init__()

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = norm_fn(48)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(48)
        self.sa = SpatialAttention()

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
            norm_fn(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            norm_fn(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        )

        self.initialize([self.conv1, self.bn1] + list(self.classifier.modules()))

    def forward(self, x, x_low_level):
        x_low_level = self.conv1(x_low_level)# (4,48,128,128)实验来看48要比64好
        x_low_level = self.bn1(x_low_level)# (4 48 128 128)
        x_low_level = self.relu(x_low_level)# 低阶特征经过1*1卷积(4 48 128 128)
        x_low_level = self.ca(x_low_level) * x_low_level
        sa = self.sa(x_low_level)
        print(sa.shape)
        x_low_level = self.sa(x_low_level) * x_low_level

        x = F.interpolate(x, size=x_low_level.size()[2:], mode='bilinear', align_corners=True)# 将高级特征上采样4倍(8,256,128,128)
        x = torch.cat((x, x_low_level), dim=1)# (8,304,128,128)
        x = self.classifier(x)# (8,21,128,128)

        return x

    def initialize(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()