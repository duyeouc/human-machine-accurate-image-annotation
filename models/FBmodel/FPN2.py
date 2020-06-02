# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from .resnet import resnext50_32x4d

# 用的是这个FPN
class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        self.resnext = resnext50_32x4d()

        self.latconv1 = nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0)
        self.latconv2 = nn.Conv2d(1024, 128, kernel_size=1, stride=1, padding=0)
        self.latconv3 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.latconv4 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.latconv5 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)

        self.channel_down1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.channel_down2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.channel_down3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)

        conv1 = self._make_basic(256, 256, 3, 1, 1)
        conv2 = self._make_basic(256, 128, 3, 1, 1)
        conv3 = self._make_basic(128, 128, 3, 2, 1)  # dowsample
        conv4 = self._make_basic(128, 128, 3, 1, 1)

        self.conv_final = nn.Sequential(conv1, conv2, conv3, conv4)



    def _upsample_concat(self, x, y):
        _, _, H, W = y.size()
        return torch.cat([F.upsample(x, size=(H, W), mode='bilinear'), y], dim=1)


    def _make_basic(self,
                    inchannel,
                    outchannel,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    use_bn=True,
                    activation=True):
        conv = nn.Conv2d(in_channels=inchannel,
                         out_channels=outchannel,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding)

        bn = nn.BatchNorm2d(outchannel)
        relu = nn.ReLU(inplace=True)

        return nn.Sequential(conv, bn, relu)


    def forward(self, x):
        out5, out4, out3, out2, out1 = self.resnext(x)
        #print(out5.shape)
        out1 = self.latconv1(out1)  # 最上的  128,7,7
        out2 = self.latconv2(out2)  # 128,14,14
        out3 = self.latconv3(out3)  # 128,28,28
        out4 = self.latconv4(out4)  # 128,56,56
        out5 = self.latconv5(out5)  # 128,56,56

        tmp1 = self._upsample_concat(out1, out2)  # (256, 14, 14)
        tmp1 = self.channel_down1(tmp1)

        tmp2 = self._upsample_concat(tmp1, out3)  # (256,28,28)
        delta_feature = tmp2  # (256, 28, 28)
        tmp2 = self.channel_down2(tmp2)
        tmp3 = self._upsample_concat(tmp2, out4)  # (256, 56, 56)
        tmp3 = self.channel_down3(tmp3)
        tmp4 = self._upsample_concat(tmp3, out5)  # (256, 56, 56)  # 这个相当于没有做Upsample
        features = self.conv_final(tmp4)  # (128, 28, 28)

        return features, delta_feature


if __name__ == '__main__':
    x = FPN()
    input = torch.rand(1, 3, 224, 224)
    x(input)


