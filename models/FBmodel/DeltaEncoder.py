# -*- coding: utf-8 -*-
from __future__ import print_function
import torch.nn as nn

# 分辨率增强网络，用的这个
class DeltaEncoder(nn.Module):
    def __init__(self, input_channels=256, feats_channels=128):
        super(DeltaEncoder, self).__init__()
        # self.conv_out1 = nn.Sequential(
        #     nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=True),
        #     nn.BatchNorm2d(128),
        #     # 更改一下, 简化一下
        #     nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(128),
        # )  # out: [128, 14, 14]


        # +的4层额外的卷积操作
        basic1 = self._make_basic(256, 256, 3, 1, 1)
        basic2 = self._make_basic(256, 128, 3, 1, 1)
        basic3 = self._make_basic(128, 128, 3, 2, 1)
        basic4 = self._make_basic(128, 128, 3, 1, 1)
        self.conv_out1 = nn.Sequential(basic1, basic2, basic3, basic4)
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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


    def forward(self, encoder_feats):
        return self.conv_out1(encoder_feats)

if __name__ == '__main__':
    from models.FPN2 import FPN
    import torch
    m = FPN()
    # print(m)
    img = torch.rand((1, 3, 224, 224))
    x1, x2 = m(img)
    print(x1.shape)
    print(x2.shape)
    m2 = DeltaEncoder()
    out = m2(x2)
    print(out.shape)