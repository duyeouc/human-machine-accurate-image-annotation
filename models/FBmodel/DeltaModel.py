# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO： 欠拟合，效果不好，再改进一下
class DeltaModel(nn.Module):
    """双重注意力机制, attn+self_attention"""
    def __init__(self, logits_dim=[28, 28],
                 rnn_state_dim=[16+64, 28, 28],
                 feats_dim=[128, 14, 14],
                 delta_grid_size=15):
        super(DeltaModel, self).__init__()
        self.use_delta_attn = True
        self.logits_dim = logits_dim
        self.rnn_state_dim = rnn_state_dim
        self.feats_dim = feats_dim
        self.delta_grid_size = delta_grid_size
        self.feat_channels = feats_dim[0]
        self.delta_downsample_conv = nn.Conv2d(in_channels=17+64,
                                               out_channels=17+64,
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               bias=True)
        # 17=16+1
        self.delta_conv1 = nn.Conv2d(in_channels=128 + 17 + 64,
                                     out_channels=64,
                                     kernel_size=3,
                                     padding=1,
                                     stride=1,
                                     bias=True)
        self.delta_bn1 = nn.BatchNorm2d(64)
        self.delta_res1_conv = nn.Conv2d(in_channels=128 + 17 + 64,
                                         out_channels=64,
                                         kernel_size=1,
                                         stride=1,
                                         bias=True)
        self.delta_res1_bn = nn.BatchNorm2d(64)
        self.delta_relu1 = nn.ReLU(inplace=True)  # out: (N, 64, 14, 14)

        self.delta_conv2 = nn.Conv2d(in_channels=64,
                                     out_channels=16,
                                     kernel_size=3,
                                     padding=1,
                                     stride=1,
                                     bias=True)

        self.delta_bn2 = nn.BatchNorm2d(16)
        self.delta_res2_conv = nn.Conv2d(in_channels=64,
                                         out_channels=16,
                                         kernel_size=1,
                                         stride=1,
                                         bias=True)
        self.delta_res2_bn = nn.BatchNorm2d(16)
        self.delta_relu2 = nn.ReLU(inplace=True)

        self.delta_final = nn.Linear(16 * 14 * 14, delta_grid_size ** 2)  # 输出一个15*15的grid

        self.use_delta_attn = True
        # delta attn
        self.delta_fc_att = nn.Linear(in_features=self.feat_channels,
                                      out_features=1)
        self.delta_conv_att = nn.Conv2d(in_channels=17+64,
                                        out_channels=self.feat_channels,
                                        kernel_size=1,
                                        padding=0,
                                        bias=True)

    def self_attn(self, mask):
        pass


    def forward(self, last_rnn_h, rnn_logits, encoder_features, mode='train_ce', temperature=0.0):
        """
        last_rnn_h 接收convLSTM传过来的LSTM隐藏层向量，传过来的已经做了拼接，拼接了两层的h向量
        rnn_logits：最后得到的逻辑值
        encoder_feats：是传过来的FPN得到的特征map, [bs, 128, 14, 14]
        :param last_rnn_h: [bs, 71, 64+16, 28, 28]=[bs, 71, 16, 28, 28] + [bs, 71, 64, 28, 28] => 16+64 channel
        :param rnn_logits: [bs, 71, 28, 28]
        :param encoder_features: # [bs, 128, 14, 14]
        :param temperature:
        :return:
        """
        time_steps = last_rnn_h.shape[1]
        rnn_h = last_rnn_h.contiguous().view(-1, 16+64, 28, 28)
        rnn_logits = rnn_logits.contiguous().view(-1, 28, 28).unsqueeze(1)
        pre_feats = torch.cat([rnn_h, rnn_logits], dim=1)  # [N*71, 16+64+1, 28, 28]

        pre_feats = self.delta_downsample_conv(pre_feats)  # [N*71, 16+64+1, 14, 14]

        # Extend encoder feats
        encoder_features = encoder_features.unsqueeze(1).repeat(1, time_steps, 1, 1, 1)  # [bs, 128, 14, 14]
        encoder_features = encoder_features.contiguous().view(-1, 128, 14, 14)  # [N*71, 128, 14, 14]

        if self.use_delta_attn:
            encoder_features = self.delta_attn(pre_feats, encoder_features)
        out_dict = self.delta_step(pre_feats, encoder_features, temperature=temperature)

        delta_logits = out_dict['delta_logits']  # [N*71, 15*15]
        delta_logprob = out_dict['delta_logprob']  # [N, 71]
        delta_pred = out_dict['delta_pred']  # [N*71]

        delta_logits = delta_logits.view(-1, time_steps, 15*15)
        delta_logprob = delta_logprob.view(-1, time_steps)
        delta_pred = delta_pred.view(-1, time_steps)

        del(out_dict)

        out_dict = defaultdict()
        out_dict['delta_logits'] = delta_logits
        out_dict['delta_logprob'] = delta_logprob
        out_dict['delta_pred'] = delta_pred

        return out_dict


    def delta_step(self, pre_features, encoder_features, mode='train_ce', temperature=0.0):
        """
        prefeats: [N*71, 64+16+1, 14, 14]
        encoder_feats: 经过attn处理过的 [N*71, 128, 14, 14]


        :param last_rnn_h:
        :param rnn_logits:
        :param delta_features:
        :return:
                    final_h: (N, 16, 28, 28)
            final_logits: (N, 28, 28)
            cat: [N, 17, 28, 28]
            -> downsampling: [N, 17, 14, 14]
            encoder_features: [N, 128, 14, 14]
            + attn: [N, 128, 14, 14]
            -> cat: [N, 128+17, 14, 14]   =>输入进去即可
            采取一种resnet的方式

            train_ce 用logits
            rl 用logprobs
            test 用pred-delta
        """
        # pre_features = torch.cat([last_rnn_h, rnn_logits.unsqueeze(1)], dim=1)
        # pre_features = self.delta_downsample_conv(pre_features)  # [N, 17, 14, 14]

        # Use attn, same with rnn attention mechanism
        # if self.use_delta_attn:
        #    encoder_features = self.delta_attn(pre_features, encoder_features)

        features = torch.cat([pre_features, encoder_features], dim=1)  # [N, 128+16+64+1, 14, 14]

        res1 = self.delta_res1_conv(features)
        res1 = self.delta_res1_bn(res1)

        out1 = self.delta_conv1(features)
        out1 = self.delta_bn1(out1)

        out1 = self.delta_relu1(out1 + res1)

        res2 = self.delta_res2_conv(out1)
        res2 = self.delta_res2_bn(res2)
        out2 = self.delta_conv2(out1)
        out2 = self.delta_bn2(out2)

        out = self.delta_relu2(res2 + out2)  # (N, 16, 14, 14)

        bs = out.shape[0]
        out = out.view(bs, -1)
        delta_logits = self.delta_final(out)  # (N, 14*14)

        out_delta = {}

        out_delta['delta_logits'] = delta_logits

        delta_logprobs = F.log_softmax(delta_logits, dim=-1)
        if temperature < 0.01:
            delta_logprob, delta_pred = torch.max(delta_logprobs, dim=-1)  # greedy
            out_delta['delta_logprob'] = delta_logprob
            out_delta['delta_pred'] = delta_pred

        else:
            probs = torch.exp(delta_logprobs / temperature)
            cur_pred = torch.multinomial(probs, 1)
            cur_logprob = delta_logprobs.gather(1, cur_pred)
            cur_pred = torch.squeeze(cur_pred, dim=-1)  # (bs, )
            cur_logprob = torch.squeeze(cur_logprob, dim=-1)  # prob (bs,)
            out_delta['delta_logprob'] = cur_logprob
            out_delta['delta_pred'] = cur_pred

        return out_delta


    # delta-attention mechanism
    def delta_attn(self, input, feats):
        """
        :param input: pre_features, 即last_rnn_h + logits, 17, [N, 17, 14, 14]
        :param feats: encoder_feats  [N, 128, 14, 14]
        :return:
        """
        f12 = self.delta_conv_att(input)  # [N, 128, 14, 14]
        f12 = F.relu(f12)  # [N, 128, 14, 14]
        f12 = f12.permute(0, 2, 3, 1).contiguous().view(-1, self.feat_channels)  # [N*14*14, 128]
        f_att = self.delta_fc_att(f12)  # [N*14*14, 1]
        f_att = f_att.contiguous().view(-1, (self.delta_grid_size-1) ** 2)  # (N, 14*14)
        a_t = F.log_softmax(f_att, dim=-1)
        a_t = a_t.view(-1, 1, self.delta_grid_size-1, self.delta_grid_size-1)  # (N, 1, 14, 14)
        feats = feats * a_t
        return feats
