# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
from models.FBmodel.FPN2 import FPN
from models.FBmodel.DeltaEncoder import DeltaEncoder
from models.convLSTM import AttConvLSTM
from models.FBmodel.DeltaModel import DeltaModel

class PolygonModel(nn.Module):
    def __init__(self,
                 load_predtrained_resnet50=False,
                 predict_delta=False,
                 loop_T=1):
        super(PolygonModel, self).__init__()
        self.load_predtrained_resnet50 = load_predtrained_resnet50
        self.predict_delta = predict_delta
        self.loop_T = loop_T
        self.encoder = FPN()
        self.decoder = AttConvLSTM(predict_delta=False, loop_T=self.loop_T)  # 注意阈值
        # delta model
        if self.predict_delta:
            self.delta_encoder = DeltaEncoder()
            self.delta_model = DeltaModel()

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        res_path = '/data/duye/pretrained_models/resnet/resnext50_32x4d-7cdf4587.pth'

        if load_predtrained_resnet50:
            self.encoder.resnext.load_state_dict(torch.load(res_path))
            print('Load pretrained resnet50 completed!')

    def forward(self, img,
                pre_v2=None,
                pre_v1=None,
                temperature=0.0,
                temperature2=0.0,
                mode='train_ce',
                fp_beam_size=1,
                beam_size=1,
                return_attention=False,
                use_correction=False,
                gt_28=None):

        if beam_size == 1:
            if self.predict_delta:
                if mode == 'train_ce':
                    feats, feats_delta = self.encoder(img)
                    feats_delta = self.delta_encoder(feats_delta)

                    decoder_dict = self.decoder(feats, feats_delta, pre_v2, pre_v1, mode=mode, temperature=temperature)
                    rnn_h = decoder_dict['last_rnn_h']
                    bs = rnn_h.shape[0]
                    time_steps = rnn_h.shape[1]
                    logits = decoder_dict['logits'][:, :, :-1].view(bs, time_steps, 28, 28)
                    out_delta_dict = self.delta_model(rnn_h, logits, feats_delta, mode=mode, temperature=temperature)
                    decoder_dict.update(out_delta_dict)
                    return decoder_dict
                elif mode == 'train_rl':
                    feats, feats_delta = self.encoder(img)
                    # 提取delta, train_delta时这个DeltaEncoder训练
                    feats_delta = self.delta_encoder(feats_delta)

                    # decoder并没有用到feats_delta, 因为他不预测delta
                    decoder_dict = self.decoder(feats, feats_delta, pre_v2, pre_v1, mode=mode, temperature=temperature)
                    rnn_h = decoder_dict['last_rnn_h']
                    bs = rnn_h.shape[0]
                    time_steps = rnn_h.shape[1]
                    logits = decoder_dict['logits'][:, :, :-1].view(bs, time_steps, 28, 28)
                    # 偏置是用这个预测的
                    out_delta_dict = self.delta_model(rnn_h, logits, feats_delta, mode='mode', temperature=temperature2)

                    # get final vertex
                    delta_pred = out_delta_dict['delta_pred']  # (bs, len_s=71, ignore EOS)
                    pred = decoder_dict['pred_polys']  # (bs, len_s=71)

                    dx = delta_pred % 15 - 7  # (bs, len_s)
                    dy = (delta_pred // 15) - 7

                    # To(0, 224)
                    pred_x = (pred % 28) * 8.0 + 4  # (bs, len_s)
                    pred_y = (pred // 28) * 8.0 + 4  # (bs, len_s)
                    # To(0, 112)
                    pred_x = torch.floor((pred_x + 0.5) / 224 * 112).int()
                    pred_y = torch.floor((pred_y + 0.5) / 224 * 112).int()
                    # (0,112) +delta 防溢出
                    pred_x = pred_x + dx
                    pred_y = pred_y + dy

                    index1 = (pred_x > 111)
                    pred_x[index1] = 111
                    index2 = (pred_x < 0)
                    pred_x[index2] = 0
                    index1 = (pred_y > 111)
                    pred_y[index1] = 111
                    index2 = (pred_y < 0)
                    pred_y[index2] = 0

                    # (0, 112) To (0, 224)
                    pred_x = torch.floor((pred_x.float() + 0.5) / 112 * 224).int()
                    pred_y = torch.floor((pred_y.float() + 0.5) / 112 * 224).int()

                    out_delta_dict['final_pred_x'] = pred_x
                    out_delta_dict['final_pred_y'] = pred_y

                    decoder_dict.update(out_delta_dict)
                    # 主要用logprobs, final_pred_x, final_pred_y

                    return decoder_dict

                elif mode == 'test':
                    feats, feats_delta = self.encoder(img)
                    # 加上一个feats delta的提取器
                    feats_delta = self.delta_encoder(feats_delta)

                    decoder_dict = self.decoder(feats, feats_delta, pre_v2, pre_v1, mode=mode, temperature=temperature)
                    rnn_h = decoder_dict['last_rnn_h']
                    bs = rnn_h.shape[0]
                    time_steps = rnn_h.shape[1]
                    logits = decoder_dict['logits'][:, :, :-1].view(bs, time_steps, 28, 28)
                    out_delta_dict = self.delta_model(rnn_h, logits, feats_delta, mode=mode, temperature=temperature)

                    # get final vertex
                    delta_pred = out_delta_dict['delta_pred']  # (bs, len_s=71, ignore EOS)
                    pred = decoder_dict['pred_polys']  # (bs, len_s=71)

                    dx = delta_pred % 15 - 7  # (bs, len_s)
                    dy = (delta_pred // 15) - 7

                    # To(0, 224)
                    pred_x = (pred % 28) * 8.0 + 4  # (bs, len_s)
                    pred_y = (pred // 28) * 8.0 + 4  # (bs, len_s)
                    # To(0, 112)
                    pred_x = torch.floor((pred_x + 0.5) / 224 * 112).int()
                    pred_y = torch.floor((pred_y + 0.5) / 224 * 112).int()
                    # (0,112) +delta 防溢出
                    pred_x = pred_x + dx
                    pred_y = pred_y + dy

                    index1 = (pred_x > 111)
                    pred_x[index1] = 111
                    index2 = (pred_x < 0)
                    pred_x[index2] = 0
                    index1 = (pred_y > 111)
                    pred_y[index1] = 111
                    index2 = (pred_y < 0)
                    pred_y[index2] = 0

                    # (0, 112) To (0, 224)
                    pred_x = torch.floor((pred_x.float() + 0.5) / 112 * 224).int()
                    pred_y = torch.floor((pred_y.float() + 0.5) / 112 * 224).int()

                    out_delta_dict['final_pred_x'] = pred_x
                    out_delta_dict['final_pred_y'] = pred_y

                    decoder_dict.update(out_delta_dict)

                    return decoder_dict

                elif mode == 'interaction_loop':
                    feats, feats_delta = self.encoder(img)
                    # 加上一个feats delta的提取器
                    feats_delta = self.delta_encoder(feats_delta)

                    decoder_dict = self.decoder(feats, feats_delta, pre_v2, pre_v1, mode=mode,
                                                temperature=temperature, gt_28=gt_28)
                    rnn_h = decoder_dict['last_rnn_h']
                    bs = rnn_h.shape[0]
                    time_steps = rnn_h.shape[1]
                    logits = decoder_dict['logits'][:, :, :-1].view(bs, time_steps, 28, 28)  # 不行就改成one-hot结果
                    out_delta_dict = self.delta_model(rnn_h, logits, feats_delta, mode=mode, temperature=temperature)

                    # get final vertex
                    delta_pred = out_delta_dict['delta_pred']  # (bs, len_s=71, ignore EOS)
                    pred = decoder_dict['pred_polys']  # (bs, len_s=71)

                    dx = delta_pred % 15 - 7  # (bs, len_s)
                    dy = (delta_pred // 15) - 7

                    # To(0, 224)
                    pred_x = (pred % 28) * 8.0 + 4  # (bs, len_s)
                    pred_y = (pred // 28) * 8.0 + 4  # (bs, len_s)
                    # To(0, 112)
                    pred_x = torch.floor((pred_x + 0.5) / 224 * 112).int()
                    pred_y = torch.floor((pred_y + 0.5) / 224 * 112).int()
                    # (0,112) +delta 防溢出
                    pred_x = pred_x + dx
                    pred_y = pred_y + dy

                    index1 = (pred_x > 111)
                    pred_x[index1] = 111
                    index2 = (pred_x < 0)
                    pred_x[index2] = 0
                    index1 = (pred_y > 111)
                    pred_y[index1] = 111
                    index2 = (pred_y < 0)
                    pred_y[index2] = 0

                    # (0, 112) To (0, 224)
                    pred_x = torch.floor((pred_x.float() + 0.5) / 112 * 224).int()
                    pred_y = torch.floor((pred_y.float() + 0.5) / 112 * 224).int()

                    out_delta_dict['final_pred_x'] = pred_x
                    out_delta_dict['final_pred_y'] = pred_y

                    decoder_dict.update(out_delta_dict)

                    return decoder_dict

            else:
                if mode == 'train_ce':
                    feats, feats_delta = self.encoder(img)
                    return self.decoder(feats, feats_delta, pre_v2, pre_v1, mode=mode, temperature=temperature)

                elif mode == 'train_rl':
                    feats, feats_delta = self.encoder(img)
                    return self.decoder(feats, feats_delta, pre_v2, pre_v1, mode=mode, temperature=temperature)

                elif mode == 'test' and self.predict_delta:  # 这个实际上不会走
                    feats, feats_delta = self.encoder(img)
                    outdict = self.decoder(feats, feats_delta, pre_v2, pre_v1, mode=mode, temperature=temperature)

                    delta_pred = outdict['delta_pred']  # (bs, len_s=71, ignore EOS)
                    pred = outdict['pred_polys']  # (bs, len_s=71)

                    dx = delta_pred % 15 - 7  # (bs, len_s)
                    dy = (delta_pred // 15) - 7

                    # To(0, 224)
                    pred_x = (pred % 28) * 8.0 + 4   # (bs, len_s)
                    pred_y = (pred // 28) * 8.0 + 4  # (bs, len_s)
                    # To(0, 112)
                    pred_x = (((pred_x + 0.5) / 224) * 112).int()
                    pred_y = (((pred_y + 0.5) / 224) * 112).int()

                    # (0,112) +delta 防溢出
                    pred_x = pred_x + dx
                    pred_y = pred_y + dy

                    index1 = (pred_x > 111)
                    pred_x[index1] = 111
                    index2 = (pred_x < 0)
                    pred_x[index2] = 0
                    index1 = (pred_y > 111)
                    pred_y[index1] = 111
                    index2 = (pred_y < 0)
                    pred_y[index2] = 0

                    # To (0, 224)
                    pred_x = torch.floor((pred_x + 0.5) / 112 * 224).int()
                    pred_y = torch.floor((pred_y + 0.5) / 112 * 224).int()

                    outdict['final_pred_x'] = pred_x
                    outdict['final_pred_y'] = pred_y

                    return outdict
                elif mode == 'test':  # train_FPN_lstm的val实际上走这里
                    feats, feats_delta = self.encoder(img)
                    return self.decoder(feats, feats_delta, pre_v2, pre_v1, mode=mode, temperature=temperature)
        else:
            # beam size > 1
            return None

    def append(self, module):
        if module == 'delta_module':
            self.decoder.predict_delta = True
            self.decoder.delta_downsample_conv = nn.Conv2d(in_channels=17,
                                                       out_channels=17,
                                                       kernel_size=3,
                                                       stride=2,
                                                       padding=1,
                                                       bias=True)
            self.decoder.delta_conv1 = nn.Conv2d(in_channels=128 + 17,
                                             out_channels=64,
                                             kernel_size=3,
                                             padding=1,
                                             stride=1,
                                             bias=True)

            # init
            nn.init.kaiming_normal_(self.decoder.delta_conv1.weight, mode='fan_in', nonlinearity='relu')
            if self.decoder.delta_conv1.bias is not None:
                nn.init.constant_(self.decoder.delta_conv1.bias, 0)

            self.decoder.delta_bn1 = nn.BatchNorm2d(64)
            # init
            nn.init.constant_(self.decoder.delta_bn1.weight, 1)
            nn.init.constant_(self.decoder.delta_bn1.bias, 0)

            self.decoder.delta_res1_conv = nn.Conv2d(in_channels=128 + 17,
                                                 out_channels=64,
                                                 kernel_size=1,
                                                 stride=1,
                                                 bias=True)

            nn.init.kaiming_normal_(self.decoder.delta_res1_conv.weight, mode='fan_in', nonlinearity='relu')
            if self.decoder.delta_res1_conv.bias is not None:
                nn.init.constant_(self.decoder.delta_res1_conv.bias, 0)

            self.decoder.delta_res1_bn = nn.BatchNorm2d(64)
            # init
            nn.init.constant_(self.decoder.delta_res1_bn.weight, 1)
            nn.init.constant_(self.decoder.delta_res1_bn.bias, 0)

            self.decoder.delta_relu1 = nn.ReLU(inplace=True)  # out: (N, 64, 14, 14)

            self.decoder.delta_conv2 = nn.Conv2d(in_channels=64,
                                             out_channels=16,
                                             kernel_size=3,
                                             padding=1,
                                             stride=1,
                                             bias=True)

            # init
            nn.init.kaiming_normal_(self.decoder.delta_conv2.weight, mode='fan_in', nonlinearity='relu')
            if self.decoder.delta_conv2.bias is not None:
                nn.init.constant_(self.decoder.delta_conv2.bias, 0)

            self.decoder.delta_bn2 = nn.BatchNorm2d(16)
            # init
            nn.init.constant_(self.decoder.delta_bn2.weight, 1)
            nn.init.constant_(self.decoder.delta_bn2.bias, 0)

            self.decoder.delta_res2_conv = nn.Conv2d(in_channels=64,
                                                 out_channels=16,
                                                 kernel_size=1,
                                                 stride=1,
                                                 bias=True)
            # init
            nn.init.kaiming_normal_(self.decoder.delta_res2_conv.weight, mode='fan_in', nonlinearity='relu')
            if self.decoder.delta_res2_conv.bias is not None:
                nn.init.constant_(self.decoder.delta_res2_conv.bias, 0)

            self.decoder.delta_res2_bn = nn.BatchNorm2d(16)
            nn.init.constant_(self.decoder.delta_res2_bn.weight, 1)
            nn.init.constant_(self.decoder.delta_res2_bn.bias, 0)


            self.decoder.delta_relu2 = nn.ReLU(inplace=True)

            self.decoder.delta_final = nn.Linear(16 * 14 * 14, 14 * 14)  # 输出一个14*14的grid
            # init
            nn.init.xavier_uniform_(self.decoder.delta_final.weight)
            nn.init.constant_(self.decoder.delta_final.bias, 0)

            self.decoder.use_delta_attn = True
                # delta attn
            self.decoder.delta_fc_att = nn.Linear(in_features=self.feat_channels,
                                              out_features=1)
            # init
            nn.init.xavier_uniform_(self.decoder.delta_fc_att.weight)
            nn.init.constant_(self.decoder.delta_fc_att.bias, 0)

            self.decoder.delta_conv_att = nn.Conv2d(in_channels=17,
                                                out_channels=self.feat_channels,
                                                kernel_size=1,
                                                padding=0,
                                                bias=True)
            # init
            nn.init.kaiming_normal_(self.decoder.delta_conv_att.weight, mode='fan_in', nonlinearity='relu')
            if self.decoder.delta_conv_att.bias is not None:
                nn.init.constant_(self.decoder.delta_conv_att.bias, 0)


if __name__=='__main__':
    x = PolygonModel(predict_delta=True)
    print(x)
