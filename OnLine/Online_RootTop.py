# -*- coding: utf-8 -*-
from __future__ import print_function
import torch.optim as optim
from models.model import PolygonModel
from utils import *
from dataloader import loadData
import warnings
import torch.nn as nn
import numpy as np
from collections import defaultdict
import losses
import os
from RoofTop import loadRooftop
import argparse
warnings.filterwarnings('ignore')
devices = 'cuda' if torch.cuda.is_available() else 'cpu'


class OnLineTrainer:
    def __init__(self, num_workers=8, update_every=8, save_every=50, t1=0.1, t2=0.1, pre=None):
        self.max_epoch = 20
        self.num_workers = num_workers
        self.update_every = update_every
        self.save_every = save_every
        self.t1 = t1
        self.t2 = t2
        self.model = PolygonModel(predict_delta=True).to(devices)
        if pre != None:
            self.model.load_state_dict(torch.load(pre))
        self.dataloader = loadRooftop('train', 71, self.num_workers)
        self.model.encoder.eval()
        self.model.delta_encoder.eval()
        for n, p in self.model.named_parameters():
            if 'encoder' in n:
                p.requires_grad = False
        self.train_params = [p for p in self.model.parameters() if p.requires_grad==True]
        self.optimizer = optim.Adam(self.train_params,
                                    lr=2e-5,
                                    amsgrad=False)

    def train(self):
        accum = defaultdict(float)
        accum2 = defaultdict(float)
        global_step = 0
        for epoch in range(self.max_epoch):
            for step, batch in enumerate(self.dataloader):
                global_step += 1
                img = torch.tensor(batch[0], dtype=torch.float).cuda()
                bs = img.shape[0]
                WH = batch[-1]  # WH_dict
                left_WH = WH['left_WH']
                origion_WH = WH['origion_WH']
                object_WH = WH['object_WH']
                # TODO： step1
                self.model.delta_model.eval()
                self.model.decoder.train()
                outdict_sample = self.model(img, mode='train_rl', temperature=self.t1,
                                            temperature2=0.0)
                # greedy
                with torch.no_grad():
                    outdict_greedy = self.model(img, mode='train_rl', temperature=0.0)
                # Get RL loss
                sampling_pred_x = outdict_sample['final_pred_x'].cpu().numpy()
                sampling_pred_y = outdict_sample['final_pred_y'].cpu().numpy()
                sampling_pred_len = outdict_sample['lengths'].cpu().numpy()
                greedy_pred_x = outdict_greedy['final_pred_x'].cpu().numpy()
                greedy_pred_y = outdict_greedy['final_pred_y'].cpu().numpy()
                greedy_pred_len = outdict_greedy['lengths'].cpu().numpy()
                sampling_iou = np.zeros(bs, dtype=np.float32)
                greedy_iou = np.zeros(bs, dtype=np.float32)
                vertices_GT = []  # (bs, 70, 2)
                vertices_sampling = []
                vertices_greedy = []
                GT_polys = batch[-2].numpy()  # (bs, 70, 2)
                GT_mask = batch[7]  # (bs, 70)
                for ii in range(bs):
                    scaleW = 224.0 / float(object_WH[0][ii])
                    scaleH = 224.0 / float(object_WH[1][ii])
                    leftW = left_WH[0][ii]
                    leftH = left_WH[1][ii]
                    tmp = []
                    all_len = np.sum(GT_mask[ii].numpy())
                    cnt_target = GT_polys[ii][:all_len]
                    for vert in cnt_target:
                        tmp.append((vert[0] / scaleW + leftW,
                                    vert[1] / scaleH + leftH))
                    vertices_GT.append(tmp)

                    tmp = []
                    for j in range(sampling_pred_len[ii] - 1):
                        vertex = (
                            sampling_pred_x[ii][j] / scaleW + leftW,
                            sampling_pred_y[ii][j] / scaleH + leftH
                        )
                        tmp.append(vertex)
                    vertices_sampling.append(tmp)

                    tmp = []
                    for j in range(greedy_pred_len[ii] - 1):
                        vertex = (
                            greedy_pred_x[ii][j] / scaleW + leftW,
                            greedy_pred_y[ii][j] / scaleH + leftH
                        )
                        tmp.append(vertex)
                    vertices_greedy.append(tmp)
                # IoU between sampling/greedy and GT
                for ii in range(bs):
                    sam = vertices_sampling[ii]
                    gt = vertices_GT[ii]
                    gre = vertices_greedy[ii]
                    if len(sam) < 2:
                        sampling_iou[ii] = 0.
                    else:
                        iou_sam, _, _ = iou(sam, gt, origion_WH[1][ii], origion_WH[0][ii])
                        sampling_iou[ii] = iou_sam
                    if len(gre) < 2:
                        greedy_iou[ii] = 0.
                    else:
                        iou_gre, _, _ = iou(gre, gt, origion_WH[1][ii], origion_WH[0][ii])
                        greedy_iou[ii] = iou_gre
                logprobs = outdict_sample['log_probs']
                # 强化学习损失，logprob是两个logprob加和
                loss = losses.self_critical_loss(logprobs, outdict_sample['lengths'],
                                                 torch.from_numpy(sampling_iou).to(devices),
                                                 torch.from_numpy(greedy_iou).to(devices))
                self.model.zero_grad()
                nn.utils.clip_grad_norm_(self.model.parameters(), 40)
                loss.backward()
                self.optimizer.step()  # 更新参数
                accum['loss_total'] += loss
                accum['sampling_iou'] += np.mean(sampling_iou)
                accum['greedy_iou'] += np.mean(greedy_iou)
                # 打印损失
                print('Update {}, RL training of Main decoder, loss {}, model IoU {}'.format(step + 1,
                                                                                 accum['loss_total'],
                                                                                 accum['greedy_iou']))
                accum = defaultdict(float)
                # TODO:训练delta_model decoder step2
                self.model.decoder.eval()
                self.model.delta_model.train()
                outdict_sample = self.model(img, mode='train_rl',
                                            temperature=0.0,
                                            temperature2=self.t2)
                # greedy
                with torch.no_grad():
                    outdict_greedy = self.model(img, mode='train_rl',
                                           temperature=0.0,
                                           temperature2=0.0)
                # Get RL loss
                sampling_pred_x = outdict_sample['final_pred_x'].cpu().numpy()
                sampling_pred_y = outdict_sample['final_pred_y'].cpu().numpy()
                sampling_pred_len = outdict_sample['lengths'].cpu().numpy()
                greedy_pred_x = outdict_greedy['final_pred_x'].cpu().numpy()
                greedy_pred_y = outdict_greedy['final_pred_y'].cpu().numpy()
                greedy_pred_len = outdict_greedy['lengths'].cpu().numpy()
                sampling_iou = np.zeros(bs, dtype=np.float32)
                greedy_iou = np.zeros(bs, dtype=np.float32)
                vertices_GT = []  # (bs, 70, 2)
                vertices_sampling = []
                vertices_greedy = []
                GT_polys = batch[-2].numpy()  # (bs, 70, 2)
                GT_mask = batch[7]  # (bs, 70)

                for ii in range(bs):
                    scaleW = 224.0 / float(object_WH[0][ii])
                    scaleH = 224.0 / float(object_WH[1][ii])
                    leftW = left_WH[0][ii]
                    leftH = left_WH[1][ii]
                    tmp = []
                    all_len = np.sum(GT_mask[ii].numpy())
                    cnt_target = GT_polys[ii][:all_len]
                    for vert in cnt_target:
                        tmp.append((vert[0] / scaleW + leftW,
                                    vert[1] / scaleH + leftH))
                    vertices_GT.append(tmp)

                    tmp = []
                    for j in range(sampling_pred_len[ii] - 1):
                        vertex = (
                            sampling_pred_x[ii][j] / scaleW + leftW,
                            sampling_pred_y[ii][j] / scaleH + leftH
                        )
                        tmp.append(vertex)
                    vertices_sampling.append(tmp)

                    tmp = []
                    for j in range(greedy_pred_len[ii] - 1):
                        vertex = (
                            greedy_pred_x[ii][j] / scaleW + leftW,
                            greedy_pred_y[ii][j] / scaleH + leftH
                        )
                        tmp.append(vertex)
                    vertices_greedy.append(tmp)

                # IoU between sampling/greedy and GT
                for ii in range(bs):
                    sam = vertices_sampling[ii]
                    gt = vertices_GT[ii]
                    gre = vertices_greedy[ii]

                    if len(sam) < 2:
                        sampling_iou[ii] = 0.
                    else:
                        iou_sam, _, _ = iou(sam, gt, origion_WH[1][ii], origion_WH[0][ii])
                        sampling_iou[ii] = iou_sam

                    if len(gre) < 2:
                        greedy_iou[ii] = 0.
                    else:
                        iou_gre, _, _ = iou(gre, gt, origion_WH[1][ii], origion_WH[0][ii])
                        greedy_iou[ii] = iou_gre

                # TODO:
                logprobs = outdict_sample['delta_logprob']
                # 强化学习损失，logprob是两个logprob加和
                loss = losses.self_critical_loss(logprobs, outdict_sample['lengths'],
                                                 torch.from_numpy(sampling_iou).to(devices),
                                                 torch.from_numpy(greedy_iou).to(devices))
                self.model.zero_grad()
                nn.utils.clip_grad_norm_(self.model.parameters(), 40)
                loss.backward()
                self.optimizer.step()
                accum2['loss_total'] += loss
                accum2['sampling_iou'] += np.mean(sampling_iou)
                accum2['greedy_iou'] += np.mean(greedy_iou)
                # 打印损失
                print('Update {}, RL training of Second decoder, loss {}, model IoU {}'.format(step + 1,
                                                                                 accum2['loss_total'],
                                                                                 accum2['greedy_iou']))
                accum2 = defaultdict(float)

                if global_step % self.save_every == 0:
                    print('Saving training parameters after Updating...')
                    save_dir = '/data/duye/pretrained_models/OnLineTraining_RoofTop/'
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    torch.save(self.model.state_dict(), save_dir + str(global_step) + '.pth')


if __name__ == '__main__':
    dir = '/data/duye/pretrained_models/'
    pre = 'FPNRLtrain/ResNext_Plus_RL2_retain_Epoch1-Step4000_ValIoU0.6316584628283326.pth'
    online_trainer = OnLineTrainer(pre=dir+pre, num_workers=4)
    online_trainer.train()