# -*- coding: utf-8 -*-
from __future__ import print_function
import torch.optim as optim
from models.model import PolygonModel
from utils import *
from dataloader import loadData
import warnings
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from collections import defaultdict
import losses
import os

import glob
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings('ignore')
devices = 'cuda' if torch.cuda.is_available() else 'cpu'
class Ade20K(Dataset):
    def __init__(self, path=None, seq_len=71, transform=None):
        super(Ade20K, self).__init__()
        self.path = path
        self.seq_len = seq_len
        self.transform = transform
        # print(self.transform)
        self.img_path = '/data/duye/ADE20K/training/'
        self.lbl_path = '/data/duye/ADE20K/train_new/label/*.png'
        self.labels = glob.glob(self.lbl_path)
        # print(self.labels)
        self.total_count = len(self.labels)

    def __len__(self):
        return self.total_count

    def __getitem__(self, index):
        label = Image.open(self.labels[index])
        label_index = self.labels[index].split('_')[2]
            # 相应的txt文件
        txt_file = '/data/duye/ADE20K/train_new/img/img_' + label_index + '.txt'
        # print(txt_file)
            # 打开txt获取相应的img路径
        with open(txt_file, "r") as f:  # 打开文件
            img_path = f.readline().replace('\n', '')  # 读取文件
            # 提取路径
        img_path = self.img_path + img_path[34:]
            # print('img_path:', img_path)
            # 提取文件
        img = Image.open(img_path).convert('RGB')

        W = img.width
        H = img.height
        # 根据label
        label = np.array(label)  # (H, W)
        Hs, Ws = np.where(label == np.max(label))
        minH = np.min(Hs)
        maxH = np.max(Hs)
        minW = np.min(Ws)
        maxW = np.max(Ws)
        curW = maxW - minW
        curH = maxH - minH
        extendrate = 0.1
        extendW = round(curW * extendrate)
        extendH = round(curH * extendrate)
        leftW = np.maximum(minW - extendW, 0)
        leftH = np.maximum(minH - extendH, 0)
        rightW = np.minimum(maxW + extendW, W)
        rightH = np.minimum(maxH + extendH, H)
        objectW = rightW - leftW
        objectH = rightH - leftH
        img_new = img.crop(box=(leftW, leftH, rightW, rightH)).resize((224, 224), Image.BILINEAR)
        img_new = self.transform(img_new)
        left_WH = [leftW, leftH]
        object_WH = [objectW, objectH]
        origion_WH = [W, H]
        # 记录Object WH / WH /left WH
        gt = label
        WH = {'left_WH': left_WH, 'object_WH': object_WH, 'origion_WH': origion_WH}
        return img_new, gt, WH


def collate_fn(batch):

    sample1 = batch[0]
    sample2 = batch[1]
    sample3 = batch[2]
    sample4 = batch[3]

    return {
        's1': sample1,
        's2': sample2,
        's3': sample3,
        's4': sample4
    }

def loadAde20K(batch_size, shuffle=False):
    """

    :param path: train/test/val，数据集名
    :param data_num: 实际无用的..
    :param len_s: step_num, max-num of vertex,
                Based on this statistics, we choose a hard limit of 70 time steps for our RNN,
                taking also GPU memory requirements into account.
    :param batch_size: bs
    :return: dataloader, (bs, img_tensor, first, yt-2,yt-1, label_index)
    """
    transform = transforms.Compose([transforms.ToTensor(), ])
    Ade = Ade20K(transform=transform)
    dataloader = DataLoader(Ade, batch_size=batch_size, shuffle=shuffle,
                            drop_last=True, collate_fn=collate_fn)
    print('DataLoader complete!', dataloader)
    return dataloader



class OnLineTrainer:
    def __init__(self, num_workers=8, update_every=8, save_every=100, t1=0.1, t2=0.1, pre=None):
        self.num_workers = num_workers
        self.update_every = update_every
        self.save_every = save_every
        self.t1 = t1
        self.t2 = t2
        self.max_epoch = 2
        self.model = PolygonModel(predict_delta=True).to(devices)
        if pre != None:
            self.model.load_state_dict(torch.load(pre))
        self.dataloader = loadAde20K(batch_size=4)
        self.model.encoder.eval()
        self.model.delta_encoder.eval()
        for n, p in self.model.named_parameters():
            if 'encoder' in n:
                p.requires_grad = False
        self.train_params = [p for p in self.model.parameters() if p.requires_grad==True]
        self.optimizer = optim.Adam(self.train_params,
                                    lr=2e-5,
                                    amsgrad=False)

    # TODO: 1. 多训练一下Rooftop把精度提升上去(现在是每50次保存一下)
    #  2. ADE20K train数据集加载不对，gt维度不匹配
    #  3. cityscape每20保存一下，看看是否有不妥？

    def train(self):
        accum = defaultdict(float)
        accum2 = defaultdict(float)
        global_step = 0.
        for epoch in range(self.max_epoch):
            for step, batch in enumerate(self.dataloader):
                global_step += 1
                b = []
                b.append(batch['s1'])
                b.append(batch['s2'])
                b.append(batch['s3'])
                b.append(batch['s4'])

                # print(b1[0].shape)  # (3, 224, 224)

                img = torch.cat([b[0][0].unsqueeze(0),
                           b[1][0].unsqueeze(0),
                           b[2][0].unsqueeze(0),
                           b[3][0].unsqueeze(0)], dim=0).to(devices)

                bs = img.shape[0]

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

                vertices_sampling = []
                vertices_greedy = []
                for ii in range(bs):
                    WH = b[ii][-1]
                    object_WH = WH['object_WH']
                    left_WH = WH['left_WH']
                    #     WH = {'left_WH': left_WH, 'object_WH': object_WH, 'origion_WH': origion_WH}
                    scaleW = 224.0 / float(object_WH[0])
                    scaleH = 224.0 / float(object_WH[1])
                    leftW = left_WH[0]
                    leftH = left_WH[1]

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
                    gt = b[ii][1]  # (H, W)
                    WH = b[ii][-1]
                    origion_WH = WH['origion_WH']
                    sam = vertices_sampling[ii]
                    gre = vertices_greedy[ii]
                    if len(sam) < 2:
                        sampling_iou[ii] = 0.
                    else:
                        # iou_sam, _, _ = iou(sam, gt, origion_WH[1], origion_WH[0])
                        # sampling_iou[ii] = iou_sam
                        img1 = Image.new('L', (origion_WH[0], origion_WH[1]), 0)
                        ImageDraw.Draw(img1).polygon(sam, outline=1, fill=1)
                        mask1 = np.array(img1)  # (h, w)
                        intersection = np.logical_and(mask1, gt)  # 都是1
                        union = np.logical_or(mask1, gt)  # 有个1
                        nu = np.sum(intersection)
                        de = np.sum(union)
                        sampling_iou[ii] = nu*1.0 / de if de!=0 else 0.
                    if len(gre) < 2:
                        greedy_iou[ii] = 0.
                    else:
                        # iou_gre, _, _ = iou(gre, gt, origion_WH[1], origion_WH[0])
                        # greedy_iou[ii] = iou_gre
                        img1 = Image.new('L', (origion_WH[0], origion_WH[1]), 0)
                        ImageDraw.Draw(img1).polygon(gre, outline=1, fill=1)
                        mask1 = np.array(img1)  # (h, w)
                        intersection = np.logical_and(mask1, gt)  # 都是1
                        union = np.logical_or(mask1, gt)  # 有个1
                        nu = np.sum(intersection)
                        de = np.sum(union)
                        greedy_iou[ii] = nu*1.0 / de if de!=0 else 0.
                logprobs = outdict_sample['log_probs']
                # 强化学习损失，logprob是两个logprob加和
                loss = losses.self_critical_loss(logprobs, outdict_sample['lengths'],
                                                 torch.from_numpy(sampling_iou).to(devices),
                                                 torch.from_numpy(greedy_iou).to(devices))
                self.model.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 40)
                self.optimizer.step()   # 更新参数

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
                vertices_sampling = []
                vertices_greedy = []
                for ii in range(bs):
                    WH = b[ii][-1]
                    object_WH = WH['object_WH']
                    left_WH = WH['left_WH']
                    #     WH = {'left_WH': left_WH, 'object_WH': object_WH, 'origion_WH': origion_WH}
                    scaleW = 224.0 / float(object_WH[0])
                    scaleH = 224.0 / float(object_WH[1])
                    leftW = left_WH[0]
                    leftH = left_WH[1]

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
                    gt = b[ii][1]  # (H, W)
                    WH = b[ii][-1]
                    origion_WH = WH['origion_WH']
                    sam = vertices_sampling[ii]
                    gre = vertices_greedy[ii]
                    if len(sam) < 2:
                        sampling_iou[ii] = 0.
                    else:
                        # iou_sam, _, _ = iou(sam, gt, origion_WH[1], origion_WH[0])
                        # sampling_iou[ii] = iou_sam
                        img1 = Image.new('L', (origion_WH[0], origion_WH[1]), 0)
                        ImageDraw.Draw(img1).polygon(sam, outline=1, fill=1)
                        mask1 = np.array(img1)  # (h, w)
                        intersection = np.logical_and(mask1, gt)  # 都是1
                        union = np.logical_or(mask1, gt)  # 有个1
                        nu = np.sum(intersection)
                        de = np.sum(union)
                        sampling_iou[ii] = nu*1.0 / de if de!=0 else 0.
                    if len(gre) < 2:
                        greedy_iou[ii] = 0.
                    else:
                        # iou_gre, _, _ = iou(gre, gt, origion_WH[1], origion_WH[0])
                        # greedy_iou[ii] = iou_gre
                        img1 = Image.new('L', (origion_WH[0], origion_WH[1]), 0)
                        ImageDraw.Draw(img1).polygon(gre, outline=1, fill=1)
                        mask1 = np.array(img1)  # (h, w)
                        intersection = np.logical_and(mask1, gt)  # 都是1
                        union = np.logical_or(mask1, gt)  # 有个1
                        nu = np.sum(intersection)
                        de = np.sum(union)
                        greedy_iou[ii] = nu*1.0 / de if de!=0 else 0.
                logprobs = outdict_sample['log_probs']
                # 强化学习损失，logprob是两个logprob加和
                loss = losses.self_critical_loss(logprobs, outdict_sample['lengths'],
                                                 torch.from_numpy(sampling_iou).to(devices),
                                                 torch.from_numpy(greedy_iou).to(devices))
                self.model.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 40)
                self.optimizer.step()   # 更新参数
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
                    save_dir = '/data/duye/pretrained_models/OnLineTraining_ADE20K/'
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    torch.save(self.model.state_dict(), save_dir + str(global_step) + '.pth')


if __name__ == '__main__':
    dir = '/data/duye/pretrained_models/'
    pre = 'FPNRLtrain/ResNext_Plus_RL2_retain_Epoch1-Step4000_ValIoU0.6316584628283326.pth'
    online_trainer = OnLineTrainer(pre=dir+pre, num_workers=4)
    online_trainer.train()