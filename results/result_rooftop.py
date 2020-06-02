# -*- coding: utf-8 -*-
from __future__ import print_function

import glob
import scipy.io as scio
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import argparse
import json
import os
import torchvision.transforms as transforms
from models.model import PolygonModel
from utils import *

class RoofTop(Dataset):
    def __init__(self, path, seq_len, transform=None):
        super(RoofTop, self).__init__()
        self.path = path
        self.seq_len = seq_len
        self.transform = transform
        self.img_path = '/data/duye/Aerial_Imagery/Rooftop/'+self.path+'_new/img/'
        self.lbl_path = '/data/duye/Aerial_Imagery/Rooftop/' + self.path + '_new/label/'
        self.meta = '/data/duye/Aerial_Imagery/Rooftop/'+ self.path + '_new/' + \
                    self.path + '_meta.json'
        self.meta = json.load(open(self.meta))
        self.total_count = self.meta['total_count']

    def __len__(self):
        return self.total_count

    def __getitem__(self, index):
        try:
            img = Image.open(os.path.join(self.img_path, str(index) + '.JPG')).convert('RGB')
        except FileNotFoundError:
            return None
        assert not (img is None)

        W = img.width  # 224
        H = img.height  # 224

        # json file
        js = json.load(open(os.path.join(self.lbl_path, str(index) + '.json')))
        label = js['polygon']  # 多边形顶点
        left_WH = js['left_WH']  # 裁减图片左上角坐标在原图中的WH
        object_WH = js['object_WH']  # 裁减下来的图片(scale到224,224之前)的WH
        origion_WH = js['origion_WH']  # 原始图片的WH
        WH = {'left_WH': left_WH, 'object_WH': object_WH, 'origion_WH': origion_WH}
        origin_polygon = js['origin_polygon']
        point_num = len(label)
        polygon = np.array(label, dtype=np.float)  # (point_num, 2)
        polygon_GT = np.zeros((self.seq_len, 2), dtype=np.int32)

        # To (0,1)
        polygon = polygon / (W * 1.0)

        # To (0, g) 道格拉斯算法, 去重
        polygon = poly01_to_poly0g(polygon, 28)
        label_onehot = np.zeros([self.seq_len, 28 * 28 + 1])
        label_index = np.zeros([self.seq_len])
        point_scaled = []

        mask_final = np.zeros(self.seq_len, dtype=np.int)
        mask_delta = np.zeros(self.seq_len - 1, dtype=np.int)

        cnt = 0
        point_num = len(polygon)
        polygon01 = np.zeros([self.seq_len - 1, 2])
        tmp = np.array(poly0g_to_poly01(polygon, 28) * W, dtype=np.int)  # To(0, 224)

        ind = 0
        for vert in origin_polygon:
            x = int(vert[0])
            y = int(vert[1])
            polygon_GT[ind, 0] = x
            polygon_GT[ind, 1] = y
            ind += 1
            if ind >= self.seq_len:
                break


        if point_num <= 70:
            polygon01[:point_num] = tmp
        else:
            polygon01[:70] = tmp[:70]

        if point_num < self.seq_len:  # < 70
            for point in polygon:
                x = point[0]
                y = point[1]
                indexs = y * 28 + x
                label_index[cnt] = indexs
                label_onehot[cnt, indexs] = 1
                cnt += 1
                point_scaled.append([x, y])
            mask_final[:cnt + 1] = 1
            mask_delta[:cnt] = 1
            # end point
            label_index[cnt] = 28 * 28
            label_onehot[cnt, 28 * 28] = 1
            cnt += 1
            for ij in range(cnt, self.seq_len):
                label_index[ij] = 28 * 28
                label_onehot[ij, 28 * 28] = 1
                cnt += 1
        else:
            for iii in range(self.seq_len - 1):
                point = polygon[iii]  # 取点
                x = point[0]
                y = point[1]
                xx = x
                yy = y
                indexs = yy * 28 + xx
                label_index[cnt] = indexs
                label_onehot[cnt, indexs] = 1
                cnt += 1
                point_scaled.append([xx, yy])
            mask_final[:cnt + 1] = 1  # +1才会计算final最后EOS的损失
            mask_delta[:cnt] = 1
            # EOS
            label_index[self.seq_len - 1] = 28 * 28
            label_onehot[self.seq_len - 1, 28 * 28] = 1

        # ToTensor
        if self.transform:
            img = self.transform(img)

        point_scaled = np.array(point_scaled)
        # 边界，edge上的点为1，其余点为0
        edge_mask = np.zeros((28, 28), dtype=np.float)
        edge_mask = get_edge_mask(point_scaled, edge_mask)

        return img, \
               label_onehot[0], \
               label_onehot[:-2], \
               label_onehot[:-1], \
               label_index, \
               edge_mask, \
               mask_final, \
               mask_delta, \
               polygon_GT, \
               polygon01, WH
        # polygon01: 在224*224中的坐标, 而非0-1



def loadRooftop(path, len_s, batch_size, shuffle=True):
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
    Rooftop = RoofTop(path, len_s, transform)
    dataloader = DataLoader(Rooftop, batch_size=batch_size, shuffle=shuffle,
                            drop_last=False)
    print('DataLoader complete!', dataloader)
    return dataloader


devices = 'cuda' if torch.cuda.is_available() else 'cpu'

loader = loadRooftop('train', 71, 1)
def results(self):
    global_step = 0
    ious = []
    for epoch in range(self.max_epoch):
        for step, batch in enumerate(loader):
            global_step += 1
            img = torch.tensor(batch[0], dtype=torch.float).cuda()
            bs = img.shape[0]
            WH = batch[-1]  # WH_dict
            left_WH = WH['left_WH']
            origion_WH = WH['origion_WH']
            object_WH = WH['object_WH']
            # TODO： step1
            with torch.no_grad():
                outdict_sample = self.model(img, mode='test', temperature=self.t1,
                                            temperature2=0.0)

            pred_x = outdict_sample['final_pred_x'].cpu().numpy()
            pred_y = outdict_sample['final_pred_y'].cpu().numpy()
            pred_len = outdict_sample['lengths'].cpu().numpy()
            vertices_GT = []  # (bs, 70, 2)
            vertices_sampling = []
            GT_polys = batch[-2].numpy()  # (bs, 70, 2)
            GT_mask = batch[7]  # (bs, 70)
            for ii in range(bs):
                tmp = []
                all_len = np.sum(GT_mask[ii].numpy())
                cnt_target = GT_polys[ii][:all_len]
                for vert in cnt_target:
                    tmp.append((vert[0],
                                vert[1]))
                vertices_GT.append(tmp)

                tmp = []
                for j in range(pred_len[ii] - 1):
                    vertex = (
                        pred_x[ii][j],
                        pred_y[ii][j]
                    )
                    tmp.append(vertex)
                vertices_sampling.append(tmp)

            # IoU between sampling/greedy and GT
            for ii in range(bs):
                sam = vertices_sampling[ii]
                gt = vertices_GT[ii]
                if len(sam) < 2:
                    ious.append(0.)
                else:
                    iou_sam, _, _ = iou(sam, gt, origion_WH[1][ii], origion_WH[0][ii])
                    ious.append(iou_sam)
            # save image

#TODO: 将三个数据集分别找一些样例，三列：左：泛化性能，中：经过在线学习后的图，右：真值
# 三数据集 rooftop屋顶2个，ADE20K电视一个，ssTEM也找两个，一共5个就可以了
