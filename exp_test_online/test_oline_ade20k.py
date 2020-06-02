# -*- coding: utf-8 -*-
from __future__ import print_function
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import glob
import json
import os
import argparse
from models.model import PolygonModel
import torchvision.transforms as transforms
from utils import *
class Ade20K(Dataset):
    def __init__(self, path=None, seq_len=71, transform=None):
        super(Ade20K, self).__init__()
        self.path = path
        self.seq_len = seq_len
        self.transform = transform
        # print(self.transform)
        self.img_path = '/data/duye/ADE20K/validation/'
        self.lbl_path = '/data/duye/ADE20K/val_new/label/*.png'
        self.labels = glob.glob(self.lbl_path)
        # print(self.labels)
        self.total_count = len(self.labels)

    def __len__(self):
        return self.total_count

    def __getitem__(self, index):
        label = Image.open(self.labels[index])
        label_index = self.labels[index].split('_')[2]
            # 相应的txt文件
        txt_file = '/data/duye/ADE20K/val_new/img/img_' + label_index + '.txt'
            # print(txt_file)
            # 打开txt获取相应的img路径
        with open(txt_file, "r") as f:  # 打开文件
            img_path = f.readline().replace('\n', '')  # 读取文件
            # 提取路径
        img_path = self.img_path + img_path[36:]
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
        extendW = int(round(curW * extendrate))
        extendH = int(round(curH * extendrate))
        leftW = np.maximum(minW - extendW, 0)
        leftH = np.maximum(minH - extendH, 0)
        rightW = np.minimum(maxW + extendW, W)
        rightH = np.minimum(maxH + extendH, H)
        objectW = rightW - leftW
        objectH = rightH - leftH
        # print(leftH, rightH, leftW, rightW)
        img_new = img.crop(box=(leftW, leftH, rightW, rightH)).resize((224, 224), Image.BILINEAR)
        img_new = self.transform(img_new)
        left_WH = [leftW, leftH]
        object_WH = [objectW, objectH]
        origion_WH = [W, H]
        # 记录Object WH / WH /left WH
        gt = label
        WH = {'left_WH': left_WH, 'object_WH': object_WH, 'origion_WH': origion_WH}
        return img_new, gt, WH

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
                            drop_last=False)
    print('DataLoader complete!', dataloader)
    return dataloader

# 测试得分
devices = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_score_ADE20K(pre, saved=False, maxnum=float('inf')):
    model = PolygonModel(predict_delta=True).to(devices)
    # pre = 'ResNext_Plus_RL2_retain_Epoch1-Step4000_ValIoU0.6316584628283326.pth'
    dirs = '/data/duye/pretrained_models/OnLineTraining_ADE20K/' + pre
    model.load_state_dict(torch.load(dirs))
    model.eval()

    iou = []
    print('starting.....')
    img_PATH = '/data/duye/ADE20K/validation/'
    lbl_path = '/data/duye/ADE20K/val_new/label/*.png'
    labels = glob.glob(lbl_path)
    for label in labels:
        name = label
        label = Image.open(label)
        label_index = name.split('_')[2]
        # 相应的txt文件
        txt_file = '/data/duye/ADE20K/val_new/img/img_' + label_index + '.txt'
        with open(txt_file, "r") as f:  # 打开文件
            img_path = f.readline().replace('\n', '')  # 读取文件
            # 提取路径
        img_path = img_PATH + img_path[36:]
        # raw image
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
        extendrate = 0.10
        extendW = int(round(curW * extendrate))
        extendH = int(round(curH * extendrate))
        leftW = np.maximum(minW - extendW, 0)
        leftH = np.maximum(minH - extendH, 0)
        rightW = np.minimum(maxW + extendW, W)
        rightH = np.minimum(maxH + extendH, H)
        objectW = rightW - leftW
        objectH = rightH - leftH
        # print(leftH, rightH, leftW, rightW)
        # img_new = img.crop(box=(leftW, leftH, rightW, rightH)).resize((224, 224), Image.BILINEAR)
        I = np.array(img)
        I_obj = I[leftH:rightH, leftW:rightW, :]
        # To PIL image
        I_obj_img = Image.fromarray(I_obj)
        # resize
        I_obj_img = I_obj_img.resize((224, 224), Image.BILINEAR)
        I_obj_new = np.array(I_obj_img)  # (H, W, C)
        I_obj_new = I_obj_new.transpose(2, 0, 1)  # (C, H, W)
        I_obj_new = I_obj_new / 255.0
        I_obj_tensor = torch.from_numpy(I_obj_new)  # (C, H, W)
        I_obj_tensor = torch.tensor(I_obj_tensor.unsqueeze(0), dtype=torch.float).cuda()

        color = [np.random.randint(0, 255) for _ in range(3)]
        color += [100]
        color = tuple(color)

        with torch.no_grad():
            pre_v2 = None
            pre_v1 = None
            result_dict = model(I_obj_tensor, pre_v2, pre_v1, mode='test', temperature=0.0)  # (bs, seq_len)

        # [0, 224] index 0: only one sample in mini-batch here
        pred_x = result_dict['final_pred_x'].cpu().numpy()[0]
        pred_y = result_dict['final_pred_y'].cpu().numpy()[0]
        pred_lengths = result_dict['lengths'].cpu().numpy()[0]
        pred_len = np.sum(pred_lengths) - 1  # sub EOS
        vertices1 = []

        scaleW = 224.0 / float(objectW)
        scaleH = 224.0 / float(objectH)
        # Get the pred poly
        for i in range(pred_len):
            vert = (pred_x[i] / scaleW + leftW,
                    pred_y[i] / scaleH + leftH)
            vertices1.append(vert)
        img1 = Image.new('L', (W, H), 0)
        ImageDraw.Draw(img1).polygon(vertices1, outline=1, fill=1)
        pre_mask = np.array(img1)  # (H, W)

        if saved:
            try:
                drw = ImageDraw.Draw(img, 'RGBA')
                drw.polygon(vertices1, color)
            except TypeError:
                continue

        gt_mask = np.array(label)
        gt_mask[gt_mask == 255] = 1
        filt = np.sum(gt_mask)
        if filt <= 20*20:
            continue
        intersection = np.logical_and(gt_mask, pre_mask)
        union = np.logical_or(gt_mask, pre_mask)
        nu = np.sum(intersection)
        de = np.sum(union)
        # 求IoU
        iiou = nu / (de * 1.0) if de != 0 else 0.
        iou.append(iiou)

    iou.sort()
    iou.reverse()

    # print(iou)
    # print(len(iou))

    print(pre)
    print('IoU:', np.mean(np.array(iou)))

if __name__ == '__main__':


    z1 = [100, 200, 300, 400, 500, 600]
    z2 = [700, 800, 900, 1000, 1100, 1200, 1300]
    z3 = [1400, 1500, 1600, 1700, 1800]
    for z in z1:
        pre = str(z) +'.0.pth'
        get_score_ADE20K(pre)




