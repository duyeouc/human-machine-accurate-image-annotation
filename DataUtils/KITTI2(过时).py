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

devices = 'cuda' if torch.cuda.is_available() else 'cpu'

def getscore2(model_path, saved=False, maxnum=float('inf')):

    model = PolygonModel(predict_delta=True).to(devices)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
        print('Model loaded!')
    # set to eval
    model.eval()
    iou_score = 0.
    nu = 0.  # Intersection
    de = 0.  # Union
    count = 0
    files = glob.glob('/data/duye/KITTI/image/*')  # 所有img
    iouss = []
    trans = transforms.Compose([transforms.ToTensor(),])
    for idx, f in enumerate(files):
        # data = scio.loadmat(f)
        # 读取相应的Image文件
        # img_f = f[:-3] + 'JPG'
        image = Image.open(f).convert('RGB')  # png文件
        # img_gt = Image.open(f).convert('RGB')
        W = image.width
        H = image.height
        scaleH = 224.0 / float(H)
        scaleW = 224.0 / float(W)
        # 裁减，resize到224*224
        img_new = image.resize((224, 224), Image.BILINEAR)
        img_new = trans(img_new)
        img_new = img_new.unsqueeze(0)
        color = [np.random.randint(0, 255) for _ in range(3)]
        color += [100]
        color = tuple(color)

        with torch.no_grad():
            pre_v2 = None
            pre_v1 = None
            result_dict = model(img_new.to(devices), pre_v2, pre_v1, mode='test', temperature=0.0)  # (bs, seq_len)

            # [0, 224] index 0: only one sample in mini-batch here
            pred_x = result_dict['final_pred_x'].cpu().numpy()[0]
            pred_y = result_dict['final_pred_y'].cpu().numpy()[0]
            pred_lengths = result_dict['lengths'].cpu().numpy()[0]
            pred_len = np.sum(pred_lengths) - 1  # sub EOS
            vertices1 = []

            # Get the pred poly
            for i in range(pred_len):
                vert = (pred_x[i] / scaleW,
                        pred_y[i] / scaleH)
                vertices1.append(vert)

            if saved:
                try:
                    drw = ImageDraw.Draw(image, 'RGBA')
                    drw.polygon(vertices1, color)
                except TypeError:
                    continue
            #  GT
            gt_name = '/data/duye/KITTI/label/' + f.split('/')[-1][:-4] + '.png'
            # print(gt_name)
            # 读取mask
            gt_mask = Image.open(gt_name)
            gt_mask = np.array(gt_mask)  # (H, W)

            gt_mask[gt_mask > 0] = 255
            gt_mask[gt_mask == 255] = 1

            if saved:
                pass
                #  GT draw
                # drw_gt = ImageDraw.Draw(img_gt, 'RGBA')
                # drw_gt.polygon(vertices2, color)

            # calculate IoU
            img1 = Image.new('L', (W, H), 0)
            ImageDraw.Draw(img1).polygon(vertices1, outline=1, fill=1)
            pre_mask = np.array(img1)  # (H, W)
            # get iou
            intersection = np.logical_and(gt_mask, pre_mask)
            union = np.logical_or(gt_mask, pre_mask)
            nu = np.sum(intersection)
            de = np.sum(union)
            iiou = nu / (de * 1.0) if de != 0 else 0.
            iouss.append(iiou)
        count += 1
        print(count)
        if saved:
            print('saving test result image...')
            save_result_dir = '/data/duye/save_dir/'
            image.save(save_result_dir + str(idx) + '_pred_rooftop.png', 'PNG')
            # img_gt.save(save_result_dir + str(idx) + '_gt_rooftop.png', 'PNG')
        if count >= maxnum:
            break

    iouss.sort()
    iouss.reverse()
    print(iouss)
    true_iou = np.mean(np.array(iouss[:741]))

    return iou_score, nu, de, true_iou


# TODO： 从原始图片中加载，根据Bounding Box 扩展10%，之后再计算在原图中的IoU，用提供的950张图片(或者所有一共1229张)，取前741个结果
# TODO：直接1229张可行，950不行。或者尝试一下过滤一下大小，把小的过滤掉？
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('-m', '--maxnum', type=float, default=float('inf'))
    parser.add_argument('-s', '--saved', type=bool, default=False)
    args = parser.parse_args()
    dataset = 'KITTI'
    maxnum = args.maxnum
    saved = args.saved
    # load_model = 'RL_retain_Epoch1-Step1500_ValIoU0.6238014593919948.pth'
    load_model = 'ResNext_Plus_RL2_retain_Epoch1-Step4000_ValIoU0.6316584628283326.pth'
    polynet_pretrained = '/data/duye/pretrained_models/FPNRLtrain/' + load_model
    ious, nu2, de2, true_iou = getscore2(polynet_pretrained, saved=saved, maxnum=maxnum)
    # print('Generalization on {}, pre-IoU={}'.format(dataset, nu2*1.0/de2))
    print('Generalization on {} gt-IoU={}'.format(dataset, true_iou))

# Generalization on KITTI取741个object： IoU=0.760618645492

