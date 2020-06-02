# -*- coding: utf-8 -*-
from __future__ import print_function

import glob
import scipy.io as scio
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import argparse
import torch
import json
import os
import torchvision.transforms as transforms
from models.model import PolygonModel
from utils import *

devices = 'cuda' if torch.cuda.is_available() else 'cpu'
def getscore2(model_path, dataset='Rooftop', saved=False, maxnum=float('inf')):

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
    files = glob.glob('/data/duye/Aerial_Imagery/Rooftop/test/*.mat')  # 所有mat文件
    iouss = []
    for idx, f in enumerate(files):
        data = scio.loadmat(f)
        # 读取相应的Image文件
        img_f = f[:-3] + 'JPG'
        image = Image.open(img_f).convert('RGB')
        img_gt = Image.open(img_f).convert('RGB')
        I = np.array(image)
        W = image.width
        H = image.height
        lens = data['gt'][0].shape[0]
        for instance_id in range(lens):
            polygon = data['gt'][0][instance_id]
            polygon = np.array(polygon, dtype=np.float)
            vertex_num = len(polygon)
            if vertex_num < 3:
                continue
            # find min/max X,Y
            minW, minH = np.min(polygon, axis=0)
            maxW, maxH = np.max(polygon, axis=0)
            curW = maxW - minW
            curH = maxH - minH
            extendrate = 0.10
            extendW = curW * extendrate
            extendH = curH * extendrate
            leftW = int(np.maximum(minW - extendW, 0))
            leftH = int(np.maximum(minH - extendH, 0))
            rightW = int(np.minimum(maxW + extendW, W))
            rightH = int(np.minimum(maxH + extendH, H))
            objectW = rightW - leftW
            objectH = rightH - leftH

            # 过滤掉小的和过大的
            if objectW >= 150 or objectH >= 150:
                continue
            if objectW <= 20 or objectH <= 20:
                continue

            scaleH = 224.0 / float(objectH)
            scaleW = 224.0 / float(objectW)
            # 裁减，resize到224*224
            # img_new = image.crop(box=(leftW, leftH, rightW, rightH)).resize((224, 224), Image.BILINEAR)
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
                result_dict = model(I_obj_tensor, pre_v2, pre_v1,
                                    mode='test',
                                    temperature=0.0)  # (bs, seq_len)
            pred_x = result_dict['final_pred_x'].cpu().numpy()[0]
            pred_y = result_dict['final_pred_y'].cpu().numpy()[0]
            pred_lengths = result_dict['lengths'].cpu().numpy()[0]
            pred_len = np.sum(pred_lengths) - 1  # sub EOS
            vertices1 = []
            vertices2 = []

            # Get the pred poly
            for i in range(pred_len):
                vert = (pred_x[i] / scaleW + leftW,
                        pred_y[i] / scaleH + leftH)
                vertices1.append(vert)
            if len(vertices1) < 3:
                continue

            if saved:
                try:
                    drw = ImageDraw.Draw(image, 'RGBA')
                    drw.polygon(vertices1, color)
                except TypeError:
                    continue
            #  GT
            for points in polygon:
                vertex = (points[0], points[1])
                vertices2.append(vertex)

            if saved:
                #  GT draw
                drw_gt = ImageDraw.Draw(img_gt, 'RGBA')
                drw_gt.polygon(vertices2, color)

            # calculate IoU
            tmp, nu_cur, de_cur = iou(vertices1, vertices2, H, W)
            nu += nu_cur
            de += de_cur
            iouss.append(tmp)
        count += 1
        if saved:
            print('saving test result image...')
            save_result_dir = '/data/duye/save_dir/'
            image.save(save_result_dir + str(idx) + '_pred_rooftop.png', 'PNG')
            img_gt.save(save_result_dir + str(idx) + '_gt_rooftop.png', 'PNG')
        if count >= maxnum:
            break

    iouss.sort()
    iouss.reverse()
    true_iou = np.mean(np.array(iouss))
    print(iouss)
    return iou_score, nu, de, true_iou


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('-d', '--dataset', type=str, default='Rooftop')
    parser.add_argument('-m', '--maxnum', type=float, default=float('inf'))
    parser.add_argument('-s', '--saved', type=bool, default=False)
    args = parser.parse_args()
    dataset = args.dataset
    maxnum = args.maxnum
    saved = args.saved
    # load_model = ''

    z1 = [500, 550]
    for z in z1:
        print(z)
        polynet_pretrained = '/data/duye/pretrained_models/OnLineTraining_RoofTop/' + str(z) + '.pth'
        ious, nu2, de2, true_iou = getscore2(polynet_pretrained, dataset, saved=saved, maxnum=maxnum)
        print('  Generalization on {} gt-IoU={}'.format(dataset, true_iou))

