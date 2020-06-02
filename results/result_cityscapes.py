# -*- coding: utf-8 -*-
from __future__ import print_function
import os
from glob import glob
import json
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import warnings
from models.model import PolygonModel
from utils import iou, poly01_to_poly0g
warnings.filterwarnings("ignore")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
selected_classes = ['person', 'bus', 'truck', 'bicycle', 'motorcycle',
                    'rider', 'car', 'train']

def get_score(net, dataset='test', maxnum=float('inf'), saved=False):

    save_result_dir = '/data/duye/cityscape/save_img/cityscapes_results/'
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)
    selected_classes = ['person', 'car', 'truck', 'bicycle', 'motorcycle',
                        'rider', 'bus', 'train']

    ious = {}
    iou_inter = {}
    count = 0
    print('origin count:', count)
    files_test = glob('/data/duye/cityscape/new_img/train/*.png')
    files = files_test
    print('All: ', len(files))

    for idx, file in enumerate(files):
        ddx = file.split('/')[-1][:-4]
        json_file = '/data/duye/cityscape/new_label/train/' + ddx + '.json'
        json_object = json.load(open(json_file))
        img = Image.open(file).convert('RGB')  # PIL
        img_inter = Image.open(file).convert('RGB')
        I = np.array(img)
        img_gt = Image.open(file).convert('RGB')
        gt_len = []
        polygon = np.array(json_object['polygon'])  # 在原图片中的坐标
        # img_new = img.crop(box=(leftW, leftH, rightW, rightH)).resize((224, 224), Image.BILINEAR)
        # To PIL image
        I_obj_img = Image.fromarray(I)
        I_obj_new = np.array(I_obj_img)  # (H, W, C)
        I_obj_new = I_obj_new.transpose(2, 0, 1)  # (C, H, W)
        # 归一化
        I_obj_new = I_obj_new / 255.0
        I_obj_tensor = torch.from_numpy(I_obj_new)  # (C, H, W)
        I_obj_tensor = torch.tensor(I_obj_tensor.unsqueeze(0), dtype=torch.float).cuda()
        color = [np.random.randint(0, 255) for _ in range(3)]
        color += [120]
        color = tuple(color)
        with torch.no_grad():
            pre_v2 = None
            pre_v1 = None
            result_dict = net(I_obj_tensor,
                              pre_v2,
                              pre_v1,
                              mode='test',
                              temperature=0.0)  # (bs, seq_len)
        pred_x = result_dict['final_pred_x'].cpu().numpy()[0]
        pred_y = result_dict['final_pred_y'].cpu().numpy()[0]
        pred_lengths = result_dict['lengths'].cpu().numpy()[0]
        pred_len = np.sum(pred_lengths) - 1  # sub EOS
        vertices1 = []
        vertices2 = []
        # Get the pred poly: 还原到原图中
        for i in range(pred_len):
            vert = (pred_x[i],
                    pred_y[i])
            vertices1.append(vert)
        # pred-draw
        if saved:
            try:
                drw = ImageDraw.Draw(img, 'RGBA')
                drw.polygon(vertices1, color, outline='darkorange')
                drw.point(vertices1, fill='red')

            except TypeError:
                continue
        if len(vertices1) < 2:
            ious[ddx] = 0.
            continue
        #  GT
        for points in polygon:
            vertex = (points[0], points[1])
            vertices2.append(vertex)
            gt_len.append(len(vertices2))
        if saved:
            #  GT draw
            drw_gt = ImageDraw.Draw(img_gt, 'RGBA')
            drw_gt.polygon(vertices2, color, outline='white')
            drw_gt.point(vertices2, fill='red')

        # calculate IoU
        tmp, nu_cur, de_cur = iou(vertices1, vertices2, 224, 224)
        ious[ddx] = tmp
        count += 1

        # get gt in ~ [224,224]
        gt_224 = []
        for vertex in polygon:
            x = vertex[0]
            y = vertex[1]
            gt_224.append([x, y])
        # To (28, 28)
        gt_224 = np.array(gt_224)
        gt_28 = gt_224 / (224 * 1.0)
        # To (0, g), int值,即在28*28中的坐标值 道格拉斯算法多边形曲线拟合，这里有一个去除重点的过程
        gt_28 = poly01_to_poly0g(gt_28, 28)
        # To indexes
        seq_len = 71
        gt_index = np.zeros([seq_len])
        point_num = len(gt_28)
        cnts = 0
        if point_num < seq_len:  # < 70
            for point in gt_28:
                x = point[0]
                y = point[1]
                indexs = y * 28 + x
                gt_index[cnts] = indexs
                cnts += 1
            # end point
            gt_index[cnts] = 28 * 28
            cnts += 1
            for ij in range(cnts, seq_len):
                gt_index[ij] = 28 * 28
                cnts += 1
        else:
            # 点数过多的话只取前70个点是不对的, 这里应该考虑一下如何选取点
            for iii in range(seq_len - 1):
                point = polygon[iii]  # 取点
                x = point[0]
                y = point[1]
                indexs = y * 28 + x
                gt_index[cnts] = indexs
                cnts += 1
            # EOS
            gt_index[seq_len - 1] = 28 * 28

        gt_index = torch.tensor(gt_index, dtype=torch.float64).unsqueeze(0).to(device)

        # 对应的标注模式
        with torch.no_grad():
            result_inter = net(I_obj_tensor, pre_v2, pre_v1,
                              mode='interaction_loop', temperature=0.0,
                              gt_28=gt_index)  # (bs, seq_len)

        pred_x = result_inter['final_pred_x'].cpu().numpy()[0]
        pred_y = result_inter['final_pred_y'].cpu().numpy()[0]
        pred_lengths = result_inter['lengths'].cpu().numpy()[0]
        pred_len = np.sum(pred_lengths) - 1  # sub EOS
        vertices_inter = []
        for i in range(pred_len):
            vert = (pred_x[i],
                    pred_y[i])
            vertices_inter.append(vert)
        # pred-draw
        if saved:
            try:
                drw_inter = ImageDraw.Draw(img_inter, 'RGBA')
                drw_inter.polygon(vertices_inter, color, outline='darkorange')
                drw_inter.point(vertices1, fill='red')
            except TypeError:
                continue
        # 求IoU
        tmp2, nu_cu2r, de_cu2r = iou(vertices_inter, vertices2, 224, 224)
        iou_inter[ddx] = tmp2

        if saved:
            print('saving test result image...')
            img.save(save_result_dir + str(idx) + '_auto.png', 'PNG')
            img_gt.save(save_result_dir + str(idx) + '_gt.png', 'PNG')
            img_inter.save(save_result_dir + str(idx) + '_inter.png', 'PNG')
        if count >= maxnum:
            break
    print(ious)
    print(iou_inter)
    # 将IoUs保存
    with open(save_result_dir + 'result_auto.json','w') as file_obj:
        json.dump(ious, file_obj)
    with open(save_result_dir + 'result_inter.json','w') as file_obj:
        json.dump(iou_inter, file_obj)


if __name__ == '__main__':
    print('start calculating...')

    # load_model = 'ResNext_Plus_RL2_retain_Epoch1-Step500_ValIoU0.6251150881397767.pth'
    load_model = 'ResNext_Plus_RL2_retain_Epoch1-Step4000_ValIoU0.6316584628283326.pth'
    polynet_pretrained = '/data/duye/pretrained_models/FPNRLtrain/' + load_model
    net = PolygonModel(predict_delta=True, loop_T=1).to(device)
    net.load_state_dict(torch.load(polynet_pretrained))
    net.eval()
    print('Pretrained model \'{}\' loaded!'.format(load_model))
    get_score(net, saved=True, maxnum=100)