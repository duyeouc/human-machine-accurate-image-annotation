# -*- coding: utf-8 -*-
from __future__ import print_function
import os
from glob import glob
import json
from PIL import Image, ImageDraw
import numpy as np
import torch
import warnings
from utils import iou
import argparse
from models.model import PolygonModel
warnings.filterwarnings("ignore")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
selected_classes = ['person', 'bus', 'truck', 'bicycle', 'motorcycle',
                    'rider', 'car', 'train']

def get_score(net, dataset='test', maxnum=float('inf'), saved=False):

    save_result_dir = '/data/duye/cityscape/save_img/FPN_save/'
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)
    selected_classes = ['person', 'car', 'truck', 'bicycle', 'motorcycle',
                        'rider', 'bus', 'train']
    iou_score = {}
    iouss = {}
    nu = {}  # Intersection
    de = {}  # Union
    for cls in selected_classes:
        iou_score[cls] = 0.0
        nu[cls] = 0.0
        de[cls] = 0.0
        iouss[cls] = []

    count = 0
    print('origin count:', count)
    files_test = glob('img/{}/*/*.png'.format(dataset))  # 原始图像集,dataset指定训练集/测试集/验证集
    print('test:', len(files_test))
    files = files_test
    print('All: ', len(files))
    less2 = 0

    for idx, file in enumerate(files):
        json_file = 'label' + file[3:-15] + 'gtFine_polygons.json'
        json_object = json.load(open(json_file))
        H = json_object['imgHeight']
        W = json_object['imgWidth']
        objects = json_object['objects']
        img = Image.open(file).convert('RGB')  # PIL
        I = np.array(img)
        img_gt = Image.open(file).convert('RGB')

        for obj in objects:
            if obj['label'] in selected_classes:
                polygon = np.array(obj['polygon'])  # 在原图片中的坐标
                # find min/max X,Y
                minW, minH = np.min(polygon, axis=0)
                maxW, maxH = np.max(polygon, axis=0)
                curW = maxW - minW
                curH = maxH - minH
                # Extend 10% ~ 测试鲁邦性，可以把这个扩展5%-15%
                extendW = int(round(curW * 0.1))
                extendH = int(round(curH * 0.1))
                leftW = np.maximum(minW - extendW, 0)  # minrow, mincol, maxrow, maxcol
                leftH = np.maximum(minH - extendH, 0)
                rightW = np.minimum(maxW + extendW, W)
                rightH = np.minimum(maxH + extendH, H)
                objectW = rightW - leftW
                objectH = rightH - leftH
                scaleH = 224.0 / objectH
                scaleW = 224.0 / objectW
                img_new = img.crop(box=(leftW, leftH, rightW, rightH)).resize((224, 224), Image.BILINEAR)
                I_obj = I[leftH:rightH, leftW:rightW, :]
                # To PIL image
                I_obj_img = Image.fromarray(I_obj)
                # resize
                I_obj_img = I_obj_img.resize((224, 224), Image.BILINEAR)
                I_obj_new = np.array(I_obj_img)  # (H, W, C)
                I_obj_new = I_obj_new.transpose(2, 0, 1)  # (C, H, W)
                # 归一化
                I_obj_new = I_obj_new / 255.0
                I_obj_tensor = torch.from_numpy(I_obj_new)  # (C, H, W)
                I_obj_tensor = torch.tensor(I_obj_tensor.unsqueeze(0), dtype=torch.float).cuda()
                color = [np.random.randint(0, 255) for _ in range(3)]
                color += [100]
                color = tuple(color)

                with torch.no_grad():
                    pre_v2 = None
                    pre_v1 = None
                    result_dict = net(I_obj_tensor, pre_v2, pre_v1, mode='test', temperature=0.0)  # (bs, seq_len)

                # result = result_dict['pred_polys']
                # [0, 224] index 0: only one sample in mini-batch here
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

                # pred-draw
                if saved:
                    try:
                        drw = ImageDraw.Draw(img, 'RGBA')
                        drw.polygon(vertices1, color)
                    except TypeError:
                        continue
                if len(vertices1) < 2:
                    nu[obj['label']] += 0
                    de[obj['label']] += 0
                    less2 += 1
                    iouss[obj['label']].append(0)
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
                iouss[obj['label']].append(tmp)
                nu[obj['label']] += nu_cur
                de[obj['label']] += de_cur

        count += 1
        if saved:
            print('saving test result image...')
            img.save(save_result_dir + str(idx) + '_pred_pp.png', 'PNG')
            img_gt.save(save_result_dir + str(idx) + '_gt_pp.png', 'PNG')
        if count >= maxnum:
            break

        # print('count {} over'.format(count))
    # IoU
    for cls in iou_score:
        iou_score[cls] = nu[cls] * 1.0 / de[cls] if de[cls] != 0 else 0

    iosss = 0.
    iou_mean_class = {}
    for cls in iouss:
        iou_mean_class[cls] = np.mean(np.array(iouss[cls]))
        iosss += iou_mean_class[cls]
    iosss = iosss / 8.
    return iou_score, less2, nu, de, iou_mean_class, iosss


if __name__ == '__main__':
    print('start calculating...')
    z1 = range(200, 250, 10)
    net = PolygonModel(predict_delta=True).to(device)
    for t in z1:
        load_model = str(t) + '.pth'
        polynet_pretrained = '/data/duye/pretrained_models/OnLineTraining/' + load_model
        net.load_state_dict(torch.load(polynet_pretrained))
        net.eval()
        ious_test, less2_test, nu_test, de_test, iou_mean_class, iou_mean = get_score(net, saved=False)
        print('Online Training after {}: {}'.format(t, iou_mean))
        print(iou_mean_class)
        print()
