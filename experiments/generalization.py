# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import argparse
import glob
import scipy.io as scio
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import json
import os
import torchvision.transforms as transforms
from Utils import *
from RoofTop import loadRooftop
from models.model2 import PolygonModel

# 测试泛化性能: KITTI/Rooftop/ADE20K/医学数据集

devices = 'cuda' if torch.cuda.is_available() else 'cpu'

def getscore(model_path, dataset='Rooftop'):
    if dataset == 'Rooftop':
        dataloader = loadRooftop('test', 71, 16, shuffle=False)

    model = PolygonModel(predict_delta=True).to(devices)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
        print('Model loaded!')
    # set to eval
    model.eval()
    with torch.no_grad():
        val_IoU = []
        nu = 0.
        de = 0.
        less_than2 = 0
        for val_index, val_batch in enumerate(dataloader):
            img = torch.tensor(val_batch[0], dtype=torch.float).to(devices)
            bs = img.shape[0]
            WH = val_batch[-1]  # WH_dict
            polygon_GT = val_batch[-3]
            left_WH = WH['left_WH']
            origion_WH = WH['origion_WH']
            object_WH = WH['object_WH']
            val_target = val_batch[-2].numpy()  # (bs, 70, 2)
            val_mask_final = val_batch[7]  # (bs, 70)
            out_dict = model(img, mode='test')  # (N, seq_len) # test_time
            pred_x = out_dict['final_pred_x'].cpu().numpy()
            pred_y = out_dict['final_pred_y'].cpu().numpy()
            pred_len = out_dict['lengths']  # 预测的长度
            # 求IoU
            for ii in range(bs):
                scaleW = 224.0 / object_WH[0][ii]
                scaleH = 224.0 / object_WH[1][ii]
                leftW = left_WH[0][ii]
                leftH = left_WH[1][ii]
                vertices1 = []
                vertices2 = []

                # GT

                for vert in polygon_GT[ii]:
                    if vert[0] == 0 and vert[1] == 0:
                        break
                    v = (vert[0], vert[1])
                    vertices2.append(v)


                pred_len_b = pred_len[ii] - 1
                if pred_len_b < 2:
                    val_IoU.append(0.)
                    less_than2 += 1
                    continue
                for j in range(pred_len_b):
                    vertex = (pred_x[ii][j] / scaleW + leftW,
                              pred_y[ii][j] / scaleH + leftH)
                    vertices1.append(vertex)

                _, nu_cur, de_cur = iou(vertices1, vertices2, origion_WH[1][ii], origion_WH[0][ii])
                nu += nu_cur
                de += de_cur
                iou_cur = nu_cur * 1.0 / de_cur if de_cur != 0 else 0
                val_IoU.append(iou_cur)
    return np.mean(np.array(val_IoU)), nu, de


def getscore2(model_path, dataset='Rooftop', saved=False, maxnum=float('inf')):
    if dataset == 'Rooftop':
        dataloader = loadRooftop('test', 71, 16, shuffle=False)

    model = PolygonModel(predict_delta=True).to(devices)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
        print('Model loaded!')
    # set to eval
    model.eval()
    iou_score = 0.
    nu = 0.  # Intersection
    de = 0.  # Union
    iouss = []
    count = 0
    print('origin count:', count)
    #files_test = glob.glob('img/{}/*/*.png'.format(dataset))  # 原始图像集,dataset指定训练集/测试集/验证集
    # files_test = '/data/duye/Aerial_Imagery/Rooftop/test_new/img/*.JPG'
    # files_test = glob.glob(files_test)
    # print('test:', len(files_test))
    less2 = 0
    transform = transforms.Compose([transforms.ToTensor(), ])

    files = glob.glob('/data/duye/Aerial_Imagery/Rooftop/test/*.mat')  # 所有mat文件


    for idx, f in enumerate(files):
        data = scio.loadmat(f)
        # 读取相应的Image文件
        img_f = f[:-3] + 'JPG'
        # 相应的img文件名
        # img_f_name = img_f.split('\\')[-1]
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
            # find min/max X,Y
            minW, minH = np.min(polygon, axis=0)
            maxW, maxH = np.max(polygon, axis=0)
            curW = maxW - minW
            curH = maxH - minH
            extendrate = 0.10
            extendW = int(round(curW * extendrate))
            extendH = int(round(curH * extendrate))
            leftW = int(np.maximum(minW - extendW, 0))
            leftH = int(np.maximum(minH - extendH, 0))
            rightW = int(np.minimum(maxW + extendW, W))
            rightH = int(np.minimum(maxH + extendH, H))
            objectW = rightW - leftW
            objectH = rightH - leftH
            scaleH = 224.0 / objectH
            scaleW = 224.0 / objectW
            # 裁减，resize到224*224
            img_new = image.crop(box=(leftW, leftH, rightW, rightH)).resize((224, 224), Image.BILINEAR)

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

            result = result_dict['pred_polys']
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
            tt, nu_cur, de_cur = iou(vertices1, vertices2, W, H)
            nu += nu_cur
            de += de_cur
            iouss.append(tt)

        count += 1
        if saved:
            print('saving test result image...')
            save_result_dir = '/data/duye/save_dir/'
            image.save(save_result_dir + str(idx) + '_pred_pp.png', 'PNG')
            img_gt.save(save_result_dir + str(idx) + '_gt_pp.png', 'PNG')
        if count >= maxnum:
            break

    """
    for idx, file in enumerate(files_test):
        # json_file = 'label' + file[3:-15] + 'gtFine_polygons.json'
        # 加载json_file 得到origion polygon
        now = file.split('/')[-1][:-4]
        json_file = '/data/duye/Aerial_Imagery/Rooftop/test_new/label/' + now + '.json'
        json_object = json.load(open(json_file))
        polygon_gt = json_object['origin_polygon']
        WH = json_object['origion_WH']  # 原图的WH
        objectWH = json_object['object_WH']  # 该物体在原图中的WH
        W = WH[0]
        H = WH[1]

        scaleW = 224.0 / objectWH[0]
        scaleH = 224.0 / objectWH[1]

        leftWH = json_object['left_WH']

        try:
            img = Image.open(file).convert('RGB')
            img_pred = img
            img_gt = Image.open(file).convert('RGB')

        except FileNotFoundError:
            return None
        assert not (img is None)

        img = transform(img)  # To Tensor To(0, 1)
        img = img.unsqueeze(0)  # add batch dim
        color = [np.random.randint(0, 255) for _ in range(3)]
        color += [100]
        color = tuple(color)

        with torch.no_grad():
            pre_v2 = None
            pre_v1 = None
            result_dict = model(img.to(devices), pre_v2, pre_v1, mode='test', temperature=0.0)  # (bs, seq_len)

        result = result_dict['pred_polys']
        # [0, 224] index 0: only one sample in mini-batch here
        pred_x = result_dict['final_pred_x'].cpu().numpy()[0]
        pred_y = result_dict['final_pred_y'].cpu().numpy()[0]
        pred_lengths = result_dict['lengths'].cpu().numpy()[0]
        pred_len = np.sum(pred_lengths) - 1  # sub EOS
        vertices1 = []
        vertices2 = []
        # Get the pred poly
        for i in range(pred_len):
            vert = (pred_x[i] / scaleW + leftWH[0],
                    pred_y[i] / scaleH + leftWH[1])
            vertices1.append(vert)

        # pred-draw
        if saved:
            try:
                drw = ImageDraw.Draw(img_pred, 'RGBA')
                drw.polygon(vertices1, color)
            except TypeError:
                continue

        #  GT
        for points in polygon_gt:
            vertex = (points[0], points[1])
            vertices2.append(vertex)
        print('V1', vertices1)
        print('V2', vertices2)
        if saved:
            #  GT draw
            drw_gt = ImageDraw.Draw(img_gt, 'RGBA')
            drw_gt.polygon(vertices2, color)

        # calculate IoU
        _, nu_cur, de_cur = iou(vertices1, vertices2, H, W)
        nu += nu_cur
        de += de_cur

        count += 1
        if saved:
            print('saving test result image...')
            save_result_dir = '/data/duye/save_dir/'
            if not os.path.exists(save_result_dir):
                os.makedirs(save_result_dir)
            img_pred.save(save_result_dir + str(idx) + '_pred_pp.png', 'PNG')
            img_gt.save(save_result_dir + str(idx) + '_gt_pp.png', 'PNG')
        print('count {} over'.format(count))

        if count > maxnum:
            break
    """
    return iou_score, nu, de, np.mean(np.array(iouss))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('-d', '--dataset', type=str, default='Rooftop')
    parser.add_argument('-m', '--maxnum', type=float, default=float('inf'))
    parser.add_argument('-s', '--saved', type=bool, default=False)
    args = parser.parse_args()
    dataset = args.dataset
    maxnum = args.maxnum
    saved = args.saved
    load_model = 'RL_retain_Epoch1-Step1500_ValIoU0.6238014593919948.pth'
    polynet_pretrained = '/data/duye/pretrained_models/FPNRLtrain/' + load_model
    # _, nu, de = getscore(polynet_pretrained, dataset)
    _, nu2, de2, true_iou = getscore2(polynet_pretrained, dataset, saved=saved, maxnum=maxnum)
    # print('Generalization on {} IoU={}'.format(dataset, nu / de*1.0))
    print('Generalization on {} IoU={}'.format(dataset, nu2 / de2 * 1.0))
    print('True iou:', true_iou)

