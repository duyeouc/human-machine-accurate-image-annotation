# -*- coding: utf-8 -*-
from __future__ import print_function
import os
from glob import glob
import json
from PIL import Image, ImageDraw
import numpy as np
import torch
from models.model import PolygonModel
import warnings
from utils import iou
import torchvision.transforms as transforms
warnings.filterwarnings("ignore")

"""
    测试，注意修改model来源
"""
def test(net, dataset, maxnum=float('inf'), saved=False):
    """

    :param net: 要测试的模型
    :param dataset: train/test/val
    :param num: 最多测试多少张图片
    :return:
    """

    # 测试结果图像保存路径
    save_result_dir = '/data/duye/cityscape/save_img/polygonrnn_pp_save/'
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)
    selected_classes = ['person', 'car', 'truck', 'bicycle', 'motorcycle',
                        'rider', 'bus', 'train']

    iou_score = {}
    iou_class = {}
    nu = {}  # Intersection
    de = {}  # Union
    for cls in selected_classes:
        iou_score[cls] = 0.0
        nu[cls] = 0.0
        de[cls] = 0.0
        iou_class[cls] = []

    count = 0
    print('origin count:', count)
    files_test = glob('img/{}/*/*.png'.format(dataset))  # 原始图像集,dataset指定训练集/测试集/验证集
    files_val = glob('img/val/*/*.png')
    print(dataset, len(files_test))
    print('val :', len(files_val))
    files = files_test + files_val
    print('All: ', len(files))
    less2 = 0

    # trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), ])

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
                polygon = np.array(obj['polygon'])
                # find min/max X,Y
                minW, minH = np.min(polygon, axis=0)
                maxW, maxH = np.max(polygon, axis=0)
                curW = maxW - minW
                curH = maxH - minH
                # Extend 10%
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

                # # I array (H, W, C)
                I_obj = I[leftH:rightH, leftW:rightW, :]
                # # To PIL image
                I_obj_img = Image.fromarray(I_obj)
                # # resize
                I_obj_img = I_obj_img.resize((224, 224), Image.BILINEAR)
                I_obj_new = np.array(I_obj_img)  # (H, W, C)
                I_obj_new = I_obj_new.transpose(2, 0, 1)  # (C, H, W)

                # div 255 归一化
                I_obj_new = I_obj_new / 255.0

                I_obj_tensor = torch.from_numpy(I_obj_new)  # (C, H, W)
                I_obj_tensor = torch.tensor(I_obj_tensor.unsqueeze(0), dtype=torch.float).cuda()

                # # I_obj_tensor = trans(img_new)
                # # print(I_obj_tensor.shape)
                # I_obj_tensor = torch.tensor(I_obj_tensor, dtype=torch.float).cuda()

                color = [np.random.randint(0, 255) for _ in range(3)]
                color += [100]
                color = tuple(color)
                # forward pass, tes

                with torch.no_grad():
                    pre_v2 = None
                    pre_v1 = None
                    # train_rl的时候, temperature=0.1
                    result_dict = net(I_obj_tensor, pre_v2, pre_v1, mode='test', temperature=0.0)  # (bs, seq_len)

                result = result_dict['pred_polys']
                label_p = result.cpu().numpy()[0]

                vertices1 = []
                vertices2 = []
                # print(label_p)
                for label in label_p:
                    # 结束标记
                    if label == 28 * 28:
                        break
                    vertex0 = (
                        ((label % 28) * 8.0 + 4) / scaleW + leftW, (
                                (int(label / 28)) * 8.0 + 4) / scaleH + leftH)
                    vertices1.append(vertex0)

                # 画多边形，把这个预测的多边形画在原Image上
                # pred-draw
                if saved:
                    try:
                        drw = ImageDraw.Draw(img, 'RGBA')
                        drw.polygon(vertices1, color)
                    except TypeError:
                        continue
                # print(len(vertices1))
                if len(vertices1) < 2:
                    print(label_p)
                    nu[obj['label']] += 0
                    de[obj['label']] += 0  # 这个是不对的~
                    less2 += 1
                    iou_class[obj['label']].append(0.)
                    # print('less2: ', less2)
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
                nu[obj['label']] += nu_cur
                de[obj['label']] += de_cur
                iou_class[obj['label']].append(tmp)  # 分类直接记录iou

        count += 1
        # 保存画了之后的img，这个太耗费空间，先不保存

        if saved:
            print('saving test result image...')
            img.save(save_result_dir + str(idx) + '_pred_pp_lstm.png', 'PNG')
            img_gt.save(save_result_dir + str(idx) + '_gt_pp_lstm.png', 'PNG')
        if count >= maxnum:
            break
        print('count ', count)
        # print(nu)
        # print(de)


    # IoU
    for cls in iou_score:
        iou_score[cls] = nu[cls] * 1.0 / de[cls] if de[cls] != 0 else 0
    iou_mean_class = {}
    for cls in iou_class:
        iou_mean_class[cls] = np.mean(np.array(iou_class[cls]))
    iou_mean = 0.
    for cls in iou_mean_class:
        iou_mean += iou_mean_class[cls]
    iou_mean = iou_mean / 8.


    # return
    return iou_score, less2, nu, de, iou_mean_class, iou_mean


if __name__ == '__main__':
    print('start calculating...')
    # 目前FCN2+ LSTM 最好的结果
    polynet_pretrained = '/data/duye/pretrained_models/' \
                        'ResNext50_FPN_LSTM_Epoch7-Step6000_ValIoU0.6115920530564407.pth'
                        # 'ResNext50_FPN_LSTM_Epoch9-Step6000_ValIoU0.6136271761956149.pth'
                        #'ResNext50_FPN_LSTM_Epoch6-Step3000_ValIoU0.6060563038942592.pth'
                         # 'New_Res50_FPN_LSTM_Epoch10-Step6000_ValIoU0.6071237347797661.pth'
                         # 'FPN_LSTM_Epoch10-Step3000_ValIoU0.5368686700291017.pth'

    net = PolygonModel(predict_delta=False).cuda()
    net.load_state_dict(torch.load(polynet_pretrained))
    net.eval()
    print('pre_trained model loaded!')
    dataset = 'test'  # train/test/val
    ious, less2, nu1, de1, iou_class, iou_mean = test(net, dataset, maxnum=20, saved=True)
    print('Result:')

    print('Pre IoU:', ious)
    print('True IoU:', iou_class)
    print('True Mean:', iou_mean)


