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
from models.model import PolygonModel
warnings.filterwarnings("ignore")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
selected_classes = ['person', 'car', 'truck', 'bicycle', 'motorcycle',
                    'rider', 'bus', 'train']
def get_score(net, dataset='test', maxnum=float('inf'), saved=False):

    save_result_dir = '/data/duye/cityscape/save_img/FPN_save/'
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)
    selected_classes = ['person', 'car', 'truck', 'bicycle', 'motorcycle',
                        'rider', 'bus', 'train']
    iou_score = {}
    nu = {}  # Intersection
    de = {}  # Union
    iou_true = {}
    for cls in selected_classes:
        iou_score[cls] = 0.0
        nu[cls] = 0.0
        de[cls] = 0.0
        iou_true[cls] = []

    count = 0
    print('origin count:', count)
    files_test = glob('img/{}/*/*.png'.format(dataset))  # 原始图像集,dataset指定训练集/测试集/验证集
    files_val = glob('img/val/*/*.png')
    print('test:', len(files_test))
    print('val :', len(files_val))
    files = files_test + files_val
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

                result = result_dict['pred_polys']
                # [0, 224] index 0: only one sample in mini-batch here
                pred_x = result_dict['final_pred_x'].cpu().numpy()[0]
                pred_y = result_dict['final_pred_y'].cpu().numpy()[0]
                pred_lengths = result_dict['lengths'].cpu().numpy()[0]
                pred_len = np.sum(pred_lengths) - 1  # sub EOS
                vertices1 = []
                vertices2 = []
                # Get the pred poly: [224,224] to origin image size [W, H]
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
                    nu[obj['label']] = 0
                    de[obj['label']] = 0
                    less2 += 1
                    iou_true[obj['label']].append(0.)
                    continue

                #  GT poly, in origin size: [W, H]
                for points in polygon:
                    vertex = (points[0], points[1])
                    vertices2.append(vertex)

                if saved:
                    #  GT draw
                    drw_gt = ImageDraw.Draw(img_gt, 'RGBA')
                    drw_gt.polygon(vertices2, color)

                # calculate IoU
                tmpp, nu_cur, de_cur = iou(vertices1, vertices2, H, W)
                nu[obj['label']] += nu_cur
                de[obj['label']] += de_cur
                iou_true[obj['label']].append(tmpp)

        count += 1
        if saved:
            print('saving test result image...')
            img.save(save_result_dir + str(idx) + '_pred_pp.png', 'PNG')
            img_gt.save(save_result_dir + str(idx) + '_gt_pp.png', 'PNG')
        if count >= maxnum:
            break
        print('count {} over'.format(count))

    # IoU
    for cls in iou_score:
        iou_score[cls] = nu[cls] * 1.0 / de[cls] if de[cls] != 0 else 0

    iou_mean_class = {}
    for cls in iou_true:
        iou_mean_class[cls] = np.mean(np.array(iou_true[cls]))
    iou_mean = 0.
    for cls in iou_mean_class:
        iou_mean += iou_mean_class[cls]
    iou_mean = iou_mean / 8.
    # return
    return iou_score, less2, nu, de, iou_mean_class, iou_mean


if __name__ == '__main__':
    print('start calculating...')

    # load_model = 'New_FPN2_DeltaModel_Epoch6-Step6000_ValIoU0.5855123247072112.pth'
    # load_model = 'New_FPN2_DeltaModel_Epoch5-Step4000_ValIoU0.6130178256915678.pth'
    # load_model = 'FPN_Epoch9-Step6000_ValIoU0.22982870834472033.pth'
    # load_model = 'Joint_FPN2_DeltaModel_Epoch7-Step3000_ValIoU0.6156470167768288.pth'   # 现在最好的是61.65
                                                                                    # 把val_every设置为500应该还可以提一下
    # load_model = 'Joint_FPN2_DeltaModel_Epoch9-Step6000_ValIoU0.6177457155898146.pth'
    load_model = 'ResNext_Plus_DeltaModel_Epoch7-Step3000_ValIoU0.619344842105648.pth'
    polynet_pretrained = '/data/duye/pretrained_models/' + load_model
    net = PolygonModel(predict_delta=True).to(device)
    net.load_state_dict(torch.load(polynet_pretrained))
    # Test mode
    net.eval()
    print('Pretrained model \'{}\' loaded!'.format(load_model))
    ious_test, less2_test, nu_test, de_test, iou_mean_class, iou_mean = get_score(net, maxnum=20, saved=True)
    print('Origin iou:', ious_test)
    print('True   iou:', iou_mean_class)
    print('Mean: ', iou_mean)
    """
    iou_mean_test = 0.
    for i in ious_test:
        iou_mean_test += ious_test[i]
    ious_val, less2_val, nu_val, de_val = get_score(net, dataset='val', saved=False)
    print('PRINT, VAL:', ious_val)
    nu_total = {}
    de_total = {}
    iou_total = {}
    for cls in selected_classes:  # init
        iou_total[cls] = 0.0
        nu_total[cls] = 0.0
        de_total[cls] = 0.0

    for cls in selected_classes:
        nu_total[cls] += nu_test[cls]
        nu_total[cls] += nu_val[cls]
        de_total[cls] += de_test[cls]
        de_total[cls] += de_val[cls]

    for cls in iou_total:
        iou_total[cls] = nu_total[cls] * 1.0 / de_total[cls] if de_total[cls] != 0 else 0


    # get mean
    meanTest = 0.
    meanVal = 0.
    meanTestVal = 0.
    for cls in iou_total:
        meanTest += ious_test[cls]
        meanVal += ious_val[cls]
        meanTestVal += iou_total[cls]

    print('Result:')
    print('IoU on TestSet:', ious_test)
    print('IoU on ValSet: ', ious_val)
    print('IoU on Test/ValSet:', iou_total)
    print('meanIoU on TestSet:', meanTest)
    print('meanIoU on ValSet: :', meanVal)
    print('meanIoU on TestValSet:', meanTestVal)
    """






