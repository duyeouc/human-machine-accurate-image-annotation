# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import json
import numpy as np
from PIL import Image
from glob import glob

"""
    假设下载数据如下：
        data:
            -train:
                -city
                    xxx.png
                    xxx.json
            -val:
            -test:
"""


def prepareData(data):
    """

    :param data: train/test/val
    :return:
    """
    print('Prepare {} Data Begining...'.format(data))

    # paper中原则了8类特定object训练
    # trian中拿出最后两个城市(Weimar and Zurich)做val，原val做test
    # we use an alternative split, where the 500 origi- nal validation images form our test set.
    # We then split the original training set and select the images from two cities (Weimar and Zurich) as our validation,
    selected_classes = ['person', 'car', 'truck', 'bicycle', 'motorcycle', 'rider',
                        'bus', 'train']
    classes_num = {'person': 0, 'car': 0, 'truck': 0, 'bicycle': 0,
                   'motorcycle': 0, 'rider': 0, 'bus': 0, 'train': 0}
    # 配置文件，object总数量+classes
    meta = {}
    meta['total_count'] = 0  # 记录sample数量
    meta['classes'] = selected_classes
    # glob返回list，为文件名
    # 原始, img/train/city/*.png(json)
    train_img = glob('img/{}/*/*.png'.format(data))  # 记得事先把最后两个城市提出来
    # train_label = glob('label/{}/*/*.json'.format(data))

    # training_set = None
    # 计数，文件名
    train_count = 0

    """
        创建目录:
        - new_img:
            - train:
                *.png
            - test
            - val
        - new_label
    """
    trainimg_dir = '/data/duye/cityscape/new_img/{}/'.format(data)
    trainlabel_dir = '/data/duye/cityscape/new_label/{}/'.format(data)
    # meta_dir = '/data/duye/cityscape/{}/'.format(data)

    if not os.path.exists(trainimg_dir):
        os.makedirs(trainimg_dir)
    if not os.path.exists(trainlabel_dir):
        os.makedirs(trainlabel_dir)

    """
    aachen_000131_000019_leftImg8bit.png
    aachen_000173_000019_gtFine_polygons.json
    """

    # 遍历train_json
    for imgpath in train_img:
        # img/val/zurich/zurich_000064_000019_leftImg8bit.png
        # label/val/zurich/zurich_000064_000019_gtFine_polygons.json
        jsonpath = 'label' + imgpath[3:-15] + 'gtFine_polygons.json'
        jsonfile = json.load(open(jsonpath))
        # imgfile = jsonpath[:-20]+'leftImg8bit.png'
        # jsonfile = json.load(open(jsonpath))
        img = Image.open(imgpath)
        """
            josn文件格式:
            {   
                imgHeight:
                imgWidth:
                objects:[
                        {label:
                        polygon:},
                    ]
            }
            处理后的image: xx.png, 224*224
            对应处理后的json格式：
            xx.json:
            {   label: class
                polygon: [[x1,y1],[x2,y2],...]
            }
        """
        H = jsonfile['imgHeight']
        W = jsonfile['imgWidth']
        polygon_num = len(jsonfile['objects'])  # 点数
        # 遍历每一个polygon
        for poly in jsonfile['objects']:
            label = poly['label']  # 记录label
            if label in selected_classes:
                classes_num[label] += 1
                # 各个点坐标
                polygon = np.array(poly['polygon'])   #  ndarray (point_num, 2)
                vertex_num = len(polygon)
                # find min/max X,Y
                minW, minH = np.min(polygon, axis=0)
                maxW, maxH = np.max(polygon, axis=0)
                curW = maxW - minW
                curH = maxH - minH
                extendrate = 0.1
                extendW = int(round(curW * extendrate))
                extendH = int(round(curH * extendrate))
                leftW = np.maximum(minW - extendW, 0)
                leftH = np.maximum(minH - extendH, 0)
                rightW = np.minimum(maxW + extendW, W)
                rightH = np.minimum(maxH + extendH, H)
                # 当前object的BBoundBox大小，用作坐标缩放
                objectW = rightW - leftW
                objectH = rightH - leftH
                # 裁减，resize到224*224
                img_new = img.crop(box=(leftW, leftH, rightW, rightH)).resize((224, 224), Image.BILINEAR)

                # 保存img_new 到新的路径下
                img_new.save(os.path.join(trainimg_dir, str(train_count)+'.png'), 'PNG')
                # json文件坐标对应缩放
                result_json = {}
                result_json['label'] = label
                result_json['polygon'] = []
                # scale保存, 裁减左上角点在原图中的坐标 scale=224/object_WH
                result_json['left_WH'] = [leftW, leftH]
                result_json['object_WH'] = [objectW, objectH]
                # 原图的H，W
                result_json['origion_WH'] = [W, H]
                for vertex in polygon:
                    x = (vertex[0] - leftW) * (224.0 / objectW)
                    y = (vertex[1] - leftH) * (224.0 / objectH)
                    # 防溢出
                    x = np.maximum(0, np.minimum(223, x))
                    y = np.maximum(0, np.minimum(223, y))
                    result_json['polygon'].append([x, y])

                # 保存json文件
                with open(os.path.join(trainlabel_dir, str(train_count)+'.json'),
                          'w') as js:
                    json.dump(result_json, js)
                train_count += 1
                # 最后train_count+=1

    meta['total_count'] = train_count
    meta['select_classes'] = selected_classes
    meta['classes_num'] = classes_num
    # 保存meta文件
    with open(str(data)+'_meta.json', 'w') as f:
        json.dump(meta, f)
    print('准备数据统计信息:')
    print('准备%s集, 总共样例数: %d' % (data, train_count))
    for cl in selected_classes:
        print('   - 类别%s有%d个样本;' % (cl, classes_num[cl]))




if __name__ == '__main__':
    data1 = 'train'
    data2 = 'val'
    data3 = 'test'
    prepareData(data1)
    print('train over')
    prepareData(data2)
    print('val over')
    prepareData(data3)
    print('test over')