import scipy.io as scio
import glob
import numpy as np
import json
import os
from PIL import Image

def generate(mode='train'):
    dir = 'C:/Users/DELL/Desktop/毕设2020/数据集/航拍数据集Aerial Imagery/Rooftop/'
    dir = dir + mode
    save_dir = 'C:/Users/DELL/Desktop/毕设2020/数据集/航拍数据集Aerial Imagery/Rooftop/'
    save_dir = save_dir + mode + '_new/'
    save_img_dir = save_dir + 'img/'
    save_label_dir = save_dir + 'label/'

    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)
    if not os.path.exists(save_label_dir):
        os.makedirs(save_label_dir)

    files = glob.glob(dir+'/*.mat')
    cnt = 0
    print(len(files))
    for f in files:
        data = scio.loadmat(f)
        # 读取相应的Image文件
        img_f = f[:-3] + 'JPG'
        # 相应的img文件名
        img_f_name = img_f.split('\\')[-1]
        image = Image.open(img_f)
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
            extendrate = 0.1
            extendW = int(round(curW * extendrate))
            extendH = int(round(curH * extendrate))
            leftW = np.maximum(minW - extendW, 0)
            leftH = np.maximum(minH - extendH, 0)
            rightW = np.minimum(maxW + extendW, W)
            rightH = np.minimum(maxH + extendH, H)
            objectW = rightW - leftW
            objectH = rightH - leftH
            # 裁减，resize到224*224
            img_new = image.crop(box=(leftW, leftH, rightW, rightH)).resize((224, 224), Image.BILINEAR)

            # 保存img_new 到新的路径下
            img_new.save(os.path.join(save_img_dir, str(cnt) + '.JPG'))
            # json文件坐标对应缩放
            result_json = {}
            result_json['polygon'] = []
            result_json['origin_polygon'] = polygon.tolist()
            # scale保存, 裁减左上角点在原图中的坐标 scale=224/object_WH
            result_json['left_WH'] = [leftW, leftH]
            result_json['object_WH'] = [objectW, objectH]
            # 原图的H，W
            result_json['origion_WH'] = [W, H]
            result_json['origin_file'] = img_f_name
            for vertex in polygon:
                x = (vertex[0] - leftW) * (224.0 / objectW)
                y = (vertex[1] - leftH) * (224.0 / objectH)
                # 防溢出
                x = np.maximum(0, np.minimum(223, x))
                y = np.maximum(0, np.minimum(223, y))
                result_json['polygon'].append([x, y])
            with open(os.path.join(save_label_dir, str(cnt) + '.json'),
                      'w') as js:
                json.dump(result_json, js)
            cnt += 1


    meta = {}
    meta['total_count'] = cnt
    # 保存meta文件
    with open(os.path.join(save_dir, mode + '_meta.json'), 'w') as f:
        json.dump(meta, f)
    print('准备数据统计信息:')
    print('准备%s集, 总共样例数: %d' % (mode, cnt))


if __name__ == '__main__':
    generate('test')
    generate('train')