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
class RoofTop(Dataset):
    def __init__(self, path, seq_len, transform=None):
        super(RoofTop, self).__init__()
        self.path = path
        self.seq_len = seq_len
        self.transform = transform
        self.img_path = '/data/duye/Aerial_Imagery/Rooftop/'+self.path+'_new/img/'
        self.lbl_path = '/data/duye/Aerial_Imagery/Rooftop/' + self.path + '_new/label/'
        self.meta = '/data/duye/Aerial_Imagery/Rooftop/'+ self.path + '_new/' + \
                    self.path + '_meta.json'
        self.meta = json.load(open(self.meta))
        self.total_count = self.meta['total_count']

    def __len__(self):
        return self.total_count

    def __getitem__(self, index):
        try:
            img = Image.open(os.path.join(self.img_path, str(index) + '.JPG')).convert('RGB')
        except FileNotFoundError:
            return None
        assert not (img is None)

        W = img.width  # 224
        H = img.height  # 224

        # json file
        js = json.load(open(os.path.join(self.lbl_path, str(index) + '.json')))
        label = js['polygon']  # 多边形顶点
        left_WH = js['left_WH']  # 裁减图片左上角坐标在原图中的WH
        object_WH = js['object_WH']  # 裁减下来的图片(scale到224,224之前)的WH
        origion_WH = js['origion_WH']  # 原始图片的WH
        WH = {'left_WH': left_WH, 'object_WH': object_WH, 'origion_WH': origion_WH}
        origin_polygon = js['origin_polygon']
        point_num = len(label)
        polygon = np.array(label, dtype=np.float)  # (point_num, 2)
        polygon_GT = np.zeros((self.seq_len, 2), dtype=np.int32)

        # To (0,1)
        polygon = polygon / (W * 1.0)

        # To (0, g) 道格拉斯算法, 去重
        polygon = poly01_to_poly0g(polygon, 28)
        label_onehot = np.zeros([self.seq_len, 28 * 28 + 1])
        label_index = np.zeros([self.seq_len])
        point_scaled = []

        mask_final = np.zeros(self.seq_len, dtype=np.int)
        mask_delta = np.zeros(self.seq_len - 1, dtype=np.int)

        cnt = 0
        point_num = len(polygon)
        polygon01 = np.zeros([self.seq_len - 1, 2])
        tmp = np.array(poly0g_to_poly01(polygon, 28) * W, dtype=np.int)  # To(0, 224)

        ind = 0
        for vert in origin_polygon:
            x = int(vert[0])
            y = int(vert[1])
            polygon_GT[ind, 0] = x
            polygon_GT[ind, 1] = y
            ind += 1
            if ind >= self.seq_len:
                break


        if point_num <= 70:
            polygon01[:point_num] = tmp
        else:
            polygon01[:70] = tmp[:70]

        if point_num < self.seq_len:  # < 70
            for point in polygon:
                x = point[0]
                y = point[1]
                indexs = y * 28 + x
                label_index[cnt] = indexs
                label_onehot[cnt, indexs] = 1
                cnt += 1
                point_scaled.append([x, y])
            mask_final[:cnt + 1] = 1
            mask_delta[:cnt] = 1
            # end point
            label_index[cnt] = 28 * 28
            label_onehot[cnt, 28 * 28] = 1
            cnt += 1
            for ij in range(cnt, self.seq_len):
                label_index[ij] = 28 * 28
                label_onehot[ij, 28 * 28] = 1
                cnt += 1
        else:
            for iii in range(self.seq_len - 1):
                point = polygon[iii]  # 取点
                x = point[0]
                y = point[1]
                xx = x
                yy = y
                indexs = yy * 28 + xx
                label_index[cnt] = indexs
                label_onehot[cnt, indexs] = 1
                cnt += 1
                point_scaled.append([xx, yy])
            mask_final[:cnt + 1] = 1  # +1才会计算final最后EOS的损失
            mask_delta[:cnt] = 1
            # EOS
            label_index[self.seq_len - 1] = 28 * 28
            label_onehot[self.seq_len - 1, 28 * 28] = 1

        # ToTensor
        if self.transform:
            img = self.transform(img)

        point_scaled = np.array(point_scaled)
        # 边界，edge上的点为1，其余点为0
        edge_mask = np.zeros((28, 28), dtype=np.float)
        edge_mask = get_edge_mask(point_scaled, edge_mask)

        return img, \
               label_onehot[0], \
               label_onehot[:-2], \
               label_onehot[:-1], \
               label_index, \
               edge_mask, \
               mask_final, \
               mask_delta, \
               polygon_GT, \
               polygon01, WH
        # polygon01: 在224*224中的坐标, 而非0-1


def loadRooftop(path, len_s, batch_size, shuffle=True):
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
    Rooftop = RoofTop(path, len_s, transform)
    dataloader = DataLoader(Rooftop, batch_size=batch_size, shuffle=shuffle,
                            drop_last=False)
    print('DataLoader complete!', dataloader)
    return dataloader


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
            extendW = int(round(curW * extendrate))
            extendH = int(round(curH * extendrate))
            leftW = int(np.maximum(minW - extendW, 0))
            leftH = int(np.maximum(minH - extendH, 0))
            rightW = int(np.minimum(maxW + extendW, W))
            rightH = int(np.minimum(maxH + extendH, H))
            objectW = rightW - leftW
            objectH = rightH - leftH

            if objectW >=200 or objectH >= 200:
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
    load_model = 'ResNext_Plus_RL2_retain_Epoch1-Step4000_ValIoU0.6316584628283326.pth'
    polynet_pretrained = '/data/duye/pretrained_models/FPNRLtrain/' + load_model
    ious, nu2, de2, true_iou = getscore2(polynet_pretrained, dataset, saved=saved, maxnum=maxnum)
    print('Generalization on {}, pre-IoU={}'.format(dataset, nu2*1.0/de2))
    print('Generalization on {} gt-IoU={}'.format(dataset, true_iou))


# RoofTop:Generalization on Rooftop gt-IoU=0.689524889557



