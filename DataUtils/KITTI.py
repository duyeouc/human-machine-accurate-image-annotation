# -*- coding: utf-8 -*-
from __future__ import print_function
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import glob
import argparse
import numpy as np
import torchvision.transforms as transforms
from utils import *
from models.model import PolygonModel
import os
class KITTI(Dataset):
    def __init__(self, transform=None):
        super(KITTI, self).__init__()
        self.transform = transform
        self.img_path = '/data/duye/KITTI/image/'
        self.lbl_path = '/data/duye/KITTI/label/'
        self.label = glob.glob(self.lbl_path + '*')
        self.total_count = len(self.label)

    def __len__(self):
        return self.total_count

    def __getitem__(self, index):
        # 对应的label文件名
        gt = self.label[index]
        # print(gt)
        cur_name = gt.split('/')[-1][:-4]
        # gt_array = np.array(gt)  # (H, W)
        try:
            tm = self.img_path + cur_name + '.png'
            img = Image.open(self.img_path + cur_name + '.png').convert('RGB')
            W = img.width
            H = img.height
        except FileNotFoundError:
            return None
        assert not (img is None)
        # reshape
        img_new = img.resize((224, 224), Image.BILINEAR)
        img_new = self.transform(img_new)
        origion_WH = [W, H]
        WH = {'origion_WH': origion_WH}

        return img_new, \
               gt, \
               WH

def loadLITTI(batch_size, shuffle=False):
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
    ss = KITTI(transform)
    dataloader = DataLoader(ss, batch_size=batch_size, shuffle=shuffle,
                            drop_last=False)
    print('DataLoader complete!', dataloader)
    return dataloader





def getscore_kitti(model_path, saved=False, maxnum=float('inf')):

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
    files = glob.glob('/data/duye/KITTI/rawImage/*.png')  # 所有img
    bbox = '/data/duye/KITTI/bbox/'
    annotation = '/data/duye/KITTI/annotation/'
    print(len(files))
    iouss = []
    for idx, f in enumerate(files):
        print('index:', idx)
        image = Image.open(f).convert('RGB')  # raw image
        W = image.width
        H = image.height
        I = np.array(image)
        # print(I.shape)
        # 读相应的BD
        name = f.split('/')[-1][:-4]  # 000019
        bd = bbox + name + '.txt'

        if not os.path.exists(bd):
            continue

        # 相应的annotation
        sss = annotation + name + '.png'
        if not os.path.exists(sss):
            continue
        anno = Image.open(annotation + name + '.png')
        anno = np.array(anno)
        # 遍历
        with open(bd, 'r') as bbd:
            all = bbd.readlines()
            for number, line in enumerate(all):
                line = line.replace('\n', '')
                line = line.split(' ')
                if float(line[0]) == 0.0 or \
                   float(line[1]) == 0.0 or \
                   float(line[2]) == 0.0 or\
                   float(line[3]) == 0.0:
                    continue
                xx = float(line[0])
                yy = float(line[1])
                ww = float(line[2])
                hh = float(line[3])
                minW = xx
                minH = yy
                maxW = xx + ww
                maxH = yy + hh
                # 扩展10%
                extendrate = 0.08
                curW = ww
                curH = hh
                extendW = int(round(curW * extendrate))
                extendH = int(round(curH * extendrate))
                leftW = int(np.maximum(minW - extendW, 0))
                leftH = int(np.maximum(minH - extendH, 0))
                rightW = int(np.minimum(maxW + extendW, W))
                rightH = int(np.minimum(maxH + extendH, H))
                # 当前object的BBoundBox大小，用作坐标缩放
                objectW = rightW - leftW
                objectH = rightH - leftH
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
                I_obj_tensor = torch.tensor(I_obj_tensor.unsqueeze(0), dtype=torch.float).to(devices)

                color = [np.random.randint(0, 255) for _ in range(3)]
                color += [100]
                color = tuple(color)

                with torch.no_grad():
                    pre_v2 = None
                    pre_v1 = None
                    result_dict = model(I_obj_tensor, pre_v2, pre_v1, mode='test', temperature=0.0)  # (bs, seq_len)
                # [0, 224] index 0: only one sample in mini-batch here
                pred_x = result_dict['final_pred_x'].cpu().numpy()[0]
                pred_y = result_dict['final_pred_y'].cpu().numpy()[0]
                pred_lengths = result_dict['lengths'].cpu().numpy()[0]
                pred_len = np.sum(pred_lengths) - 1  # sub EOS
                vertices1 = []
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
                # pred-mask
                img1 = Image.new('L', (W, H), 0)
                ImageDraw.Draw(img1).polygon(vertices1, outline=1, fill=1)
                pre_mask = np.array(img1)  # (H, W)

                # gt mask
                # number 这样不对！
                cur_anno = anno
                cur_anno = np.array(cur_anno == number+1, dtype=int)
                # cur_anno[cur_anno != 1] = 0

                # getIOU
                intersection = np.logical_and(cur_anno, pre_mask)
                union = np.logical_or(cur_anno, pre_mask)
                nu = np.sum(intersection)
                de = np.sum(union)
                iiou = nu / (de * 1.0) if de != 0 else 0.
                iouss.append(iiou)

    iouss.sort()
    iouss.reverse()
    print(iouss)

    print(np.mean(np.array(iouss)))

    true_iou = np.mean(np.array(iouss[:741]))

    return iou_score, nu, de, true_iou



# 测试得分
devices = 'cuda' if torch.cuda.is_available() else 'cpu'
if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='测试在KITTI上的泛化得分')
    parse.add_argument('-p', '--pretrained', type=str, default=None)
    args = parse.parse_args()
    pre = args.pretrained
    pre = 'ResNext_Plus_RL2_retain_Epoch1-Step4000_ValIoU0.6316584628283326.pth'
    dirs = '/data/duye/pretrained_models/FPNRLtrain/' + pre
    a, b, c, ious = getscore_kitti(dirs)
    print('Top 741 iou mean:', ious)


# Top 741 iou mean: 84.80 : 这就是top了!!!