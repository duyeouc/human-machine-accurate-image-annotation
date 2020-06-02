# -*- coding: utf-8 -*-
from __future__ import print_function
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import glob
import argparse
import numpy as np
import torchvision.transforms as transforms
from utils import *
# import tqdm
from models.model import PolygonModel

class ssTEM(Dataset):
    def __init__(self, transform=None):
        super(ssTEM, self).__init__()
        self.transform = transform
        self.img_path = '/data/duye/ssTEM/raw/'
        self.lbl_path1 = '/data/duye/ssTEM/label/label1/*'
        self.lbl_path2 = '/data/duye/ssTEM/label/label2/*'
        self.label = glob.glob(self.lbl_path1) + glob.glob(self.lbl_path2)
        self.total_count = len(self.label)

    def __len__(self):
        return self.total_count

    def __getitem__(self, index):

        # 对应的label文件
        gt = self.label[index]
        cur_name = gt.split('/')[-1][:2]  # 00-19
        gt = Image.open(gt)
        W = gt.width
        H = gt.height
        gt_array = np.array(gt)  # (H, W)
        try:
            img = Image.open(self.img_path + cur_name + '.tif').convert('RGB')
        except FileNotFoundError:
            return None
        assert not (img is None)
        # print(gt_array.shape)
        Hs, Ws = np.where(gt_array == np.max(gt_array))
        minH = np.min(Hs)
        maxH = np.max(Hs)
        minW = np.min(Ws)
        maxW = np.max(Ws)
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
        img_new = img.crop(box=(leftW, leftH, rightW, rightH)).resize((224, 224), Image.BILINEAR)
        # img_new.save('/data/duye/save_dir/test.png')
        img_new = self.transform(img_new)
        left_WH = [leftW, leftH]
        object_WH = [objectW, objectH]
        origion_WH = [W, H]
        # 记录Object WH / WH /left WH
        WH = {'left_WH': left_WH, 'object_WH': object_WH, 'origion_WH': origion_WH}

        gt_array[gt_array == 255] = 1
        return img_new, \
               gt_array, \
               WH


def loadssTEM(batch_size, shuffle=False):
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
    ss = ssTEM(transform)
    dataloader = DataLoader(ss, batch_size=batch_size, shuffle=shuffle,
                            drop_last=False)
    print('DataLoader complete!', dataloader)
    return dataloader

# 测试得分
devices = 'cuda' if torch.cuda.is_available() else 'cpu'
if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='测试在ssTEM上的泛化得分')
    parse.add_argument('-p', '--pretrained', type=str, default=None)
    args = parse.parse_args()
    pre = args.pretrained
    model = PolygonModel(predict_delta=True).to(devices)
    pre = 'ResNext_Plus_RL2_retain_Epoch1-Step4000_ValIoU0.6316584628283326.pth'
    dirs = '/data/duye/pretrained_models/FPNRLtrain/' + pre
    model.load_state_dict(torch.load(dirs))
    model.eval()
    loader = loadssTEM(batch_size=8)

    iou = []
    for index, batch in enumerate(loader):
        print('index: ', index)
        img = batch[0]
        WH = batch[-1]  # WH_dict
        left_WH = WH['left_WH']
        origion_WH = WH['origion_WH']
        object_WH = WH['object_WH']
        gt = batch[1]

        bs = img.shape[0]
        with torch.no_grad():
            pre_v2 = None
            pre_v1 = None
            # train_rl的时候, temperature=0.1
            result_dict = model(img.to(devices), pre_v2, pre_v1, mode='test', temperature=0.0)  # (bs, seq_len)
        pred_x = result_dict['final_pred_x'].cpu().numpy()[0]
        pred_y = result_dict['final_pred_y'].cpu().numpy()[0]
        pred_lengths = result_dict['lengths'].cpu().numpy()[0]
        pred_len = np.sum(pred_lengths) - 1  # sub EOS
        # vertices1 = []

        for b in range(bs):
            scaleW = 224.0 / float(object_WH[0][b])
            scaleH = 224.0 / float(object_WH[1][b])
            leftW = left_WH[0][b]
            leftH = left_WH[1][b]
            W = origion_WH[0][b]
            H = origion_WH[1][b]

            # Get the pred poly
            vertices1 = []
            for i in range(pred_len):
                vert = (pred_x[i] / scaleW + leftW,
                        pred_y[i] / scaleH + leftH)
                vertices1.append(vert)
            # vertices2 = []
            # gt mask
            gt_mask = np.array(gt[b])  # [W, H]
            # pre mask
            img1 = Image.new('L', (W, H), 0)
            ImageDraw.Draw(img1).polygon(vertices1, outline=1, fill=1)
            pre_mask = np.array(img1)  # (H, W) 这就对过来了!

            filt = np.sum(pre_mask)
            if filt <= 100:
                continue

            # get iou
            intersection = np.logical_and(gt_mask, pre_mask)
            union = np.logical_or(gt_mask, pre_mask)
            nu = np.sum(intersection)
            de = np.sum(union)
            iiou = nu / (de * 1.0) if de!=0 else 0.
            filter = False
            iou.append(iiou)
    print(len(iou))
    iou.sort()
    iou.reverse()
    print(iou)
    print('IoU:', np.mean(np.array(iou)))


# 全部得分：IoU: 0.6038216038377007 这就是GT


