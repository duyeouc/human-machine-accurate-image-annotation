# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.optim as optim
from models.model import PolygonModel
from utils import *
from dataloader import loadData
import warnings
import torch.nn as nn
import numpy as np
from collections import defaultdict
warnings.filterwarnings('ignore')


def train(config, load_resnet50=False, pre_trained=None, cur_epochs=0):

    batch_size = config['batch_size']
    lr = config['lr']
    epochs = config['epoch']

    train_dataloader = loadData('train', 16, 71, batch_size)
    val_loader = loadData('val', 16, 71, batch_size, shuffle=False)
    model = PolygonModel(load_predtrained_resnet50=load_resnet50,
                         predict_delta=False).cuda()
    # checkpoint
    if pre_trained is not None:
        model.load_state_dict(torch.load(pre_trained))
        print('loaded pretrained polygon net!')

    # Regulation，原paper没有+regulation
    no_wd = []
    wd = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            # No optimization for frozen params
            continue
        if 'bn' in name or 'convLSTM' in name or 'bias' in name:
            no_wd.append(param)
        else:
            wd.append(param)

    optimizer = optim.Adam(
                [
                    {'params': no_wd, 'weight_decay': 0.0},
                    {'params': wd}
                ],
                lr=lr,
                weight_decay=config['weight_decay'],
                amsgrad=False)

    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=config['lr_decay'][0],
                                          gamma=config['lr_decay'][1])

    print('Total Epochs:', epochs)
    for it in range(cur_epochs, epochs):
        accum = defaultdict(float)
        # accum['loss_total'] = 0.
        # accum['loss_lstm'] = 0.
        # accum['loss_delta'] = 0.
        for index, batch in enumerate(train_dataloader):
            img = torch.tensor(batch[0], dtype=torch.float).cuda()
            bs = img.shape[0]
            pre_v2 = torch.tensor(batch[2], dtype=torch.float).cuda()
            pre_v1 = torch.tensor(batch[3], dtype=torch.float).cuda()
            outdict = model(img, pre_v2, pre_v1, mode='train_ce')  # (bs, seq_len, 28*28+1)s

            out = outdict['logits']
            # 之前训练不小心加了下面这句
            # out = torch.nn.functional.log_softmax(out, dim=-1)  # logits->log_probs
            out = out.contiguous().view(-1, 28 * 28 + 1)  # (bs*seq_len, 28*28+1)
            target = batch[4]

            # smooth target
            target = dt_targets_from_class(np.array(target, dtype=np.int), 28, 2)  # (bs, seq_len, 28*28+1)
            target = torch.from_numpy(target).cuda().contiguous().view(-1, 28 * 28 + 1)  # (bs, seq_len, 28*28+1)
            # 交叉熵损失计算
            mask_final = batch[6]  # 结束符标志mask  (bs, seq_len(70)从第一个点开始)
            mask_final = torch.tensor(mask_final).cuda().view(-1)
            mask_delta = batch[7]
            mask_delta = torch.tensor(mask_delta).cuda().view(-1)  # (bs*70)
            loss_lstm = torch.sum(-target * torch.nn.functional.log_softmax(out, dim=1), dim=1)  # (bs*seq_len)
            loss_lstm = loss_lstm * mask_final.type_as(loss_lstm)  # 从end point截断损失计算
            loss_lstm = loss_lstm.view(bs, -1)  # (bs, seq_len)
            loss_lstm = torch.sum(loss_lstm, dim=1)  # sum over seq_len  (bs,)
            real_pointnum = torch.sum(mask_final.contiguous().view(bs, -1), dim=1)
            loss_lstm = loss_lstm / real_pointnum  # mean over seq_len
            loss_lstm = torch.mean(loss_lstm)  # mean over batch

            # loss = loss_lstm + loss_delta
            loss = loss_lstm
            #TODO: 这里train_ce可以用这个loss, 但train_rl可以根据条件概率重写损失函数
            model.zero_grad()

            if 'grid_clip' in config:
                nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

            loss.backward()

            accum['loss_total'] += loss
            optimizer.step()

            # 打印损失
            if (index+1) % 20 == 0:
                print('Epoch {} - Step {}, loss_total {}'.format(
                    it + 1,
                    index,
                    accum['loss_total']/20))
                accum = defaultdict(float)
            # 每3000step一次
            if (index+1) % config['val_every'] == 0:
                # validation
                model.eval()  # 原作者只eval了这个
                val_IoU = []
                less_than2 = 0
                with torch.no_grad():
                    for val_index, val_batch in enumerate(val_loader):
                        img = torch.tensor(val_batch[0], dtype=torch.float).cuda()
                        bs = img.shape[0]

                        WH = val_batch[-1]  # WH_dict
                        left_WH = WH['left_WH']
                        origion_WH = WH['origion_WH']
                        object_WH = WH['object_WH']

                        val_mask_final = val_batch[6]
                        val_mask_final = torch.tensor(val_mask_final).cuda().contiguous().view(-1)
                        out_dict = model(img, mode='test')  # (N, seq_len) # test_time
                        pred_polys = out_dict['pred_polys']  # (bs, seq_len)
                        tmp = pred_polys
                        pred_polys = pred_polys.contiguous().view(-1)  # (bs*seq_len)
                        val_target = val_batch[4]  # (bs, seq_len)
                        # 求accuracy
                        val_target = torch.tensor(val_target, dtype=torch.long).cuda().contiguous().view(
                            -1)  # (bs*seq_len)
                        val_acc1 = torch.tensor(pred_polys == val_target, dtype=torch.float).cuda()
                        val_acc1 = (val_acc1 * val_mask_final).sum().item()
                        val_acc1 = val_acc1 * 1.0 / val_mask_final.sum().item()
                        # 用作计算IoU
                        val_result_index = tmp.cpu().numpy()  # (bs, seq_len)
                        val_target = val_batch[4].numpy()  # (bs, seq_len)

                        # 求IoU
                        for ii in range(bs):
                            vertices1 = []
                            vertices2 = []
                            scaleW = 224.0 / object_WH[0][ii]
                            scaleH = 224.0 / object_WH[1][ii]
                            leftW = left_WH[0][ii]
                            leftH = left_WH[1][ii]
                            for label in val_result_index[ii]:
                                if label == 28 * 28:
                                    break
                                vertex = (
                                    ((label % 28) * 8.0 + 4) / scaleW + leftW, (
                                            (int(label / 28)) * 8.0 + 4) / scaleH + leftH)
                                vertices1.append(vertex)
                            for label in val_target[ii]:
                                if label == 28 * 28:
                                    break
                                vertex = (
                                    ((label % 28) * 8.0 + 4) / scaleW + leftW, (
                                            (int(label / 28)) * 8.0 + 4) / scaleH + leftH)
                                vertices2.append(vertex)
                            if len(vertices1) < 2:
                                less_than2 += 1
                                # IoU=0.
                                val_IoU.append(0.)
                                continue
                            _, nu_cur, de_cur = iou(vertices1, vertices2, origion_WH[1][ii], origion_WH[0][ii])  # (H, W)
                            iou_cur = nu_cur * 1.0 / de_cur if de_cur != 0 else 0
                            val_IoU.append(iou_cur)

                val_iou_data = np.mean(np.array(val_IoU))
                print('Validation After Epoch {} - step {}'.format(str(it + 1), str(index + 1)))
                print('           IoU      on validation set: ', val_iou_data)
                print('less than 2: ', less_than2)
                if it > 4:  # it = 5
                    print('Saving training parameters after this epoch:')
                    torch.save(model.state_dict(),
                               '/data/duye/pretrained_models/ResNext50_FPN_LSTM_Epoch{}-Step{}_ValIoU{}.pth'.format(
                                   str(it + 1),
                                   str(index + 1),
                                   str(val_iou_data)))
                # set to init
                model.train()  # important

        # 衰减
        scheduler.step()
        # 打印当前lr
        print()
        print('Epoch {} Completed!'.format(str(it+1)))
        print()

if __name__ == '__main__':
    config = {}
    config['batch_size'] = 8  # 有一篇paper说最好BatchSize<=32, 原作者设置batch_size=8
    # 适当加大下学习率应该是有用的, 0.0005貌似更好用一些
    config['lr'] = 0.0001
    config['num'] = 16
    # epochs over the whole dataset
    config['epoch'] = 25
    config['lr_decay'] = [5, 0.1]
    config['weight_decay'] = 0.00001  # 目前的结果没有weight-decay
    config['grad_clip'] = 40
    config['val_every'] = 3000

    train(config, load_resnet50=True)

