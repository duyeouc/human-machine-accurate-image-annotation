# -*- coding: utf-8 -*-
from __future__ import print_function
import torch.optim as optim
from models.model import PolygonModel
from utils import *
from dataloader import loadData
import warnings
import torch.nn as nn
import numpy as np
from collections import defaultdict
import argparse
warnings.filterwarnings('ignore')
devices = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(config, load_resnet50=False, pre_trained=None, cur_epochs=0):

    batch_size = config['batch_size']
    lr = config['lr']
    epochs = config['epoch']

    train_dataloader = loadData('train', 16, 71, batch_size)
    val_loader = loadData('val', 16, 71, batch_size*2, shuffle=False)
    model = PolygonModel(load_predtrained_resnet50=load_resnet50,
                         predict_delta=True).to(devices)
    # checkpoint
    if pre_trained is not None:
        # model.load_state_dict(torch.load(pre_trained))
        # 逐参数load
        dict = torch.load(pre_trained)
        pre_name = []
        for name in dict:
            pre_name.append(name)
        for name in model.state_dict():
            if name in pre_name:
                model.state_dict()[name].data.copy_(dict[name])
        print('loaded pretrained polygon net!')

    # Set to eval
    model.encoder.eval()
    for n, p in model.named_parameters():
        if 'encoder' in n and 'delta' not in n:
            print('Not train:', n)
            p.requires_grad = False
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
        for index, batch in enumerate(train_dataloader):
            img = torch.tensor(batch[0], dtype=torch.float).cuda()
            bs = img.shape[0]
            pre_v2 = torch.tensor(batch[2], dtype=torch.float).cuda()
            pre_v1 = torch.tensor(batch[3], dtype=torch.float).cuda()
            outdict = model(img, pre_v2, pre_v1, mode='train_ce')  # (bs, seq_len, 28*28+1)s
            out = outdict['logits']
            out = out.contiguous().view(-1, 28 * 28 + 1)  # (bs*seq_len, 28*28+1)
            target = batch[4]
            mask_delta = batch[7]
            mask_delta = torch.tensor(mask_delta).cuda().view(-1)  # (bs*70)
            # # smooth target
            target = dt_targets_from_class(np.array(target, dtype=np.int), 28, 2)  # (bs, seq_len, 28*28+1)
            target = torch.from_numpy(target).cuda().contiguous().view(-1, 28 * 28 + 1)  # (bs, seq_len, 28*28+1)
            # Cross-Entropy Loss
            mask_final = batch[6]  # 结束符标志mask
            mask_final = torch.tensor(mask_final).cuda().view(-1)
            loss_lstm = torch.sum(-target * torch.nn.functional.log_softmax(out, dim=1), dim=1)  # (bs*seq_len)
            loss_lstm = loss_lstm * mask_final.type_as(loss_lstm)  # 从end point截断损失计算
            loss_lstm = loss_lstm.view(bs, -1)  # (bs, seq_len)
            loss_lstm = torch.sum(loss_lstm, dim=1)  # sum over seq_len  (bs,)
            real_pointnum = torch.sum(mask_final.contiguous().view(bs, -1), dim=1)
            loss_lstm = loss_lstm / real_pointnum  # mean over seq_len
            loss_lstm = torch.mean(loss_lstm)  # mean over batch

            # Delta prediction Cross-Entropy Loss
            delta_target = prepare_delta_target(outdict['pred_polys'], torch.tensor(batch[-2]).cuda())
            delta_target = dt_targets_from_class(np.array(delta_target.cpu().numpy(), dtype=np.int), 15, 1)  # No smooth
            delta_target = torch.from_numpy(delta_target[:, :, :-1]).cuda().contiguous().view(-1, 15*15)
            delta_logits = outdict['delta_logits'][:, :-1, :]  # (bs, 70, 225)
            delta_logits = delta_logits.contiguous().view(-1, 15*15)  # (bs*70, 225)
            # TODO:get delta loss
            tmp = torch.sum(-delta_target * torch.nn.functional.log_softmax(delta_logits, dim=1), dim=1)
            tmp = tmp * mask_delta.type_as(tmp)
            tmp = tmp.view(bs, -1)
            # sum over len_s  (bs,)
            tmp = torch.sum(tmp, dim=1)
            real_pointnum2 = torch.sum(mask_delta.contiguous().view(bs, -1), dim=1)
            tmp = tmp / real_pointnum2
            loss_delta = torch.mean(tmp)

            loss = config['lambda'] * loss_delta + loss_lstm
            model.zero_grad()

            if 'grid_clip' in config:
                nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            loss.backward()
            accum['loss_total'] += loss
            accum['loss_lstm'] += loss_lstm
            accum['loss_delta'] += loss_delta
            optimizer.step()

            # 打印损失
            if (index+1) % 20 == 0:
                print('Epoch {} - Step {}, loss_total: {} [Loss lstm: {}, loss delta: {}]'.format(
                    it + 1,
                    index + 1,
                    accum['loss_total']/20,
                    accum['loss_lstm']/20,
                    accum['loss_delta']/20))
                accum = defaultdict(float)

            # 每3000step一次
            if (index+1) % config['val_every'] == 0:
                # validation
                model.delta_encoder.eval()
                model.delta_model.eval()
                model.decoder.eval()
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
                        # target，在224*224中的坐标
                        val_target = val_batch[-2].numpy()  # (bs, 70, 2)
                        val_mask_final = val_batch[7]  # (bs, 70)
                        out_dict = model(img, mode='test')  # (N, seq_len) # test_time
                        pred_x = out_dict['final_pred_x'].cpu().numpy()
                        pred_y = out_dict['final_pred_y'].cpu().numpy()
                        pred_len = out_dict['lengths']  # 预测的长度
                        # 求IoU
                        for ii in range(bs):
                            vertices1 = []
                            vertices2 = []
                            scaleW = 224.0 / object_WH[0][ii]
                            scaleH = 224.0 / object_WH[1][ii]
                            leftW = left_WH[0][ii]
                            leftH = left_WH[1][ii]

                            all_len = np.sum(val_mask_final[ii].numpy())
                            cnt_target = val_target[ii][:all_len]
                            for vert in cnt_target:
                                vertices2.append((vert[0]/scaleW + leftW,
                                                  vert[1]/scaleH + leftH))

                            # print('target:', cnt_target)

                            pred_len_b = pred_len[ii] - 1
                            if pred_len_b < 2:
                                val_IoU.append(0.)
                                less_than2 += 1
                                continue

                            for j in range(pred_len_b):
                                vertex = (
                                    pred_x[ii][j] / scaleW + leftW,
                                    pred_y[ii][j] / scaleH + leftH
                                )
                                vertices1.append(vertex)

                            _, nu_cur, de_cur = iou(vertices1, vertices2, origion_WH[1][ii], origion_WH[0][ii])  # (H, W)
                            iou_cur = nu_cur * 1.0 / de_cur if de_cur != 0 else 0
                            val_IoU.append(iou_cur)

                val_iou_data = np.mean(np.array(val_IoU))
                print('Validation After Epoch {} - step {}'.format(str(it + 1), str(index + 1)))
                print('           IoU      on validation set: ', val_iou_data)
                print('less than 2: ', less_than2)
                if it > 4:  # it = 6
                    print('Saving training parameters after this epoch:')
                    torch.save(model.state_dict(),
                               '/data/duye/pretrained_models/ResNext_Plus_DeltaModel_Epoch{}-Step{}_ValIoU{}.pth'.format(
                                   str(it + 1),
                                   str(index + 1),
                                   str(val_iou_data)))
                # set to init
                model.delta_encoder.train()
                model.delta_model.train()  # important
                model.decoder.train()

        # 衰减
        scheduler.step()
        print()
        print('Epoch {} Completed!'.format(str(it+1)))
        print()

if __name__ == '__main__':

    parse = argparse.ArgumentParser(description='joint train of lstm and the delta model.')
    parse.add_argument('-l', '--lr', type=float, default=0.00001)
    parse.add_argument('-b', '--bs', type=int, default=8)
    parse.add_argument('-w', '--wd', type=float, default=0.00001)
    parse.add_argument('-l2', '--ld', type=float, default=2)
    parse.add_argument('-g', '--grad_clip', type=float, default=40)
    parse.add_argument('-v', '--val_every', type=int, default=3000)
    parse.add_argument('-lm', '--lamb', type=float, default=0.50)
    parse.add_argument('-p', '--pretrained', type=str, default='')
    parse.add_argument('-e', '--epoch', type=int, default=15)

    args = parse.parse_args()
    config = {}
    config['batch_size'] = args.bs
    config['lr'] = args.lr
    config['num'] = 16
    # epochs over the whole dataset
    config['epoch'] = args.epoch
    config['lr_decay'] = [args.ld, 0.1]
    config['weight_decay'] = args.wd
    config['grad_clip'] = args.grad_clip
    config['val_every'] = args.val_every
    config['lambda'] = args.lamb  # 两种方法，一是联合训练一起计算损失，二是也In retain
    # 传入train_lstm
    pretrained = args.pretrained
    if pretrained == '':
        config['pretrained'] = None
        train(config, pre_trained=config['pretrained'], load_resnet50=True, cur_epochs=0)
    else:
        config['pretrained'] = '/data/duye/pretrained_models/' + pretrained
                           # 'New_Res50_FPN_LSTM_Epoch10-Step6000_ValIoU0.6071237347797661.pth'
        train(config, pre_trained=config['pretrained'], load_resnet50=True, cur_epochs=0)


