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
import losses
import argparse

warnings.filterwarnings('ignore')
devices = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(config,
          load_resnet50=False,
          pre_trained=None,
          cur_epochs=0):
    batch_size = config['batch_size']
    lr = config['lr']
    epochs = config['epoch']
    train_dataloader = loadData('train', 16, 71, batch_size)
    val_loader = loadData('val', 16, 71, batch_size, shuffle=False)
    model = PolygonModel(load_predtrained_resnet50=load_resnet50,
                         predict_delta=True).to(devices)

    if pre_trained is not None:
        model.load_state_dict(torch.load(pre_trained))
        print('loaded pretrained polygon net!')

    # set to eval
    model.encoder.eval()
    model.delta_encoder.eval()

    for n, p in model.named_parameters():
        if 'encoder' in n:
            print('Not train:', n)
            p.requires_grad = False

    print('No weight decay in RL training')

    train_params = [p for p in model.parameters() if p.requires_grad]
    train_params1 = []
    train_params2 = []
    for n, p in model.named_parameters():
        if p.requires_grad and 'delta' not in n:
            train_params1.append(p)
        elif p.requires_grad and 'delta' in n:
            train_params2.append(p)

    # Adam 优化方法
    optimizer = optim.Adam(train_params, lr=lr, amsgrad=False)
    optimizer1 = optim.Adam(train_params1, lr=lr, amsgrad=False)
    optimizer2 = optim.Adam(train_params2, lr=lr, amsgrad=False)

    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=config['lr_decay'][0],
                                          gamma=config['lr_decay'][1])

    print('Total Epochs:', epochs)
    for it in range(cur_epochs, epochs):
        # init
        accum = defaultdict(float)
        accum2 = defaultdict(float)
        model.delta_model.train()
        model.decoder.train()
        for index, batch in enumerate(train_dataloader):
            img = torch.tensor(batch[0], dtype=torch.float).cuda()
            bs = img.shape[0]
            WH = batch[-1]  # WH_dict
            left_WH = WH['left_WH']
            origion_WH = WH['origion_WH']
            object_WH = WH['object_WH']

            # TODO： step1
            model.delta_model.eval()
            model.decoder.train()
            outdict_sample = model(img, mode='train_rl', temperature=config['temperature'],
                                   temperature2=0.0)  # (bs, seq_len, 28*28+1)
            # greedy
            with torch.no_grad():
                outdict_greedy = model(img, mode='train_rl', temperature=0.0)

            # Get RL loss
            sampling_pred_x = outdict_sample['final_pred_x'].cpu().numpy()
            sampling_pred_y = outdict_sample['final_pred_y'].cpu().numpy()
            sampling_pred_len = outdict_sample['lengths'].cpu().numpy()
            greedy_pred_x = outdict_greedy['final_pred_x'].cpu().numpy()
            greedy_pred_y = outdict_greedy['final_pred_y'].cpu().numpy()
            greedy_pred_len = outdict_greedy['lengths'].cpu().numpy()
            sampling_iou = np.zeros(bs, dtype=np.float32)
            greedy_iou = np.zeros(bs, dtype=np.float32)
            vertices_GT = []  # (bs, 70, 2)
            vertices_sampling = []
            vertices_greedy = []
            GT_polys = batch[-2].numpy()  # (bs, 70, 2)
            GT_mask = batch[7]  # (bs, 70)

            for ii in range(bs):
                scaleW = 224.0 / float(object_WH[0][ii])
                scaleH = 224.0 / float(object_WH[1][ii])
                leftW = left_WH[0][ii]
                leftH = left_WH[1][ii]
                tmp = []
                all_len = np.sum(GT_mask[ii].numpy())
                cnt_target = GT_polys[ii][:all_len]
                for vert in cnt_target:
                    tmp.append((vert[0] / scaleW + leftW,
                                vert[1] / scaleH + leftH))
                vertices_GT.append(tmp)

                tmp = []
                for j in range(sampling_pred_len[ii]-1):
                    vertex = (
                        sampling_pred_x[ii][j] / scaleW + leftW,
                        sampling_pred_y[ii][j] / scaleH + leftH
                    )
                    tmp.append(vertex)
                vertices_sampling.append(tmp)

                tmp = []
                for j in range(greedy_pred_len[ii] - 1):
                    vertex = (
                        greedy_pred_x[ii][j] / scaleW + leftW,
                        greedy_pred_y[ii][j] / scaleH + leftH
                    )
                    tmp.append(vertex)
                vertices_greedy.append(tmp)

            # IoU between sampling/greedy and GT
            for ii in range(bs):
                sam = vertices_sampling[ii]
                gt = vertices_GT[ii]
                gre = vertices_greedy[ii]

                if len(sam) < 2:
                    sampling_iou[ii] = 0.
                else:
                    iou_sam, _, _ = iou(sam, gt, origion_WH[1][ii], origion_WH[0][ii])
                    sampling_iou[ii] = iou_sam

                if len(gre) < 2:
                    greedy_iou[ii] = 0.
                else:
                    iou_gre, _, _ = iou(gre, gt, origion_WH[1][ii], origion_WH[0][ii])
                    greedy_iou[ii] = iou_gre

            logprobs = outdict_sample['log_probs']
            # 强化学习损失，logprob是两个logprob加和
            loss = losses.self_critical_loss(logprobs, outdict_sample['lengths'],
                                             torch.from_numpy(sampling_iou).to(devices),
                                             torch.from_numpy(greedy_iou).to(devices))
            model.zero_grad()
            if 'grid_clip' in config:
                nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            loss.backward()
            optimizer1.step()  # 更新参数

            accum['loss_total'] += loss
            accum['sampling_iou'] += np.mean(sampling_iou)
            accum['greedy_iou'] += np.mean(greedy_iou)
            # 打印损失
            if (index+1) % 20 == 0:
                print('Epoch {} - Step {}'.format(it+1, index+1))
                print('     Main Decoder: loss_total {}, sampling_iou {}, greedy_iou {}'.format(
                    accum['loss_total']/20,
                    accum['sampling_iou']/20,
                    accum['greedy_iou']/20))
                accum = defaultdict(float)

            # TODO:训练delta_model decoder step2
            model.decoder.eval()
            model.delta_model.train()

            outdict_sample = model(img, mode='train_rl', temperature=0.0,
                                   temperature2=config['temperature2'])  # (bs, seq_len, 28*28+1)
            # greedy
            with torch.no_grad():
                outdict_greedy = model(img, mode='train_rl', temperature=0.0, temperature2=0.0)

            # Get RL loss
            sampling_pred_x = outdict_sample['final_pred_x'].cpu().numpy()
            sampling_pred_y = outdict_sample['final_pred_y'].cpu().numpy()
            sampling_pred_len = outdict_sample['lengths'].cpu().numpy()
            greedy_pred_x = outdict_greedy['final_pred_x'].cpu().numpy()
            greedy_pred_y = outdict_greedy['final_pred_y'].cpu().numpy()
            greedy_pred_len = outdict_greedy['lengths'].cpu().numpy()
            sampling_iou = np.zeros(bs, dtype=np.float32)
            greedy_iou = np.zeros(bs, dtype=np.float32)
            vertices_GT = []  # (bs, 70, 2)
            vertices_sampling = []
            vertices_greedy = []
            GT_polys = batch[-2].numpy()  # (bs, 70, 2)
            GT_mask = batch[7]  # (bs, 70)

            for ii in range(bs):
                scaleW = 224.0 / object_WH[0][ii]
                scaleH = 224.0 / object_WH[1][ii]
                leftW = left_WH[0][ii]
                leftH = left_WH[1][ii]
                tmp = []
                all_len = np.sum(GT_mask[ii].numpy())
                cnt_target = GT_polys[ii][:all_len]
                for vert in cnt_target:
                    tmp.append((vert[0] / scaleW + leftW,
                                vert[1] / scaleH + leftH))
                vertices_GT.append(tmp)

                tmp = []
                for j in range(sampling_pred_len[ii]-1):
                    vertex = (
                        sampling_pred_x[ii][j] / scaleW + leftW,
                        sampling_pred_y[ii][j] / scaleH + leftH
                    )
                    tmp.append(vertex)
                vertices_sampling.append(tmp)

                tmp = []
                for j in range(greedy_pred_len[ii] - 1):
                    vertex = (
                        greedy_pred_x[ii][j] / scaleW + leftW,
                        greedy_pred_y[ii][j] / scaleH + leftH
                    )
                    tmp.append(vertex)
                vertices_greedy.append(tmp)

            # IoU between sampling/greedy and GT
            for ii in range(bs):
                sam = vertices_sampling[ii]
                gt = vertices_GT[ii]
                gre = vertices_greedy[ii]

                if len(sam) < 2:
                    sampling_iou[ii] = 0.
                else:
                    iou_sam, _, _ = iou(sam, gt, origion_WH[1][ii], origion_WH[0][ii])
                    sampling_iou[ii] = iou_sam

                if len(gre) < 2:
                    greedy_iou[ii] = 0.
                else:
                    iou_gre, _, _ = iou(gre, gt, origion_WH[1][ii], origion_WH[0][ii])
                    greedy_iou[ii] = iou_gre

            # TODO:
            logprobs = outdict_sample['delta_logprob']
            # 强化学习损失，logprob是两个logprob加和
            loss = losses.self_critical_loss(logprobs, outdict_sample['lengths'],
                                             torch.from_numpy(sampling_iou).to(devices),
                                             torch.from_numpy(greedy_iou).to(devices))
            model.zero_grad()
            if 'grid_clip' in config:
                nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            loss.backward()
            optimizer2.step()

            accum2['loss_total'] += loss
            accum2['sampling_iou'] += np.mean(sampling_iou)
            accum2['greedy_iou'] += np.mean(greedy_iou)
            # 打印损失
            if (index+1) % 20 == 0:
                print('     Second Decoder: loss_total {}, sampling_iou {}, greedy_iou {}'.format(
                    accum2['loss_total']/20,
                    accum2['sampling_iou']/20,
                    accum2['greedy_iou']/20))
                accum2 = defaultdict(float)

            if (index+1) % config['val_every'] == 0:
                # validation
                model.decoder.eval()
                model.delta_model.eval()
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
                            scaleW = 224.0 / float(object_WH[0][ii])
                            scaleH = 224.0 / float(object_WH[1][ii])
                            leftW = left_WH[0][ii]
                            leftH = left_WH[1][ii]

                            all_len = np.sum(val_mask_final[ii].numpy())
                            cnt_target = val_target[ii][:all_len]
                            for vert in cnt_target:
                                vertices2.append((vert[0]/scaleW + leftW,
                                                  vert[1]/scaleH + leftH))
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

                            _, nu_cur, de_cur = iou(vertices1, vertices2, origion_WH[1][ii], origion_WH[0][ii])
                            iou_cur = nu_cur * 1.0 / de_cur if de_cur != 0 else 0
                            val_IoU.append(iou_cur)

                val_iou_data = np.mean(np.array(val_IoU))
                print('Validation After Epoch {} - step {}'.format(str(it + 1), str(index + 1)))
                print('           IoU      on validation set: ', val_iou_data)
                print('less than 2: ', less_than2)
                print('Saving training parameters after this epoch:')
                torch.save(model.state_dict(),
                            '/data/duye/pretrained_models/FPNRLtrain/ResNext_Plus_RL2_retain_Epoch{}-Step{}_ValIoU{}.pth'.format(
                            str(it + 1),
                            str(index + 1),
                            str(val_iou_data)))
                # set to init
                model.decoder.train()  # important
                model.delta_model.train()

        scheduler.step()
        print('Epoch {} Completed!'.format(str(it+1)))

if __name__ == '__main__':

    parse = argparse.ArgumentParser(description='RL train in retain with lstm and the delta model.')
    parse.add_argument('-l', '--lr', type=float, default=2e-6)
    parse.add_argument('-b', '--bs', type=int, default=8)   # 实验中4更有效
    parse.add_argument('-w', '--wd', type=float, default=0.000001)
    parse.add_argument('-l2', '--ld', type=float, default=2)
    parse.add_argument('-g', '--grad_clip', type=float, default=40)
    parse.add_argument('-v', '--val_every', type=int, default=1000)
    # parse.add_argument('-lm', '--lamb', type=float, default=0.50)
    parse.add_argument('-p', '--pretrained', type=str, default='')
    parse.add_argument('-e', '--epoch', type=int, default=15)
    parse.add_argument('-t1', '--temperature1', type=float, default=0.1)
    parse.add_argument('-t2', '--temperature2', type=float, default=0.1)

    args = parse.parse_args()
    config = {}
    config['batch_size'] = args.bs
    config['lr'] = args.lr  # paper:2e-6
    config['num'] = 16
    # epochs over the whole dataset
    config['epoch'] = args.epoch
    config['lr_decay'] = [args.ld, 0.1]
    config['grad_clip'] = args.grad_clip
    config['val_every'] = args.val_every
    config['temperature'] = args.temperature1
    config['temperature2'] = args.temperature2
    config['weight_decay'] = args.wd
    # 传入train_delta
    pre = args.pretrained
    pre = 'ResNext_Plus_DeltaModel_Epoch7-Step3000_ValIoU0.619344842105648.pth'
    if pre == '':
        pass
    else:
        config['pretrained'] = '/data/duye/pretrained_models/' + pre
                            # 'Joint_FPN2_DeltaModel_Epoch9-Step6000_ValIoU0.6177457155898146.pth'
        train(config, load_resnet50=False, pre_trained=config['pretrained'])
