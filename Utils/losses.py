# -*- coding: utf-8 -*-
from __future__ import print_function
from PIL import Image, ImageDraw
import numpy as np
import torch
import cv2
from scipy.ndimage.morphology import distance_transform_cdt
from .utils import dt_targets_from_class
import warnings
warnings.filterwarnings("ignore")
devices = 'cuda' if torch.cuda.is_available() else 'cpu'

# TODO: 把loss统一起来
# TODO: 写好新的loss_rl，参考一下curveGCN中的做法
# TODO：试一下卢策吾组的显式编码，curveGCN中的curve曲线方法?
# TODO：找一下新的人机交互训练方法?
# TODO: 好好读这个RL，可能可以用：https://blog.csdn.net/u013236946/article/details/73195035



# 强化学习训练的损失
def self_critical_loss(log_probs, lengths, sampling_reward, greedy_reward):
    """
    Self critical sequence training loss for RL

    log_probs: [batch_size, time_steps]
    lengths: [batch_size], containing the lengths of the predicted polygons
    sampling_reward: [batch_size, ]
    greedy_reward: [batch_size, ]
    """
    reward = sampling_reward - greedy_reward

    loss = 0

    """
         total_loss = losses.self_critical_loss(output['log_probs'], output['lengths'],
                torch.from_numpy(sampling_ious).to(device), torch.from_numpy(greedy_ious).to(device)) 
        log_probs: list[b, seq_len] item:是个数字,即预测的点的概率 (.max)
        lengths: [bs, seq_len] 实际预测的长度
    """

    # 逐个样例计算
    for i in torch.arange(reward.size(0), dtype=torch.long, device=reward.device):
        l = -1 * log_probs[:lengths[i]] * reward[i]
        # Expectation per polygon
        l = torch.mean(l)
        loss += l

    # mean across batches
    return loss / reward.size(0)


def mle_loss(pred, gt, mask, grid_size):
    """mle loss: pred
    :param: pred: [bs, len_s, classes]
    :param: gt:  [bs, len_s]
    :param: mask: [bs, len_s]
    """
    target = dt_targets_from_class(np.array(gt, dtype=np.int), 28, 2)  # (bs, seq_len, 28*28+1)
    target = torch.from_numpy(target).to(devices).contiguous().view(-1, 28 * 28 + 1)  # (bs, seq_len, 28*28+1)
    # 交叉熵损失计算
    gt_len = torch.sum(mask, dim=-1)
    torch.sum(-gt * np.torch.nn.functional.log_softmax(pred, dim=-1), )
    pass

def delta_loss(pred_logits, gt, mask):
    """
    delta loss
    :param pred_logits: (bs*seq_len, 15*15)
    :param gt: (bs, 14*14)
    :param mask: (-1)
    :return:
    """
    bs = pred_logits.shape[0]

    loss = torch.sum(-gt * torch.nn.functional.log_softmax(pred_logits, dim=-1), dim=1)

    loss = (loss * mask.type_as(loss)).view(bs, -1)

    loss = torch.sum(loss, dim=1)

    real_pointnum = torch.sum(mask.contiguous().view(bs, -1), dim=1)
    loss = loss / real_pointnum
    return torch.mean(loss)


def iou_loss(poly1, poly2, mask):
    pass