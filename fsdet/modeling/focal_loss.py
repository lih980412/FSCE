import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def BCEFocalLoss(predict, target, gamma=2, alpha=0.25, reduction='sum'):

    pt = torch.sigmoid(predict)  # sigmoid获取概率
    # 在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
    pt = torch.clamp(pt, min=1e-8, max=1 - 1e-8)
    loss = - alpha * (1 - pt) ** gamma * target * torch.log(pt) - (1 - alpha) * pt ** gamma * (
            1 - target) * torch.log(1 - pt)

    if reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'sum':
        loss = torch.sum(loss)
    return loss * 5


def MultiCEFocalLoss(predict, target, class_num, alpha=None, gamma=2, reduction="mean"):
    if alpha is None:
        alpha = Variable(torch.ones(class_num, 1))
    else:
        alpha = alpha
    eps = 1e-7
    class_mask = F.one_hot(target, class_num)
    # y_pred = predict.view(predict.size()[0], predict.size()[1])
    y_pred = F.softmax(predict, dim=1)
    # y_pred = torch.clamp(y_pred, min=1e-8, max=1 - 1e-8)
    target = class_mask.view(y_pred.size())
    ce = -1 * torch.log(y_pred + eps) * target
    floss = torch.pow((1 - y_pred), gamma) * ce
    floss = torch.mul(floss, alpha)
    floss = floss.sum(1)
    if reduction == 'mean':
        loss = torch.mean(floss)
    elif reduction == 'sum':
        loss = torch.sum(floss)
    return loss * 5


# class BCEFocalLoss(nn.Module):
#
#     def __init__(self, gamma=2, alpha=0.25, reduction='sum'):
#         self.gamma = gamma
#         self.alpha = alpha
#         self.reduction = reduction
#
#     def forward(self, predict, target):
#         pt = torch.sigmoid(predict)  # sigmoid获取概率
#         # 在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
#         pt = torch.clamp(pt, min=1e-8, max=1 - 1e-8)
#         loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (1 - self.alpha) * pt ** self.gamma * (
#                 1 - target) * torch.log(1 - pt)
#
#         if self.reduction == 'mean':
#             loss = torch.mean(loss)
#         elif self.reduction == 'sum':
#             loss = torch.sum(loss)
#         return loss
#
# class MultiCEFocalLoss(nn.Module):
#
#     def __init__(self, alpha=None, gamma=2, reduction="mean"):
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
#
#     def forward(self, predict, target, class_num):
#         if self.alpha is None:
#             alpha = Variable(torch.ones(class_num, 1))
#         else:
#             alpha = self.alpha
#         eps = 1e-7
#         class_mask = F.one_hot(target, class_num)
#         # y_pred = predict.view(predict.size()[0], predict.size()[1])
#         y_pred = F.softmax(predict, dim=1)
#         # y_pred = torch.clamp(y_pred, min=1e-8, max=1 - 1e-8)
#         target = class_mask.view(y_pred.size())
#         ce = -1 * torch.log(y_pred + eps) * target
#         floss = torch.pow((1 - y_pred), self.gamma) * ce
#         floss = torch.mul(floss, alpha)
#         if self.reduction == 'mean':
#             loss = floss.sum(1).mean()
#         elif self.reduction == 'sum':
#             loss = floss.sum(1).sum()
#         return loss