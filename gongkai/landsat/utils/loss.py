
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from functools import partial

import utils.loss_functions as fc
from utils.loss_functions import sigmoid_focal_loss, reduced_focal_loss


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, student_1, student_2, teacher_1, teacher_2):
        B, C, H, W = student_1.shape

        student_1 = student_1.view(B, C, H * W).permute(0, 2, 1)
        student_2 = student_2.view(B, C, H * W).permute(0, 2, 1)
        teacher_1 = teacher_1.view(B, C, H * W).permute(0, 2, 1)
        teacher_2 = teacher_2.view(B, C, H * W).permute(0, 2, 1)

        student_1 = F.normalize(student_1.float(), dim=2)
        student_2 = F.normalize(student_2.float(), dim=2)
        teacher_1 = F.normalize(teacher_1.float(), dim=2)
        teacher_2 = F.normalize(teacher_2.float(), dim=2)

        cos_sim_1 = F.cosine_similarity(student_1, teacher_1, dim=2)
        cos_sim_2 = F.cosine_similarity(student_2, teacher_2, dim=2)

        loss_1 = 1 - cos_sim_1
        loss_2 = 1 - cos_sim_2

        return (loss_1.mean() + loss_2.mean()) * 0.5




class ChangeSimilarity(nn.Module):

    def __init__(self, reduction='mean'):
        super(ChangeSimilarity, self).__init__()
        self.loss_f = nn.CosineEmbeddingLoss(margin=0., reduction=reduction)


    def forward(self, x1, x2, label_change):
        b, c, h, w = x1.size()
        x1 = F.softmax(x1, dim=1)
        x2 = F.softmax(x2, dim=1)
        x1 = x1.permute(0, 2, 3, 1)
        x2 = x2.permute(0, 2, 3, 1)
        x1 = torch.reshape(x1, [b * h * w, c])
        x2 = torch.reshape(x2, [b * h * w, c])

        label_unchange = ~label_change.bool()
        target = label_unchange.float()
        target = target - label_change.float()
        target = torch.reshape(target, [b * h * w])

        loss = self.loss_f(x1, x2, target)

        return loss


class DiceLoss(_Loss):
    __name__ = "dice_loss"

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - fc.f_score(y_pr, y_gt, beta=1, eps=self.eps, threshold=None, activation=self.activation)

