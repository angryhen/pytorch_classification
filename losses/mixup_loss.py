import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTargetLoss(nn.Module):
    def __init__(self):
        super(SoftTargetLoss, self).__init__()
        print('using mixup loss!')

    def forward(self,x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()