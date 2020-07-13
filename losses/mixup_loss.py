import torch
import torch.nn as nn
import torch.nn.functional as F


class MixupLoss:
    def __init__(self):
        self.cross_loss = nn.CrossEntropyLoss()

    def __call__(self, predict, targets):
        targets_a, targets_b, lam = targets
        loss = self.cross_loss(predict, targets_a) * lam + (
            self.cross_loss(predict, targets_b) * (1 - lam)
            )

        return loss

class SoftTargetLoss(nn.Module):
    def __init__(self):
        super(SoftTargetLoss, self).__init__()

    def forward(self,x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()