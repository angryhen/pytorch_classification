import torch
import torch.nn as nn


class MixupLoss:
    def __init__(self):
        self.cross_loss = nn.CrossEntropyLoss()

    def __call__(self, predict, targets):
        targets_a, targets_b, lam = targets
        loss = self.cross_loss(predict, targets_a) * lam + (
            self.cross_loss(predict, targets_b) * (1 - lam)
            )

        return loss
