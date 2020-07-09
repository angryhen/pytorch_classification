import torch.nn as nn
from losses.focal_loss import FocalLoss
from losses.mixup_loss import MixupLoss

def get_loss(config):
    if config.train.collator.type == 'mixup':
        train_loss = MixupLoss()
    else:
        train_loss = nn.CrossEntropyLoss()
    val_loss = nn.CrossEntropyLoss()
    return train_loss, val_loss