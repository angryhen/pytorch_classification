import torch.nn as nn
import torch
from losses.focal_loss import FocalLoss
from losses.mixup_loss import MixupLoss, SoftTargetLoss
from losses.smoothing_loss import LabelSmoothingLoss

def get_loss(config):
    if config.train.label_smooth:
        train_loss = LabelSmoothingLoss(config, 'mean')
    elif config.train.collator.type == 'mixup':
        train_loss = MixupLoss()
    elif config.train.collator.type == 'mixup2':
        train_loss = SoftTargetLoss()
    else:
        train_loss = nn.CrossEntropyLoss()
    val_loss = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        train_loss = train_loss.cuda()
        val_loss = val_loss.cuda()

    return train_loss, val_loss