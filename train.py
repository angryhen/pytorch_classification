import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from alfred.utils.log import logger
from tqdm import tqdm
import pathlib
import torch
import torchvision

from configs.config import get_cfg_defaults
from data.dataloader import create_dataloader
from lib.use_model import choice_model
from losses.losses import get_loss

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--configs', type=str, default='c15',
                    help='the yml which include all parameters!')
args = parser.parse_args()


def get_config():
    config = get_cfg_defaults()
    return config


def main():
    config = get_config()

    data = pd.read_csv(config.train.dataset)
    skf = KFold(n_splits=10,shuffle=True, random_state=452)
    for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(data['filename'].values, data['filename'].values)):
        if fold_idx == config.train.fold:
            break

        # split data
        train_data = data.iloc[train_idx]
        val_data = data.iloc[val_idx]

        print(val_data)
        logger.info(f'Now is traing fold {fold_idx + 1}')
        logger.info(f"Splited train set: {train_data.shape}")
        logger.info(f"Splited val set: {val_data.shape}")

        # create dataloader
        train_loader = create_dataloader(config, train_data, 'train')
        val_loader = create_dataloader(config, val_data, 'val')

        # model
        model = choice_model(config.model.name,
                             config.train.num_classes)

        # loss
        loss = get_loss(config)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model).cuda()
            # loss = loss.cuda()
        elif torch.cuda.is_available():
            model = model.cuda()
            # loss = loss.cuda()
        else:
            model = model
            loss = loss

        # optimizer
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.train.scheduler.base_lr,
                                     weight_decay=config.train.scheduler.weight_decay,
                                     amsgrad=True)


if __name__ == '__main__':
    main()