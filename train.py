import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from alfred.utils.log import logger
from tqdm import tqdm
import pathlib
import torch
import torchvision
from tensorboardX import SummaryWriter
import apex

from configs.config import get_cfg_defaults
from data.dataloader import create_dataloader
from data.collator import targets_to_device
from lib.use_model import choice_model
from losses.losses import get_loss
from lib.optimizer import get_optimizer
from lib.tensorboard import get_tensorboard_writer
from lib.scheduler import get_scheduler


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--configs', type=str, default='c15',
                    help='the yml which include all parameters!')
args = parser.parse_args()
global_step = 0

def get_config():
    config = get_cfg_defaults()
    return config

def subdivide_batch(config, data, targets):
    subdivision = config.train.subdivision

    if subdivision == 1:
        return [data], [targets]

    data_chunks = data.chunk(subdivision)
    if config.augmentation.use_mixup or config.augmentation.use_cutmix:
        targets1, targets2, lam = targets
        target_chunks = [(chunk1, chunk2, lam) for chunk1, chunk2 in zip(
            targets1.chunk(subdivision), targets2.chunk(subdivision))]
    elif config.augmentation.use_ricap:
        target_list, weights = targets
        target_list_chunks = list(
            zip(*[target.chunk(subdivision) for target in target_list]))
        target_chunks = [(chunk, weights) for chunk in target_list_chunks]
    else:
        target_chunks = targets.chunk(subdivision)
    return data_chunks, target_chunks

def main():
    config = get_config()
    device = torch.device(config.device)
    print(device)

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
        model = choice_model(config.model.name, config.train.num_classes)
        model.to(device)

        # optimizer
        optimizer = get_optimizer(config, model)
        model, optimizer = apex.amp.initialize(model,
                                               optimizer,
                                               opt_level=config.apex_mode)
        scheduler = get_scheduler(config, optimizer, len(train_loader))
        # loss
        train_loss, val_loss = get_loss(config)
        # tensorboard
        writer = get_tensorboard_writer(config.tensorboard.log_dir,
                                        purge_step=None)

        for epoch in range(config.train.epoches):


            # switch to train mode
            model.train()

            for step, (images, targets) in enumerate(train_loader):
                global_step = 0
                step += 1

                images = images.to(device,
                                   non_blocking=config.train.dataloader.non_blocking)
                targets = targets_to_device(config, targets, device)
                data_chunks, target_chunks = subdivide_batch(config, images, targets)
                optimizer.zero_grad()
                outputs = []
                losses = []
                for data_chunk, target_chunk in zip(data_chunks, target_chunks):
                    # print(data_chunk[1])
                    output_chunk = model(data_chunk)
                    outputs.append(output_chunk)

                    loss = train_loss(output_chunk, target_chunk)
                    losses.append(loss)
                    with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                outputs = torch.cat(outputs)
                optimizer.step()

                loss_ = sum(losses)
                loss_num = loss_.item()
                print(epoch, step, loss_num, optimizer.param_groups[0]['lr'])
                scheduler.step()




if __name__ == '__main__':
    main()