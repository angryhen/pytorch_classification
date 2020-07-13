import argparse
import pandas as pd
import time
import cv2
import numpy as np
import os
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
from lib.scheduler import CosineWarmupLr
from utils.metrics import accuracy, AverageMeter, ProgressMeter

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--configs', type=str, default='c15',
                    help='the yml which include all parameters!')
args = parser.parse_args()

global_step = 0


# 定义模型的存储
def save_checkpoint(config, state, epoch):
    filename = os.path.join(config.model.save_path, f'{epoch}.pth')
    torch.save(state, filename)


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
        if config.apex:
            model, optimizer = apex.amp.initialize(model,
                                                   optimizer,
                                                   opt_level=config.apex_mode)

        scheduler = CosineWarmupLr(config, optimizer, len(train_loader))

        # loss
        train_loss, val_loss = get_loss(config)

        # tensorboard
        writer = get_tensorboard_writer(config.tensorboard.log_dir,
                                        purge_step=None)

        for epoch in range(config.train.epoches):
            logger.info(f'Epoches: {epoch}/{config.train.epoches}')

            # mertric
            batch_time = AverageMeter('Time', ':6.3f')
            losses = AverageMeter('Loss', ':.4e')
            top1 = AverageMeter('Acc@1', ':6.2f')
            top5 = AverageMeter('Acc@5', ':6.2f')
            progress = ProgressMeter(len(train_loader), batch_time, losses, top1, top5)

            # switch to train mode
            model.train()
            end = time.time()
            # train
            for step, (images, targets) in enumerate(train_loader):
                global global_step
                global_step += 1
                step += 1

                images = images.to(device,
                                   non_blocking=config.train.dataloader.non_blocking)
                targets = targets.to(device,
                                   non_blocking=config.train.dataloader.non_blocking)

                outputs = model(images)
                optimizer.zero_grad()
                loss = train_loss(outputs, targets)

                if config.apex:
                    with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()

                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                batch_time.update(time.time() - end)
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                if step % config.train.preiod == 0 or step == len(train_loader):
                    progress.pr2int(step)

                # add writer
                writer.add_scalar('Train/Loss', losses.avg, global_step)
                writer.add_scalar('Train/Acc-Top1', top1.avg, global_step)
                writer.add_scalar('Train/Acc-Top5', top5.avg, global_step)
                writer.add_scalar('Train/lr', scheduler.learning_rate, global_step)


                scheduler.step()
                end = time.time()

            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
            }
            save_checkpoint(config, state,epoch)


if __name__ == '__main__':

    main()