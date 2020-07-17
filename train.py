import argparse
import os
import time

import apex
from apex.parallel import convert_syncbn_model, DistributedDataParallel as DDP
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from alfred.utils.log import logger
from sklearn.model_selection import KFold

from configs.config import get_cfg_defaults
from data.dataloader import create_dataloader
from lib.model import get_model
from lib.optimizer import get_optimizer
from lib.scheduler import CosineWarmupLr
from lib.tensorboard import get_tensorboard_writer
from losses.losses import get_loss
from utils.get_rank import get_rank
from utils.metrics import accuracy, AverageMeter, ProgressMeter
from utils.save import save_checkpoint
from utils.set_seed import set_seed
from utils.env_info import get_env_info

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--configs', type=str, default=None,
                    help='the yml which include all parameters!')
parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse_args()

torch.backends.cudnn.benchmark = True
global_step = 0
data_time = time.strftime("%Y-%m-%d %H:%M:%S")


def get_config():
    config = get_cfg_defaults()
    if args.configs:
        yml_file = args.configs
        config.merge_from_file(yml_file)
    config.merge_from_list(['dist_local_rank', args.local_rank])
    config.freeze()
    return config


def train(config, epoch, train_loader, model, optimizer, scheduler, train_loss, writer):
    global global_step

    # switch to train mode
    model.train()
    device = torch.device(config.device)
    print('device: ', device)

    logger.info(f'Epoches: {epoch}/{config.train.epoches}')

    # mertric
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, losses, top1, top5)
    end = time.time()

    # train
    for step, (images, targets) in enumerate(train_loader):
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
        if config.dist:
            loss_reduce = dist.all_reduce(loss, op=dist.ReduceOp.SUM, async_op=True)
            acc1_reduce = dist.all_reduce(acc1, op=dist.ReduceOp.SUM, async_op=True)
            acc5_reduce = dist.all_reduce(acc5, op=dist.ReduceOp.SUM, async_op=True)

            loss_reduce.wait()
            acc1_reduce.wait()
            acc5_reduce.wait()

            loss.div_(dist.get_world_size())
            acc1.div_(dist.get_world_size())
            acc5.div_(dist.get_world_size())

        batch_time.update(time.time() - end)
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        if get_rank() == 0:
            if step % config.train.preiod == 0 or step == len(train_loader):
                progress.pr2int(step)

            # add writer
            writer.add_scalar('Train/Loss', losses.avg, global_step)
            writer.add_scalar('Train/Acc-Top1', top1.avg, global_step)
            writer.add_scalar('Train/Acc-Top5', top5.avg, global_step)
            writer.add_scalar('Train/lr', scheduler.learning_rate, global_step)

        scheduler.step()
        end = time.time()


def val(config, val_loader, model, val_loss, writer):
    logger.info('Valid.......')

    # switch to evaluate mode
    model.eval()
    device = torch.device(config.device)

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@2', ':6.2f')

    with torch.no_grad():
        end = time.time()
        for i, (images, targets) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.to(device,
                                   non_blocking=config.train.dataloader.non_blocking)
                targets = targets.to(device,
                                     non_blocking=config.train.dataloader.non_blocking)


            # compute output
            output = model(images)
            loss = val_loss(output, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, targets, topk=(1, 5))
            if config.dist:
                loss_reduce = dist.all_reduce(loss, op=dist.ReduceOp.SUM, async_op=True)
                acc1_reduce = dist.all_reduce(acc1, op=dist.ReduceOp.SUM, async_op=True)
                acc5_reduce = dist.all_reduce(acc5, op=dist.ReduceOp.SUM, async_op=True)

                loss_reduce.wait()
                acc1_reduce.wait()
                acc5_reduce.wait()

                loss.div_(dist.get_world_size())
                acc1.div_(dist.get_world_size())
                acc5.div_(dist.get_world_size())

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


        # add writer
        if get_rank() == 0:
            logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                        .format(top1=top1, top5=top5))

            writer.add_scalar('Validate/Loss', losses.avg, global_step)
            writer.add_scalar('Validate/Acc-Top1', top1.avg, global_step)
            writer.add_scalar('Validate/Acc-Top5', top5.avg, global_step)

        if torch.cuda.is_available():
            return np.squeeze(top1.avg.cpu().numpy()), losses.avg
        else:
            return np.squeeze(top1.avg.numpy()), losses.avg


def main():
    config = get_config()
    set_seed(config)

    if get_rank() == 0:
        logger.info(get_env_info())
        logger.info(f'Distributed: {config.dist},'
                    f'Apex: {config.apex},'
                    f'Sync_bn: {config.dist_sync_bn}')

    # dist
    if config.dist:
        dist.init_process_group(backend=config.dist_backend,
                                init_method=config.dist_init_method)
        torch.cuda.set_device(config.dist_local_rank)

    # create log path
    val_log_file = os.path.join(config.log_dir, data_time) + '/log.txt'
    writer_log_file = os.path.join(config.log_dir, data_time)
    save_path = os.path.join(config.log_dir, data_time)

    # tensorboard
    if get_rank() == 0:
        writer = get_tensorboard_writer(writer_log_file, purge_step=None)
    else:
        writer = None

    # model
    model = get_model(config)

    # optimizer
    optimizer = get_optimizer(config, model)

    if config.apex:
        model, optimizer = apex.amp.initialize(model,
                                               optimizer,
                                               opt_level=config.apex_mode)

    if config.dist:
        if config.dist_sync_bn:
            if config.apex:
                model = convert_syncbn_model(model)
            else:
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if config.apex:
            model = DDP(model,delay_allreduce=True)
        else:
            model = nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[config.dist_local_rank],
                                                        output_device=config.dist_local_rank)

    # loss
    train_loss, val_loss = get_loss(config)

    data = pd.read_csv(config.train.dataset)
    skf = KFold(n_splits=10,shuffle=True, random_state=452)
    for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(data['filename'].values, data['filename'].values)):
        if fold_idx == config.train.fold:
            break

        # split data
        train_data = data.iloc[train_idx]
        val_data = data.iloc[val_idx]
        if get_config()==0:
            logger.info(f"Splited train set: {train_data.shape}")
            logger.info(f"Splited val set: {val_data.shape}")

        # create dataloader
        train_loader = create_dataloader(config, train_data, 'train')
        val_loader = create_dataloader(config, val_data, 'val')

        scheduler = CosineWarmupLr(config, optimizer, len(train_loader))

        best_precision, lowest_loss = 0, 100
        for epoch in range(config.train.epoches):
            # if config.dist:
            #     train_loader.sampler.set_epoch(epoch)
            train(config, epoch, train_loader, model, optimizer, scheduler, train_loss, writer)

            if epoch % config.train.val_preiod == 0:
                precision, avg_loss = val(config, val_loader, model, val_loss, writer)
                if get_rank() == 0:
                    with open(val_log_file, 'a') as acc_file:
                        acc_file.write(
                            f'Fold: {fold_idx:2d}, '
                            f'Epoch: {epoch:2d}, '
                            f'Precision: {precision:.8f}, '
                            f'Loss: {avg_loss:.8f}\n')

                    is_best = precision > best_precision
                    is_lowest_loss = avg_loss < lowest_loss
                    best_precision = max(precision, best_precision)
                    lowest_loss = min(avg_loss, lowest_loss)
                    state = {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'best_precision': best_precision,
                        'lowest_loss': lowest_loss,
                    }
                    save_checkpoint(state, epoch, is_best, is_lowest_loss, save_path)

        writer.close()
        torch.cuda.empty_cache()

if __name__ == '__main__':

    main()