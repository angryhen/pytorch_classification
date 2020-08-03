import torch
import os
import shutil
import time

def save_checkpoint(state, epoch, is_best, is_lowest_loss, save_path):
    filename = os.path.join(save_path, f'{epoch}.pth')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save_path, f'best.pth'))
    if is_lowest_loss:
        shutil.copyfile(filename, os.path.join(save_path, f'lowest.pth'))


def create_logFile(config):
    data_time = time.strftime("%m-%d_%H:%M")
    log_name = data_time + f'_{config.model.name}_c{config.model.num_classes}'
    val_logFile = os.path.join(config.log_dir, log_name) + '/log.txt'
    writer_logFile = os.path.join(config.log_dir, log_name)
    save_path = os.path.join(config.log_dir, log_name)
    return val_logFile, writer_logFile, save_path