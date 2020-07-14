import torch
import os
import shutil

def save_checkpoint(state, epoch, is_best, is_lowest_loss, save_path):
    filename = os.path.join(save_path, f'{epoch}.pth')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save_path, f'best.pth'))
    if is_lowest_loss:
        shutil.copyfile(filename, os.path.join(save_path, f'lowest.pth'))