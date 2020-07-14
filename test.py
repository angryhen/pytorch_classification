import torch
import os

torch.distributed.init_process_group(backend='nccl',init_method='env://')# model = torch.load('/home/du/Desktop/my_project/pytorch_classification/weights/fold0_lowest_loss.pth')
# model = torch.load('/home/du/Desktop/my_project/pytorch_classification/logs/2020-07-14 11:11:21/12.pth')
# print(model)
print(torch.distributed.get_rank())
# print(int(os.environ['SLURM_LOCALID']))

