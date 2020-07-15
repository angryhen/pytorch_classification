import torch
import os
from lib.use_model import choice_model


model = choice_model('resnest50', 10)
if torch.cuda.is_available():
    model = model.cuda()

# model.load_state_dict(ch['state_dict'])
print(model)
# torch.distributed.init_process_group(backend='nccl',init_method='env://')# model = torch.load('/home/du/Desktop/my_project/pytorch_classification/weights/fold0_lowest_loss.pth')
# print(torch.distributed.get_rank())
# print(int(os.environ['SLURM_LOCALID']))

