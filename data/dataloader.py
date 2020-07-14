import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader, sampler, distributed
import torch.distributed as dist

from data.datasets import create_dataset
from data.collator import get_collate_fn

def create_dataloader(config, data, mode):
    dataset = create_dataset(config, data, mode)
    if mode == 'train':
        # create Sampler
        if dist.is_available() and dist.is_initialized():
            train_RandomSampler = distributed.DistributedSampler(dataset)
        else:
            train_RandomSampler = sampler.RandomSampler(dataset, replacement=False)

        train_BatchSampler = sampler.BatchSampler(train_RandomSampler,
                                              batch_size=config.train.batch_size,
                                              drop_last=config.train.dataloader.drop_last)

        # Augment
        collator = get_collate_fn(config)

        # DataLoader
        data_loader = DataLoader(dataset=dataset,
                                batch_sampler=train_BatchSampler,
                                collate_fn=collator,
                                pin_memory=config.train.dataloader.pin_memory,
                                num_workers=config.train.dataloader.work_nums)

    elif mode == 'val':
        if dist.is_available() and dist.is_initialized():
            val_SequentialSampler = distributed.DistributedSampler(dataset)
        else:
            val_SequentialSampler = sampler.SequentialSampler(dataset)

        val_BatchSampler = sampler.BatchSampler(val_SequentialSampler,
                                                batch_size=config.val.batch_size,
                                                drop_last=config.val.dataloader.drop_last)
        data_loader = DataLoader(dataset,
                                batch_sampler=val_BatchSampler,
                                pin_memory=config.val.dataloader.pin_memory,
                                num_workers=config.val.dataloader.work_nums)
    else:
        if dist.is_available() and dist.is_initialized():
            test_SequentialSampler = distributed.DistributedSampler(dataset)
        else:
            test_SequentialSampler = None

        data_loader = DataLoader(dataset,
                                 sampler=test_SequentialSampler,
                                 batch_size=config.test.batch_size,
                                 pin_memory=config.val.dataloader.pin_memory,
                                 num_workers=config.val.dataloader.work_nums)
    return data_loader