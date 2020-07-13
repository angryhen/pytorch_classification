import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader, sampler

from data.datasets import create_dataset
from data.collator import get_collate_fn

def create_dataloader(config, data, mode):
    dataset = create_dataset(config, data, mode)
    if mode == 'train':
        # create Sampler
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

    else:
        val_SequentialSampler = sampler.SequentialSampler(dataset)
        val_BatchSampler = sampler.BatchSampler(val_SequentialSampler,
                                                batch_size=config.val.batch_size,
                                                drop_last=config.val.dataloader.drop_last)
        data_loader = DataLoader(dataset,
                                batch_sampler=val_BatchSampler,
                                pin_memory=config.val.dataloader.pin_memory,
                                num_workers=config.val.dataloader.work_nums)

    return data_loader