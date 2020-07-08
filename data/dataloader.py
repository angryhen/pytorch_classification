import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader, sampler

from .datasets import create_dataset


def create_dataloader(config, mode):
    if mode == 'train':
        # create dataset
        train_dataset, val_dataset = create_dataset(config, mode)

        # create Sampler
        train_RandomSampler = sampler.RandomSampler(train_dataset, replacement=False)
        val_SequentialSampler = sampler.SequentialSampler(val_dataset)

        train_BatchSampler = sampler.BatchSampler(train_RandomSampler,
                                                  batch_size=config.train.batch_size,
                                                  drop_last=config.train.dataloader.drop_last)
        val_BatchSampler = sampler.BatchSampler(val_SequentialSampler,
                                                batch_size=config.val.batch_size,
                                                drop_last=config.val.dataloader.drop_last)

        # Augment
        collator = get_collate_fn(config.train.dataloader.collate_fn)

        # DataLoader
        train_loader = DataLoader(train_dataset,
                                  batch_sampler=train_BatchSampler,
                                  collate_fn=collator,
                                  pin_memory=config.train.dataloader.pin_memory,
                                  num_workers=config.train.dataloader.work_nums)
        val_loader = DataLoader(val_dataset,
                                batch_sampler=val_BatchSampler,
                                pin_memory=config.val.dataloader.pin_memory,
                                num_workers=config.val.dataloader.work_nums)

        return train_loader, val_loader