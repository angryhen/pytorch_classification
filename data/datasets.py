import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .transform import create_transform


class MyDataset(Dataset):
    def __init__(self, img_df, transform=None):
        self.img_df = img_df
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = cv2.imread(self.img_df.iloc[index]['filename'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(image=img)
        return img['image'], torch.from_numpy(np.array(self.img_df.iloc[index]['label']))

    def __len__(self):
        return len(self.img_df)


def create_dataset(config, mode):
    train_transforms = create_transform(config, mode)
    val_transforms = create_transform(config, mode)
    train_dataset = MyDataset(config.train, train_transforms)
    val_dataset = MyDataset(config.val, val_transforms)

    return train_dataset, val_dataset