import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.transform import create_transform


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


def create_dataset(config, data, mode):
    transforms = create_transform(config, mode)
    dataset = MyDataset(data, transforms)
    return dataset