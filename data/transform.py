from albumentations import CLAHE, GaussianBlur, IAASharpen, OpticalDistortion, GridDistortion, IAAPiecewiseAffine
from albumentations import OneOf, Compose, MotionBlur, MedianBlur, Blur, Transpose
from albumentations import RandomBrightnessContrast, RandomGamma, HorizontalFlip, ShiftScaleRotate, CoarseDropout
from albumentations import Resize, RandomCrop, Normalize, HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise
from albumentations.pytorch import ToTensorV2


def create_transform(config, mode):
    if mode == 'train':
        transforms = Compose([
            Resize(256, 256),
            HorizontalFlip(),
            CoarseDropout(p=0.3),
            Normalize(
                mean=config.train.mean,
                std=config.train.std,
            ),
            RandomCrop(config.train.img_size, config.train.img_size),
            ToTensorV2(),
        ])
    elif mode == 'val':
        transforms = Compose([
            Resize(config.val.img_size, config.val.img_size),
            Normalize(
                mean=config.val.mean,
                std=config.val.std,
            ),
            ToTensorV2()
        ])
    else:
        transforms = Compose([
            Resize(config.test.img_size, config.test.img_size),
            Normalize(
                mean=config.test.mean,
                std=config.test.std,
            ),
            ToTensorV2()
        ])
    return transforms
