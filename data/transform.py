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
                    Transpose(),
                    CoarseDropout(p=0.3),
                    OneOf([
                        RandomBrightnessContrast(brightness_limit=0.6),
                        RandomGamma(),
                    ], p=0.6),
                    ShiftScaleRotate(rotate_limit=45),  # 75
                    OneOf([
                        CLAHE(p=0.5),
                        GaussianBlur(3, p=0.3),
                        IAASharpen(alpha=(0.2, 0.3), p=0.3),
                    ], p=1),  # 1
                    OneOf([
                        # 畸变相关操作
                        OpticalDistortion(p=0.3),
                        GridDistortion(p=0.2),
                        IAAPiecewiseAffine(p=0.3),
                    ], p=0.2),
                    # add
                    OneOf([
                        MotionBlur(p=0.3),
                        MedianBlur(blur_limit=3, p=0.3),
                        Blur(blur_limit=3, p=0.3),
                    ], p=0.8),
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
