import numpy as np
import torch
import cv2
from torch.utils.data.dataloader import default_collate


def one_hot(x, num_classes, on_value=1., off_value=0., device='cuda'):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)


def mixup_target(target, num_classes, lam=1., smoothing=0.0, device='cuda'):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value, device=device)
    y2 = one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value, device=device)
    return lam*y1 + (1. - lam)*y2


def mixup_batch(input, target, alpha=0.2, num_classes=1000, smoothing=0.1, disable=False):
    lam = 1.
    if not disable:
        lam = np.random.beta(alpha, alpha)
    input = input.mul(lam).add_(1 - lam, input.flip(0))
    target = mixup_target(target, num_classes, lam, smoothing)
    return input, target


class FastCollateMixup:
    def __init__(self, config, label_smoothing=0.1):
        self.mixup_alpha = config.train.collator.mixup_alpha
        self.label_smoothing = label_smoothing
        self.num_classes = config.train.num_classes
        self.mixup_enabled = True

    def __call__(self, batch):
        batch_size = len(batch)
        lam = 1.
        if self.mixup_enabled:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

        target = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        target = mixup_target(target, self.num_classes, lam, self.label_smoothing, device='cpu')

        tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)
        # for i in range(batch_size):
        #     mixed = batch[i][0].astype(np.float32) * lam + \
        #             batch[batch_size - i - 1][0].astype(np.float32) * (1 - lam)
        #     np.round(mixed, out=mixed)
        #     tensor[i] += torch.from_numpy(mixed.astype(np.uint8))

        for i in range(batch_size):
            mixed = batch[i][0].numpy().astype(np.float32) * lam + \
                    batch[batch_size - i - 1][0].numpy().astype(np.float32) * (1 - lam)
            # # np.round(mixed, out=mixed)
            # mixed1 = mixed.astype(np.uint8)
            # mixed1 = np.transpose(mixed1, (1,2,0))
            # mixed1 = cv2.cvtColor(mixed1,cv2.COLOR_RGB2BGR)
            # # cv2.imshow('test', mixed1)
            # # cv2.waitKey(0)
            tensor[i] += torch.from_numpy(mixed.astype(np.uint8))
        tensor = tensor.type(torch.FloatTensor)

        return tensor, target

    
class MixupCollator:
    def __init__(self, config):
        self.alpha = config.train.collator.mixup_alpha

    def __call__(self, batch) :
        batch = default_collate(batch)
        batch = mixup(batch, self.alpha)
        return batch


def mixup(batch, alpha):
    data, targets = batch
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    data = data * lam + shuffled_data * (1 - lam)
    targets = (targets, shuffled_targets, lam)

    return data, targets


class CutMixCollator:
    def __init__(self, config):
        self.alpha = config.train.collator.cutmix_alpha

    def __call__(self, batch):
        batch = torch.utils.data.dataloader.default_collate(batch)
        batch = cutmix(batch, self.alpha)
        return batch


def cutmix(batch, alpha):
    data, targets = batch

    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)

    image_h, image_w = data.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]
    targets = (targets, shuffled_targets, lam)

    return data, targets


def get_collate_fn(config):
    if config.train.collator.type == 'mixup':
        return MixupCollator(config)
    elif config.train.collator.type == 'mixup2':
        return FastCollateMixup(config)
    elif config.train.collator.type == 'cutmix':
        return CutMixCollator(config)
    else:
        return None


def targets_to_device(config, targets, device):
    if config.train.collator.type == 'mixup' or config.train.collator.type =='cutmix':
        target_a, target_b, lam = targets
        targets = (target_a.to(device),
                   target_b.to(device),
                   lam)
    else:
        targets = targets.to(device)
    return targets

if __name__ == '__main__':
    from torchvision import transforms, datasets as ds
    import time
    from PIL import Image
    import torchvision as tv
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import cv2

    transforms_ = transforms.Compose([
        transforms.ToTensor(),
    ])
    to_pil = transforms.ToPILImage()

    data = tv.datasets.ImageFolder(root='/home/du/disk2/Desk/dataset/ibox/cls/dbox/dbox_dedup2/',
                                   transform=transforms_)
    loader = DataLoader(data,batch_size=1,shuffle=True)
    print(len(loader))

    t1 = None
    t2 = None
    for step, (image, label) in enumerate(loader):
        print(step)
        if step == 0:
            t1 = image
        if step == 1:
            t2 = image
        if step == 3:
            break

    # print(len(image))
    # print(t1.shape, t2)

    img1 = to_pil(t1[0])
    img2 = to_pil(t2[0])
    img1 = img1.resize((224,224))
    img2 = img2.resize((224,224))

    img1_ = cv2.cvtColor(np.asarray(img1), cv2.COLOR_RGB2BGR)
    img2_ = cv2.cvtColor(np.asarray(img2), cv2.COLOR_RGB2BGR)

    img = img1_ * 0.8 + 0.2 * img2_
    img = np.uint8(np.asarray(img))

    # img1.show()
    # img2.show()
    #
    # plt.imshow(img1)
    # plt.imshow(img2)
    # plt.show()

    print(img.shape)
    cv2.imshow('test', img)
    cv2.imshow('test1', img1_)
    cv2.imshow('test2', img2_)
    cv2.waitKey(0)


    # time.sleep(2)