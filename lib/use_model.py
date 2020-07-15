import torch
import torch.nn as nn
from model.resnest.torch import resnest50, resnest101, resnest200, resnest269


class ResNest_50(nn.Module):
    def __init__(self, pretrained, num_classes):
        super(ResNest_50, self).__init__()
        self.model = resnest50(pretrained=pretrained)
        self.model.fc = nn.Linear(2048, num_classes, bias=True)

    def forward(self, img):
        output = self.model(img)
        return output


class ResNest_101(nn.Module):
    def __init__(self, pretrained, num_classes):
        super(ResNest_101, self).__init__()
        self.model = resnest50(pretrained=pretrained)
        self.model.fc = nn.Linear(2048, num_classes, bias=True)

    def forward(self, img):
        output = self.model(img)
        return output


class ResNest_200(nn.Module):
    def __init__(self, pretrained, num_classes):
        super(ResNest_200, self).__init__()
        self.model = resnest200(pretrained=pretrained)
        self.model.fc = nn.Linear(2048, num_classes, bias=True)

    def forward(self, img):
        output = self.model(img)
        return output


class ResNest_269(nn.Module):
    def __init__(self, pretrained, num_classes):
        super(ResNest_269, self).__init__()
        self.model = resnest50(pretrained=pretrained)
        self.model.fc = nn.Linear(2048, num_classes, bias=True)

    def forward(self, img):
        output = self.model(img)
        return output


class ResNesXt_101(nn.Module):
    def __init__(self, pretrained, num_classes):
        super(ResNesXt_101, self).__init__()
        self.model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
        self.model.fc = nn.Linear(2048, num_classes, bias=True)

    def forward(self, img):
        output = self.model(img)
        return output


def choice_model(flag, num_classes):
    if flag == 'resnest50':
        return ResNest_50(pretrained=True, num_classes=num_classes)
    elif flag == 'resnest101':
        return ResNest_101(pretrained=True, num_classes=num_classes)
    elif flag == 'resnest200':
        return ResNest_200(pretrained=True, num_classes=num_classes)
    elif flag == 'resnest269':
        return ResNest_269(pretrained=True, num_classes=num_classes)
    elif flag == 'resnext101':
        return ResNesXt_101(pretrained=True, num_classes=num_classes)

def resume_custom(config, model):
    ch = torch.load(config.model.custom_checkpoint)
    ch['state_dict'].pop('model.fc.weight')
    ch['state_dict'].pop('model.fc.bias')
    model_dict = model.state_dict()
    model_dict.update(ch['state_dict'])
    model.load_state_dict(model_dict)
    return model