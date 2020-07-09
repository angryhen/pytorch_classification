import torch.nn as nn
from model.resnest.torch import resnest50, resnest101, resnest200, resnest269


class ResNest_50(nn.Module):
    def __init__(self, pretrained, num_classes):
        super(ResNest_50, self).__init__()
        model = resnest50(pretrained=pretrained)
        model.fc = nn.Sequential(
            nn.Linear(2048, num_classes, bias=True),
        )
        self.net = model

    def forward(self, img):
        output = self.net(img)
        return output


class ResNest_101(nn.Module):
    def __init__(self, pretrained, num_classes):
        super(ResNest_101, self).__init__()
        model = resnest101(pretrained=pretrained)
        model.fc = nn.Sequential(
            nn.Linear(2048, num_classes, bias=True),
        )
        self.net = model

    def forward(self, img):
        output = self.net(img)
        return output


class ResNest_200(nn.Module):
    def __init__(self, pretrained, num_classes):
        super(ResNest_200, self).__init__()
        model = resnest200(pretrained=pretrained)
        model.fc = nn.Sequential(
            nn.Linear(2048, num_classes, bias=True),
        )
        self.net = model

    def forward(self, img):
        output = self.net(img)
        return output


class ResNest_269(nn.Module):
    def __init__(self, pretrained, num_classes):
        super(ResNest_269, self).__init__()
        model = resnest269(pretrained=pretrained)
        model.fc = nn.Sequential(
            nn.Linear(2048, num_classes, bias=True),
        )
        self.net = model

    def forward(self, img):
        output = self.net(img)
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