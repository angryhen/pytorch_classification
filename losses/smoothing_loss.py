import torch
import torch.nn.functional as F


def onehot_encoding(label, n_classes):
    return torch.zeros(label.size(0), n_classes).to(label.device).scatter_(
        1, label.view(-1, 1), 1)


def cross_entropy_loss(data, target,
                       reduction):
    logp = F.log_softmax(data, dim=1)
    loss = torch.sum(-logp * target, dim=1)
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(
            '`reduction` must be one of \'none\', \'mean\', or \'sum\'.')


class LabelSmoothingLoss:
    def __init__(self, config, reduction):
        self.n_classes = config.model.num_classes
        self.epsilon = 0.1
        self.reduction = reduction

    def __call__(self, predictions,
                 targets):
        device = predictions.device

        onehot = onehot_encoding(
            targets, self.n_classes).type_as(predictions).to(device)
        targets = onehot * (1 - self.epsilon) + torch.ones_like(onehot).to(
            device) * self.epsilon / self.n_classes
        loss = cross_entropy_loss(predictions, targets, self.reduction)
        return loss