import torch.nn as nn
import torch
import torch.nn.functional as F


class LabelSmoothingCELoss(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, config):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCELoss, self).__init__()
        assert config.train.label_smooth < 1.0
        self.smoothing = config.train.label_smooth
        self.confidence = 1. - config.train.label_smooth
        print(f'using labelsmoothing ! smoothing: {config.train.label_smooth}')

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


@torch.no_grad()
def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method
    Warning: This function has no grad.
    """
    # assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))

    smooth_label = torch.empty(size=label_shape, device=true_labels.device)
    smooth_label.fill_(smoothing / (classes - 1))
    smooth_label.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return smooth_label

class LabelSmoothingCELoss_2(nn.Module):
    """This is label smoothing loss function.
    """

    def __init__(self, config, dim=-1):
        super(LabelSmoothingCELoss_2, self).__init__()
        self.confidence = 1.0 - config.train.label_smooth
        self.smoothing = config.train.label_smooth
        self.cls = config.model.num_classes
        self.dim = dim
        print(f'using labelsmoothing ! smoothing: {config.train.label_smooth}')

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        true_dist = smooth_one_hot(target, self.cls, self.smoothing)

        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
