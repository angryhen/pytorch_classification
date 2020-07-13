import torch

def get_optimizer(config, model):
    if config.train.optimizer.method == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.train.scheduler.base_lr,
                                     weight_decay=config.train.scheduler.weight_decay,
                                     amsgrad=True)
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=config.train.scheduler.base_lr,
                                    weight_decay=config.train.scheduler.weight_decay,
                                    momentum=0.9,
                                    nesterov=True)
    return optimizer