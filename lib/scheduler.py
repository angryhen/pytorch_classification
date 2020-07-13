import torch
import numpy as np
from torch.optim.optimizer import Optimizer
from math import cos, pi


def warmup_schedule(step, warmup_init_lr, base_lr, warmup_stepss):
    lr = warmup_init_lr + (
            base_lr - warmup_init_lr) * (step / warmup_stepss)
    return lr


def cosine_scheduler(step, base_lr, min_lr, steps):
    lr = min_lr + (base_lr - min_lr) * (
            1 + np.cos(np.pi * (step / steps))) / 2

    return lr


class CosineWarmupLr:
    def __init__(self, config, optimizer, steps_per_epoch, last_iter=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_iter == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
            last_iter = 0
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError(
                        "param 'initial_lr' is not specified "
                        "in param_groups[{}] when resuming an optimizer".format(i))

        self.warmup_epoches = config.train.scheduler.warmup_epoches
        self.warmup_steps = self.warmup_epoches * steps_per_epoch
        self.base_lr = config.train.scheduler.base_lr
        self.min_lr = config.train.scheduler.min_lr
        self.warmup_init_lr = config.train.scheduler.warmup_init_lr
        self.total_steps = (config.train.epoches - self.warmup_epoches) * steps_per_epoch
        self.last_iter = last_iter
        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value for key,
                           value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        if self.last_iter <= self.warmup_steps:
            self.learning_rate = warmup_schedule(self.last_iter,
                                 self.warmup_init_lr,
                                 self.base_lr,
                                 self.warmup_steps)
        else:
            self.now_step = self.last_iter - self.warmup_steps
            self.learning_rate = cosine_scheduler(self.now_step, self.base_lr, self.min_lr, self.total_steps)

    def step(self, iteration=None):
        """Update status of lr.

        Args:
            iteration(int, optional): now training iteration of all epochs.
                Normally need not to set it manually.
        """
        if iteration is None:
            iteration = self.last_iter + 1
        self.last_iter = iteration
        self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate


class MultiScheduler:
    def __init__(self, config, steps_per_epoch):
        self.warmup_epoches = config.train.scheduler.warmup_epoches
        self.warmup_steps = self.warmup_epoches * steps_per_epoch
        self.base_lr = config.train.scheduler.base_lr
        self.min_lr = config.train.scheduler.min_lr
        self.warmup_init_lr = config.train.scheduler.warmup_init_lr
        self.total_steps = (config.train.epoches - self.warmup_epoches) * steps_per_epoch


    def __call__(self, step):
        if step <= self.warmup_steps:
            lr = warmup_schedule(step,
                                 self.warmup_init_lr,
                                 self.base_lr,
                                 self.warmup_steps)
        else:
            step -= self.warmup_steps
            lr = cosine_scheduler(step, self.base_lr, self.min_lr, self.total_steps)

        return lr


def get_scheduler(config, optimizer, steps_per_epoch):
    lr_scheduler = MultiScheduler(config, steps_per_epoch)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scheduler)


if __name__ == '__main__':
    from configs.config import get_cfg_defaults
    import matplotlib.pyplot as plt

    config = get_cfg_defaults()
    steps_per_epoches = 200
    total_step = config.train.epoches * steps_per_epoches

    model = torch.nn.Conv2d(3, 64, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)

    # lr = MultiScheduler(config, steps_per_epoches)
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr)
    lr_c = CosineWarmupLr(config, optimizer, steps_per_epoches)

    a = []
    for j in range(total_step):
        optimizer.zero_grad()

        # print(j, optimizer.param_groups[0]['lr'])
        print(lr_c.last_iter, lr_c.learning_rate)
        # print(j, lr_scheduler.get_lr()[0])
        # print(j, lr_scheduler.get_last_lr())

        x = model(torch.randn(3,3,64,64))
        loss = x.sum()
        loss.backward()
        optimizer.step()

        # a.append(lr_scheduler.get_lr()[0])
        a.append(lr_c.learning_rate)
        # lr_scheduler.step()
        lr_c.step()
    plt.plot(range(total_step), a)
    plt.show()