import torch
import numpy as np


def warmup_schedule(step, warmup_init_lr, base_lr, warmup_stepss):
    lr = warmup_init_lr + (base_lr - warmup_init_lr) * (step / warmup_stepss)
    return lr


def cosine_scheduler(step, base_lr, min_lr, steps):
    lr = base_lr * (min_lr +
                    (1 - min_lr) * 0.5 *
                     (1 + np.cos(step / steps * np.pi)))
    return lr


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


if __name__ == '__main__':
    from configs.config import get_cfg_defaults
    import matplotlib.pyplot as plt

    config = get_cfg_defaults()
    steps_per_epoches = 200
    total_step = config.train.epoches * steps_per_epoches

    lr = MultiScheduler(config, steps_per_epoches)

    # lr_list = []
    # for i in range(total_step):
    #     lr_list.append(lr(i))
    # plt.plot(range(total_step), lr_list)
    # plt.show()

    model = torch.nn.Conv2d(3,64,3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1, amsgrad=True)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr)
    a = []
    for j in range(total_step):
        optimizer.zero_grad()

        print(j, optimizer.param_groups[0]['lr'])
        print(j, lr_scheduler.get_lr()[0])
        print(j, lr_scheduler.get_last_lr())

        x = model(torch.randn(3,3,64,64))
        loss = x.sum()
        loss.backward()
        optimizer.step()

        a.append(lr_scheduler.get_lr()[0])
        lr_scheduler.step()
    plt.plot(range(total_step), a)
    plt.show()