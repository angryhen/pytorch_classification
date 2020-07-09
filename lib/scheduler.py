import torch


def warmup_schedule(step, warmup_init_lr, base_lr, warmup_stepss):
    lr = warmup_init_lr + (base_lr - warmup_init_lr) * (step / warmup_stepss)
    return lr


class MultiScheduler:
    def __init__(self, config, steps_per_epoch):
        self.warmup_epoches = config.train.scheduler.warmup_epoches
        self.warmup_steps = self.warmup_epoches * steps_per_epoch
        self.base_lr = config.train.scheduler.base_lr
        self.min_lr = config.train.scheduler.min_lr
        self.warmup_init_lr = config.train.scheduler.warmup_init_lr


    def __call__(self, step):
        if step < self.warmup_steps:
            lr = warmup_schedule(step,
                                 self.warmup_init_lr,
                                 self.base_lr,
                                 self.warmup_steps)
        else:
            lr = self.base_lr

        return lr


if __name__ == '__main__':
    from configs.config import get_cfg_defaults
    import matplotlib.pyplot as plt

    steps_per_epoches = 500
    total_step = 600
    config = get_cfg_defaults()

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

        x = model(torch.randn(3,3,64,64))
        loss = x.sum()
        loss.backward()
        optimizer.step()

        a.append(lr_scheduler.get_lr()[0])
        lr_scheduler.step()
    plt.plot(range(total_step), a)
    plt.show()