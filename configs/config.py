from yacs.config import CfgNode as CN

config = CN()

# model
config.model = CN()
config.model.name = 'resnest50'
config.model.num_classes = 15

# train
config.train = CN()
config.train.dataset = '/home/du/Desktop/work_project/TBox-server-dh/classification/ResNeSt/scripts/dbox_c15_dedup.csv'
config.train.img_size = 224
config.train.batch_size = 64
config.train.epoches = 30
config.train.mean = [0.485, 0.456, 0.406]
config.train.std = [0.229, 0.224, 0.225]
config.train.checkpoint = f'logs/{config.model.name}'

# train -- optimizer
config.train.optimizer = CN()
config.train.optimizer.method = 'adam'

# train -- scheduler
config.train.scheduler = CN()
config.train.scheduler.method = 'cosine'  # (method: cosine, ...)
config.train.scheduler.base_lr = 1e-4
config.train.scheduler.min_lr = 1e-6
config.train.scheduler.lr_decay = 0.95
config.train.scheduler.weight_decay = 1e-4
config.train.scheduler.warmup_epoches = 1  # set 0 when no use warmup
config.train.scheduler.warmup_init_lr = 1e-7

# train -- dataloader
config.train.dataloader = CN()
config.train.dataloader.work_nums = 10
config.train.dataloader.shuffle = True
config.train.dataloader.drop_last = True
config.train.dataloader.pin_memory = True  # False when memory not enough
config.train.dataloader.non_blocking = True  # False when memory not enough
config.train.dataloader.collate_fn = True

# valid
config.val = CN()
config.val.batch_size = 64

# valid --dataloader
config.val.dataloader = CN()
config.val.dataloader.work_nums = 4
config.val.dataloader.drop_last = False
config.train.dataloader.pin_memory = False  # False when memory not enough


# tensorboard
config.tensorboard = CN()

# model
config.model = CN()
config.model.name = 'resnest50'


def get_cfg_defaults():
    return config.clone()
