from yacs.config import CfgNode as CN

config = CN()
config.device = 'cuda'
config.apex = False
config.apex_mode = 'O1'  #  "O0", "O1", "O2", and "O3"

# model
config.model = CN()
config.model.name = 'resnest50'
config.model.num_classes = 15
config.model.save_path = 'logs'

# train
config.train = CN()
config.train.dataset = '/home/du/Desktop/work_project/TBox-server-dh/classification/ResNeSt/scripts/new_c15_all.csv'
config.train.img_size = 224
config.train.batch_size = 64
config.train.num_classes = 15
config.train.epoches = 50
config.train.mean = [0.485, 0.456, 0.406]
config.train.std = [0.229, 0.224, 0.225]
config.train.checkpoint = f'logs/{config.model.name}'
config.train.fold = 1
config.train.subdivision = 1
config.train.preiod = 10


# train -- optimizer
config.train.optimizer = CN()
config.train.optimizer.method = 'adam'

# train -- scheduler
config.train.scheduler = CN()
config.train.scheduler.method = 'cosine'  # (method: cosine, ...)
config.train.scheduler.base_lr = 1e-4
config.train.scheduler.min_lr = 1e-7
config.train.scheduler.lr_decay = 0.95
config.train.scheduler.weight_decay = 0.0001
config.train.scheduler.warmup_epoches = 1  # set 0 for no use warmup
config.train.scheduler.warmup_init_lr = 1e-7

# train -- dataloader
config.train.dataloader = CN()
config.train.dataloader.work_nums = 10
config.train.dataloader.shuffle = True
config.train.dataloader.drop_last = True
config.train.dataloader.pin_memory = True  # False when memory not enough
config.train.dataloader.non_blocking = True  # False when memory not enough

# train -- collator
config.train.collator = CN()
config.train.collator.type = 'None'  # ['mixup', 'mixup2', 'cutmix', None]
config.train.collator.mixup_alpha = 1
config.train.collator.cutmix_alpha = 1.0


# valid
config.val = CN()
config.val.batch_size = 64
config.val.img_size = 224
config.val.mean = [0.485, 0.456, 0.406]
config.val.std = [0.229, 0.224, 0.225]

# valid --dataloader
config.val.dataloader = CN()
config.val.dataloader.work_nums = 8
config.val.dataloader.drop_last = False
config.val.dataloader.pin_memory = False  # False when memory not enough

# test
config.test = CN()
config.test.dataset = '/home/du/Desktop/work_project/TBox-server-dh/classification/ResNeSt/scripts/new_val_dbox.csv'
config.test.batch_size = 64
config.test.img_size = 224
config.test.mean = [0.485, 0.456, 0.406]
config.test.std = [0.229, 0.224, 0.225]

# test --dataloader
config.test.dataloader = CN()
config.test.dataloader.work_nums = 8
config.test.dataloader.drop_last = False
config.test.dataloader.pin_memory = False  # False when memory not enough

# tensorboard
config.tensorboard = CN()
config.tensorboard.log_dir = 'logs'


def get_cfg_defaults():
    return config.clone()
