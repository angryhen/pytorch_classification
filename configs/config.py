from yacs.config import CfgNode as CN

config = CN()
config.device = 'cuda'
config.apex = False
config.apex_mode = 'O1'  #  "O0", "O1", "O2", and "O3"

config.log_dir = 'logs'

# dist
config.dist = False
config.dist_backend = 'nccl'
config.dist_init_method = 'env://'
config.dist_local_rank = 0
config.dist_sync_bn = False

# model
config.model = CN()
config.model.name = 'resnest50'
config.model.num_classes = 6
config.model.checkpoint ='logs/resnext101/lowest.pth'
config.model.custom_pretrain = False
config.model.custom_checkpoint = 'weights/fold0_lowest_loss.pth'


# train
config.train = CN()
config.train.dataset = '/home/du/Desktop/work_project/TBox-server-dh/classification/ResNeSt/scripts/tbox_shape_train.csv'
config.train.img_size = 224
config.train.batch_size = 48
config.train.num_classes = 6
config.train.epoches = 50
config.train.mean = [0.485, 0.456, 0.406]
config.train.std = [0.229, 0.224, 0.225]
config.train.fold = 1

config.train.checkpoint = f'logs/{config.model.name}'
config.train.subdivision = 1
config.train.preiod = 10
config.train.val_preiod = 1
config.train.label_smooth = 0.
config.train.seed = 0

# train -- optimizer
config.train.optimizer = CN()
config.train.optimizer.method = 'adam'  # ['adam', 'sgd']

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
config.val.dataloader.non_blocking = True

# test
config.test = CN()
config.test.dataset = '/home/du/Desktop/work_project/TBox-server-dh/classification/ResNeSt/scripts/new_val_dbox.csv'
config.test.batch_size = 64 * 4
config.test.img_size = 224
config.test.mean = [0.485, 0.456, 0.406]
config.test.std = [0.229, 0.224, 0.225]
config.test.log_file = 'logs/test_result.txt'

# test --dataloader
config.test.dataloader = CN()
config.test.dataloader.work_nums = 8
config.test.dataloader.drop_last = False
config.test.dataloader.pin_memory = True  # False when memory not enough
config.test.dataloader.non_blocking = True

# tensorboard
config.tensorboard = CN()

# label
# config.labels = {}
labels = {0: 'tz', 1: 'dz', 2: 'pz', 3: 'hz', 4: 'bz', 5: 'gz'}

config.labels_list = ['tz', 'dz', 'pz', 'hz', 'bz', 'gz']



def get_cfg_defaults():
    return config.clone()
