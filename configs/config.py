from yacs.config import CfgNode as CN

config = CN()
config.device = 'cuda'
config.apex = False
config.apex_mode = 'O1'  #  "O0", "O1", "O2", and "O3"

# dist
config.dist = False
config.dist_backend = 'nccl'
config.dist_init_method = 'env://'
config.dist_local_rank = 0


# model
config.model = CN()
config.model.name = 'resnest50'
config.model.num_classes = 15
config.model.save_path = 'logs'
config.model.checkpoint ='logs/resnext101/lowest.pth'
config.model.custom_pretrain = False
config.model.custom_checkpoint = 'weights/fold0_lowest_loss.pth'


# train
config.train = CN()
config.train.dataset = '/home/du/Desktop/work_project/TBox-server-dh/classification/ResNeSt/scripts/dbox_c15_dedup.csv'
config.train.img_size = 224
config.train.batch_size = 48
config.train.num_classes = 15
config.train.epoches = 50
config.train.mean = [0.485, 0.456, 0.406]
config.train.std = [0.229, 0.224, 0.225]
config.train.fold = 1

config.train.checkpoint = f'logs/{config.model.name}'
config.train.subdivision = 1
config.train.preiod = 10
config.train.val_preiod = 1
config.train.label_smooth = False
config.train.seed = 0

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
config.val.log_file = 'logs'

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
config.test.dataloader.pin_memory = False  # False when memory not enough
config.test.dataloader.non_blocking = True

# tensorboard
config.tensorboard = CN()
config.tensorboard.log_dir = 'logs'

# label
# config.labels = {0: 'hzy-hzy-pz-bxgw-500ml', 1: 'ty-hzy-pz-gw-500ml', 2: 'ty-tyxmtx-pz-lplldc-480ml', 3: 'ty-tylythsnr-tz-hsnrw-105g', 4: 'tdr-tdrstglm-tz-yw-83g', 5: 'ty-tyxmtx-pz-lpqnhc-480ml', 6: 'asm-asmnc-pz-yw-500ml', 7: 'ty-qbsds-pz-nmw-500ml', 8: 'ty-tync-pz-mxw-310ml', 9: 'yh-yhbkf-pz-yw-450ml', 10: 'ty-tybhc-pz-nmw-500ml', 11: 'ty-tylc-pz-mlw-500ml', 12: 'ty-tyhsnrm-tz-nr-105g', 13: 'tdr-tdrsslltgm-tz-yw-90g', 14: 'ty-tyhld-gz-xclc-310ml'}
config.labels_list = ['hzy-hzy-pz-bxgw-500ml', 'ty-hzy-pz-gw-500ml', 'ty-tyxmtx-pz-lplldc-480ml', 'ty-tylythsnr-tz-hsnrw-105g', 'tdr-tdrstglm-tz-yw-83g', 'ty-tyxmtx-pz-lpqnhc-480ml', 'asm-asmnc-pz-yw-500ml', 'ty-qbsds-pz-nmw-500ml', 'ty-tync-pz-mxw-310ml', 'yh-yhbkf-pz-yw-450ml', 'ty-tybhc-pz-nmw-500ml', 'ty-tylc-pz-mlw-500ml', 'ty-tyhsnrm-tz-nr-105g', 'tdr-tdrsslltgm-tz-yw-90g', 'ty-tyhld-gz-xclc-310ml']


def get_cfg_defaults():
    return config.clone()
