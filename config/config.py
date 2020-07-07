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
config.train.base_lr = 0.0001