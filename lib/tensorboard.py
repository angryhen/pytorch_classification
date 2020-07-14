from tensorboardX import SummaryWriter
import time

def get_tensorboard_writer(log_dir, purge_step):
    return SummaryWriter(log_dir, purge_step=purge_step)