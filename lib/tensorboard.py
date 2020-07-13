from tensorboardX import SummaryWriter
import time

def get_tensorboard_writer(log_dir, purge_step):
    data_time = time.strftime("%Y-%m-%d %H:%M:%S")
    return SummaryWriter(log_dir+'/'+data_time, purge_step=purge_step)