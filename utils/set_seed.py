import random
import numpy as np
import torch


def set_seed(config):
    seed = config.train.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)