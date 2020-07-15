import torch
import torch.nn as nn
from lib.use_model import choice_model, resume_custom


def get_model(config, local_rank):
    model = choice_model(config.model.name, config.model.num_classes)
    if config.model.custom_pretrain:
        model = resume_custom(config, model)
    device = torch.device(config.device)
    model.to(device)
    if config.dist:
        if config.dist_sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model,
                                                    device_ids=[local_rank],
                                                    output_device=local_rank)
    return model