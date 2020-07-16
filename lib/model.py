import torch
import torch.nn as nn
from lib.use_model import choice_model, resume_custom


def get_model(config):
    model = choice_model(config.model.name, config.model.num_classes)
    if config.model.custom_pretrain:
        model = resume_custom(config, model)
    device = torch.device(config.device)
    model.to(device)

    return model