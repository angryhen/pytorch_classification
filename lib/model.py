import torch
from alfred.utils.log import logger

from lib.use_model import choice_model, resume_custom
from utils.get_rank import get_rank


def get_model(config):
    model = choice_model(config.model.name, config.model.num_classes)
    if config.model.custom_pretrain:
        model = resume_custom(config, model)

    # print info
    if get_rank():
        get_model_info(model, config.model.name)

    # to cuda
    device = torch.device(config.device)
    model.to(device)

    return model


def get_model_info(model, name):
    params = sum([i.numel() for i in model.parameters()])
    logger.info(f'Model: {name} , '
                f'Total param: {params}')

    return params