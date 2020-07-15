import torch

def get_env_info():
    info = {}
    if torch.cuda.is_available():
        info['gpus_number'] = torch.cuda.device_count()
        info['gpus_name'] = torch.cuda.get_device_name()

    return info
